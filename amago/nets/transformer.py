import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange, einsum
import gin

from .utils import activation_switch
from amago.utils import amago_warning

try:
    import flash_attn
except ImportError:
    amago_warning("Missing FlashAttention (2.0) Install")
from typing import Optional, Tuple, Union, List, Dict
from random import random as rand

class Normalization(nn.Module):
    def __init__(self, method: str, d_model: int):
        super().__init__()
        assert method in ["layer", "none"]
        if method == "layer":
            self.norm = nn.LayerNorm(d_model)
        elif method == "none":
            self.norm = lambda x: x
        self.method = method

    def forward(self, x):
        return self.norm(x)


@gin.configurable(allowlist=["window_size"])
class FlashAttention(nn.Module):
    def __init__(
        self,
        causal: bool = True,
        attention_dropout: float = 0.0,
        window_size: tuple[int, int] = (-1, -1),
    ):
        super().__init__()
        self.dropout = attention_dropout
        self.causal = causal
        self.window_size = window_size

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = qkv.to(torch.bfloat16)
        if key_cache is None or val_cache is None or cache_seqlens is None:
            out = flash_attn.flash_attn_qkvpacked_func(
                qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal,
                window_size=self.window_size,
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k_cache=key_cache,
                v_cache=val_cache,
                cache_seqlens=cache_seqlens,
                k=k,
                v=v,
                causal=self.causal,
                window_size=self.window_size,
            )
        return out


class SigmaReparam(nn.Module):
    """ "
    https://arxiv.org/pdf/2303.06296.pdf Appendix C
    """

    def __init__(self, d_in, d_out, bias: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.randn(d_out, d_in), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(d_out), requires_grad=True) if bias else None
        u = torch.randn(d_out)
        self.u = nn.Parameter(u / u.norm(dim=0), requires_grad=False)
        v = torch.randn(d_in)
        self.v = nn.Parameter(v / v.norm(dim=0), requires_grad=False)
        self.gamma = nn.Parameter(torch.ones(1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        # same as nn.Linear
        nn.init.kaiming_uniform_(self.W, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.W)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x):
        if self.training:
            with torch.no_grad():
                u = (self.W @ self.v).float()
                self.u.data = u / u.norm(dim=0)
                v = (self.W.T @ self.u).float()
                self.v.data = v / v.norm(dim=0)
        sigma = einsum(self.u, self.W, self.v, "d, d c , c->")
        W_hat = self.gamma / sigma * self.W
        out = F.linear(x, W_hat, self.b)
        return out


class VanillaAttention(nn.Module):
    def __init__(self, causal: bool = True, attention_dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.causal = causal
        self._mask = None

    def forward(self, qkv, key_cache=None, val_cache=None, cache_seqlens=None):
        assert (
            key_cache is None and val_cache is None
        ), "VanillaAttention does not support `fast_inference` mode"
        queries, keys, values = torch.unbind(qkv, dim=2)
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1.0 / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self._mask is None or self._mask.shape != (B, 1, L, L):
            self._mask = torch.triu(
                torch.ones((B, 1, L, L), dtype=torch.bool, device=qkv.device),
                diagonal=1,
            )
        if self.causal:
            scores.masked_fill_(self._mask, -torch.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V


@gin.configurable(allowlist=["head_scaling", "sigma_reparam"])
class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_qkv,
        n_heads,
        dropout_qkv=0.0,
        head_scaling: bool = True,
        sigma_reparam: bool = True,
    ):
        super().__init__()
        self.attention = attention
        FF = SigmaReparam if sigma_reparam else nn.Linear
        self.qkv_projection = FF(d_model, 3 * d_qkv * n_heads, bias=False)
        self.dropout_qkv = nn.Dropout(dropout_qkv)
        self.out_projection = FF(d_qkv * n_heads, d_model)
        self.head_scaler = nn.Parameter(
            torch.ones(1, 1, n_heads, 1), requires_grad=head_scaling
        )
        self.n_heads = n_heads

    def forward(self, sequence, key_cache=None, val_cache=None, cache_seqlens=None):
        qkv = self.dropout_qkv(self.qkv_projection(sequence))
        qkv = rearrange(
            qkv,
            "batch len (three d_qkv heads) -> batch len three heads d_qkv",
            heads=self.n_heads,
            three=3,
        )
        out = self.head_scaler * self.attention(
            qkv=qkv,
            key_cache=key_cache,
            val_cache=val_cache,
            cache_seqlens=cache_seqlens,
        )
        out = rearrange(out, "batch len heads dim -> batch len (heads dim)")
        out = self.out_projection(out)
        return out


@gin.configurable(denylist=["activation", "norm", "dropout_ff"])
class TransformerLayer(nn.Module):
    """
    Pre-Norm Self-Attention
    """

    def __init__(
        self,
        self_attention,
        d_model: int,
        d_ff: int,
        dropout_ff: float = 0.1,
        activation: str = "leaky_relu",
        norm: str = "layer",
        sigma_reparam: bool = True,
        normformer_norms: bool = True,
    ):
        super().__init__()
        self.self_attention = self_attention
        FF = SigmaReparam if sigma_reparam else nn.Linear
        self.ff1 = FF(d_model, d_ff)
        self.ff2 = FF(d_ff, d_model)
        self.norm1 = Normalization(method=norm, d_model=d_model)
        self.norm2 = (
            Normalization(method=norm, d_model=d_model)
            if normformer_norms
            else lambda x: x
        )
        self.norm3 = Normalization(method=norm, d_model=d_model)
        self.norm4 = (
            Normalization(method=norm, d_model=d_ff)
            if normformer_norms
            else lambda x: x
        )
        self.dropout_ff = nn.Dropout(dropout_ff)
        self.activation = activation_switch(activation)

    def forward(self, self_seq, key_cache=None, val_cache=None, cache_seqlens=None):
        q1 = self.norm1(self_seq)  # pre-norm
        q1 = self.self_attention(
            q1, key_cache=key_cache, val_cache=val_cache, cache_seqlens=cache_seqlens
        )
        q1 = self.norm2(q1)  # normformer extra norm 1
        self_seq = self_seq + q1
        q1 = self.norm3(self_seq)  # regular norm
        # normformer extra norm 2
        q1 = self.norm4(self.activation(self.ff1(q1)))
        q1 = self.dropout_ff(self.ff2(q1))
        self_seq = self_seq + q1
        return self_seq


class Cache:
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
    ):
        self.data = torch.zeros(
            (batch_size, max_seq_len, n_heads, head_dim), dtype=dtype, device=device
        )
        # make silent bugs in k/v cache... much louder
        self.data[:] = torch.nan

    def __len__(self):
        return self.data.shape[1]

    def roll_back(self, idx):
        roll = self.data[idx, 1:].clone()
        self.data[idx, :-1] = roll
        self.data[idx, -1] = torch.nan  # no silent bugs

    def delete_n(self, idx, n, timestep):
        # some index related asserts
        self.data[idx, timestep-n : timestep] = torch.nan  # no silent bugs


class TformerHiddenState:
    def __init__(
        self, key_cache: list[Cache], val_cache: list[Cache], timesteps: torch.Tensor
    ):
        assert isinstance(key_cache, list) and len(key_cache) == len(val_cache)
        assert timesteps.dtype == torch.int32
        self.n_layers = len(key_cache)
        self.key_cache = key_cache
        self.val_cache = val_cache
        self.timesteps = timesteps
        self.max_seq_len_tracker = 0
        
    def reset(self, idxs):
        self.timesteps[idxs] = 0

    def reset_and_clear_all(self):
        self.timesteps[:] = 0
        for k, v in zip(self.key_cache, self.val_cache):
            k.data[:] = torch.nan
            v.data[:] = torch.nan

    def update(self):
        if self.timesteps[0].item() + 1 > self.max_seq_len_tracker:
            self.max_seq_len_tracker = self.timesteps[0].item() +1
        self.timesteps += 1
        for i, timestep in enumerate(self.timesteps):
            if timestep == len(self.key_cache[0]):
                for k, v in zip(self.key_cache, self.val_cache):
                    k.roll_back(i)
                    v.roll_back(i)
                self.timesteps[i] -= 1

    def __getitem__(self, layer_idx):
        assert layer_idx < self.n_layers
        return (
            self.key_cache[layer_idx].data,
            self.val_cache[layer_idx].data,
            self.timesteps,
        )

    def update_n(self, remove=0, add=0):
        if self.timesteps[0].item() + add > self.max_seq_len_tracker:
            self.max_seq_len_tracker = self.timesteps[0].item() + add

        self.timesteps += add
        if remove > 0:
            for i, timestep in enumerate(self.timesteps):
                for k, v in zip(self.key_cache, self.val_cache):
                    k.delete_n(i, remove, timestep)
                    v.delete_n(i, remove, timestep)
                self.timesteps[i] -= remove

    def update_n_var_eps_lens(self, remove=0, add=0, seg_len=0, summary_len=0):
        if self.timesteps[0].item() + add > self.max_seq_len_tracker:
            self.max_seq_len_tracker = self.timesteps[0].item() + add

        self.timesteps += add
        if remove > 0:
            for i, timestep in enumerate(self.timesteps):
                if timestep < seg_len + summary_len:
                    remove_val = timestep
                else:
                    remove_val = remove
                for k, v in zip(self.key_cache, self.val_cache):
                    k.delete_n(i, remove_val, timestep)
                    v.delete_n(i, remove_val, timestep)
                self.timesteps[i] -= remove_val

class FixedPosEmb(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, pos_idxs: torch.LongTensor):
        B, L = pos_idxs.shape
        emb = torch.zeros(
            (B, L, self.d_model), device=pos_idxs.device, dtype=torch.float32
        )
        coeff = torch.exp(
            (
                torch.arange(0, self.d_model, 2, device=emb.device, dtype=torch.float32)
                * -(math.log(10000.0) / self.d_model)
            )
        )
        emb[..., 0::2] = torch.sin(pos_idxs.float().unsqueeze(-1) * coeff)
        emb[..., 1::2] = torch.cos(pos_idxs.float().unsqueeze(-1) * coeff)
        return emb


class Transformer(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        max_pos_idx: int,
        d_model: int = 128,
        d_ff: int = 512,
        d_emb_ff: int = None,
        n_heads: int = 4,
        layers: int = 3,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        attention: str = "flash",
        activation: str = "leaky_relu",
        norm: str = "layer",
        causal: bool = True,
        pos_emb: str = "learnable",
    ):
        super().__init__()
        assert attention in ["flash", "vanilla"]
        assert pos_emb in ["learnable", "fixed"]

        # embedding
        if pos_emb == "learnable":
            self.position_embedding = nn.Embedding(
                max_pos_idx + 1, embedding_dim=d_model
            )
        elif pos_emb == "fixed":
            self.position_embedding = FixedPosEmb(d_model)
        d_emb_ff = d_emb_ff or d_model
        self.inp = nn.Linear(inp_dim, d_model)
        self.dropout = nn.Dropout(dropout_emb)

        self.head_dim = d_model // n_heads
        assert self.head_dim in range(8, 129, 8)
        self.n_heads = n_heads
        self.n_layers = layers
        Attn = FlashAttention if attention == "flash" else VanillaAttention

        def make_layer():
            return TransformerLayer(
                self_attention=AttentionLayer(
                    attention=Attn(causal=causal, attention_dropout=dropout_attn),
                    d_model=d_model,
                    d_qkv=self.head_dim,
                    n_heads=self.n_heads,
                    dropout_qkv=dropout_qkv,
                ),
                d_model=d_model,
                d_ff=d_ff,
                dropout_ff=dropout_ff,
                activation=activation,
                norm=norm,
            )

        self.layers = nn.ModuleList([make_layer() for _ in range(layers)])
        self.norm = Normalization(method=norm, d_model=d_model)
        self.d_model = d_model

    @property
    def emb_dim(self):
        return self.d_model

    def forward(self, seq, pos_idxs, hidden_state: None | TformerHiddenState):
        if self.training:
            assert hidden_state is None
        batch, length, dim = seq.shape
        h = hidden_state or [[None, None, None] for _ in range(self.n_layers)]

        # emedding
        pos_emb = self.position_embedding(pos_idxs)
        traj_emb = self.inp(seq)
        traj_emb = self.dropout(traj_emb + pos_emb)

        # self-attention
        for i, layer in enumerate(self.layers):
            traj_emb = layer(traj_emb, *h[i])
        traj_emb = self.norm(traj_emb)

        if hidden_state is not None:
            # controls the sequence length of the k/v cache
            hidden_state.update()

        return traj_emb, hidden_state


class AutoCompressorTransformer(Transformer):
    """Mixin class to turn a AutoModelForCausalLM into an AutoCompressor."""

    def __init__(
        self,
        inp_dim: int,
        max_pos_idx: int,
        d_model: int = 128,
        d_ff: int = 512,
        d_emb_ff: int = None,
        n_heads: int = 4,
        layers: int = 3,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        attention: str = "flash",
        activation: str = "leaky_relu",
        norm: str = "layer",
        causal: bool = True,
        pos_emb: str = "learnable",
        accumulate_summary: bool = True,
        segment_gradient_checkpointing: bool = False,
        summary_length: int = 0,
        segment_lengths: int = 0,
        detach_prob: float = 0.0,
    ):

        super().__init__(
            inp_dim=inp_dim,
            max_pos_idx=max_pos_idx,
            d_model=d_model,
            d_ff=d_ff,
            d_emb_ff=d_emb_ff,
            n_heads=n_heads,
            layers=layers,
            dropout_emb=dropout_emb,
            dropout_ff=dropout_ff,
            dropout_attn=dropout_attn,
            dropout_qkv=dropout_qkv,
            attention=attention,
            activation=activation,
            norm=norm,
            causal=causal,
            pos_emb=pos_emb,
        )

        self.accumulate_summary = accumulate_summary
        self.segment_gradient_checkpointing = segment_gradient_checkpointing
        self.summary_length = summary_length
        self.segment_lengths = segment_lengths
        self.detach_prob = detach_prob

        print("SUMMARY LENGTH: ", self.summary_length)
        print("SEGMENT LENGTHS: ", self.segment_lengths)

        assert self.segment_lengths > self.summary_length
        assert self.segment_lengths > 0
        assert self.summary_length >= 0

        self.interaction_step = 0

        # assert self.segment_lngths > self.summary_length
        if summary_length > 0:
            self.embed_summary = nn.Embedding(summary_length, self.d_model)

    """
    this class currently only works for constant length episodes - we need the same 
    """

    def forward(self, seq, pos_idxs, hidden_state: None | TformerHiddenState):
        batch, length, dim = seq.shape

        # if we are in train mode or are getting a batch of sequences of data points rather than a single step
        if self.training or pos_idxs.size(1) > 1:

            min_val = self.summary_length*80

            # we want to sample a segment length between 1-self.segment_lengths, with a bias towards the upper end, which is the segment length
            sampled_segment_length = min_val + (self.segment_lengths + 1 - min_val) * pow(rand(), 0.5)
            sampled_segment_length = int(sampled_segment_length)
            # print(sampled_segment_length)
            # round this to the nearest power of 2 loweer than the sampled segment length
            # sampled_segment_length = 2 ** int(math.log2(sampled_segment_length))
            # sampled_segment_length = self.segment_lengths if rand() > 0.3 else self.segment_lengths//2


            # sampled_segment_length = int(self.segment_lengths)
            assert hidden_state is None

            h = hidden_state or [[None, None, None] for _ in range(self.n_layers)]

            traj_emb = self.inp(seq)
            traj_embs = []
            # ipdb.set_trace()
            seq_segments = torch.split(traj_emb, sampled_segment_length, dim=1)
            softprompt = None

            per_segment_summary_lens = []
            for i, segment in enumerate(seq_segments):
                if segment.size(1) < sampled_segment_length:
                    per_segment_summary_lens.append(0)
                else:
                    per_segment_summary_lens.append(self.summary_length)

            for i, segment in enumerate(seq_segments):

                segments_len = per_segment_summary_lens[i]
                cumsum_segment_lens = sum(per_segment_summary_lens[:i+1])

                segment_pos_idxs = torch.arange( segment.size(1) + cumsum_segment_lens, device=pos_idxs.device).unsqueeze(0).expand(segment.size(0), -1)

                summary_token_ids = torch.arange(segments_len, dtype=torch.long, device=traj_emb.device).unsqueeze(0).expand(traj_emb.size(0), -1)
                summary_token_embeds = self.embed_summary(summary_token_ids).to(traj_emb.dtype)
                if softprompt is None:
                    segment_embed = torch.cat([segment, summary_token_embeds], dim=1)
                else:
                    segment_embed = torch.cat([softprompt, segment, summary_token_embeds], dim=1)

                segment_embed = self.dropout(segment_embed + self.position_embedding(segment_pos_idxs))

                # self-attention
                for j, layer in enumerate(self.layers):
                    segment_embed = layer(segment_embed, *h[j])

                traj_emb_i = self.norm(segment_embed)
                # ignore the first summary_length tokens and get the middle segment embeddings, use the last summary_ln tokens as the softprompt

                if segments_len > 0:
                    new_softprompt = traj_emb_i[:, -self.summary_length:]
                    traj_emb_i = traj_emb_i[:, i*self.summary_length:-self.summary_length]
                    if self.accumulate_summary and softprompt is not None:
                        if torch.rand(1) < self.detach_prob:
                            new_softprompt = new_softprompt.detach()
                        softprompt = torch.cat([softprompt, new_softprompt], dim=1)
                    else:
                        if torch.rand(1) < self.detach_prob:
                            softprompt = new_softprompt.detach()
                        else:
                            softprompt = new_softprompt
                else:
                    traj_emb_i = traj_emb_i[:, i*self.summary_length:]

                traj_embs.append(traj_emb_i)

            traj_emb = torch.cat(traj_embs, dim=1)

        else:
            rollout_idx = self.interaction_step  
            h = hidden_state

            index_within_segment = pos_idxs % self.segment_lengths
            softprompt_pos_offset = (pos_idxs // self.segment_lengths) * self.summary_length
            segment_pos_idxs = index_within_segment + softprompt_pos_offset

            traj_emb = self.inp(seq)

            if (rollout_idx + 1) % self.segment_lengths == 0:
                # if it is empty or zero, there is a bug since the code is expoecting to have collected segment_length worth of previous hidden states
                assert torch.all(hidden_state.timesteps > 0)

                # now we also want to gebnerate a new summary so we will add summary_length tokens to the end of the segment
                summary_token_ids = torch.arange(self.summary_length, dtype=torch.long, device=traj_emb.device).unsqueeze(0).expand(traj_emb.size(0), -1)
                summary_token_embeds = self.embed_summary(summary_token_ids).to(traj_emb.dtype)
                segment_pos_idxs = torch.cat([segment_pos_idxs, torch.arange(self.summary_length, device=pos_idxs.device).unsqueeze(0).expand(traj_emb.size(0), -1)], dim=1)
                traj_emb = torch.cat([traj_emb, summary_token_embeds], dim=1)

            traj_emb = self.dropout(traj_emb + self.position_embedding(segment_pos_idxs))

            # self-attention
            for i, layer in enumerate(self.layers):
                traj_emb = layer(traj_emb, *hidden_state[i])
            traj_emb = self.norm(traj_emb)



            if (rollout_idx + 1) % self.segment_lengths == 0:
                # ignore the first summary_length tokens and get the middle segment embeddings, use the last summary_ln tokens as the softprompt
                new_softprompt_inp_embeds = traj_emb[:, -self.summary_length:]
                traj_emb = traj_emb[:, :-self.summary_length]

                # now do a forward pass on just the embedding outputs from the summary tokens to get the new softprompt, use pos_idxs correspondng to how many softprompts vecs were collected previously
                new_softprompt_pos_idxs = torch.arange(self.summary_length, device=pos_idxs.device).unsqueeze(0).expand(traj_emb.size(0), -1) + softprompt_pos_offset

                # new_softprompt = self.inp(new_softprompt_inp_embeds)                                                                                          ## TODO:  this is needed or not
                new_softprompt = self.dropout(new_softprompt_inp_embeds + self.position_embedding(new_softprompt_pos_idxs))

                # update the cache by removing previous inputs within the segment as well as the cache created when making summary input embeddings,
                # to only retain things up till the previous softprompt
                hidden_state.update_n_var_eps_lens(
                    add=self.summary_length+1,
                    remove=self.segment_lengths + self.summary_length,
                    seg_len=self.segment_lengths,
                    summary_len=self.summary_length,
                )

                try:
                    assert torch.all(torch.isnan(hidden_state.key_cache[0].data[:, -self.summary_length:]))
                except:
                    ipdb.set_trace()
                for i, layer in enumerate(self.layers):
                    new_softprompt = layer(new_softprompt, *hidden_state[i])                            # TODO assert that hidden state only has num_softprompts * summary_len entries    
                new_softprompt = self.norm(new_softprompt)

                hidden_state.update_n(
                    add=self.summary_length,
                )

            else:
                if hidden_state is not None:
                    # controls the sequence length of the k/v cache
                    hidden_state.update()
                else:
                    raise ValueError("Hidden state must be provided during inference")
            
            
            self.interaction_step += 1       


        return traj_emb, hidden_state

class RmtTransformer(Transformer):
    """Mixin class to turn a AutoModelForCausalLM into an AutoCompressor."""

    def __init__(
        self,
        inp_dim: int,
        max_pos_idx: int,
        d_model: int = 128,
        d_ff: int = 512,
        d_emb_ff: int = None,
        n_heads: int = 4,
        layers: int = 3,
        dropout_emb: float = 0.05,
        dropout_ff: float = 0.05,
        dropout_attn: float = 0.00,
        dropout_qkv: float = 0.00,
        attention: str = "flash",
        activation: str = "leaky_relu",
        norm: str = "layer",
        causal: bool = True,
        pos_emb: str = "learnable",
        accumulate_summary: bool = True,
        segment_gradient_checkpointing: bool = False,
        summary_length: int = 0,
        segment_lengths: int = 0,
        detach_prob: float = 0.0,
    ):

        super().__init__(
            inp_dim=inp_dim,
            max_pos_idx=max_pos_idx,
            d_model=d_model,
            d_ff=d_ff,
            d_emb_ff=d_emb_ff,
            n_heads=n_heads,
            layers=layers,
            dropout_emb=dropout_emb,
            dropout_ff=dropout_ff,
            dropout_attn=dropout_attn,
            dropout_qkv=dropout_qkv,
            attention=attention,
            activation=activation,
            norm=norm,
            causal=causal,
            pos_emb=pos_emb,
        )

        self.accumulate_summary = accumulate_summary
        self.segment_gradient_checkpointing = segment_gradient_checkpointing
        self.summary_length = summary_length
        self.segment_lengths = segment_lengths
        self.detach_prob = detach_prob

        print("SUMMARY LENGTH: ", self.summary_length)
        print("SEGMENT LENGTHS: ", self.segment_lengths)

        assert self.segment_lengths > self.summary_length
        assert self.segment_lengths > 0
        assert self.summary_length >= 0

        self.interaction_step = 0

        # assert self.segment_lngths > self.summary_length
        if summary_length > 0:
            self.embed_summary = nn.Embedding(summary_length, self.d_model)

    """
    this class currently only works for constant length episodes - we need the same 
    """

    def forward(self, seq, pos_idxs, hidden_state: None | TformerHiddenState):
        batch, length, dim = seq.shape

        # if we are in train mode or are getting a batch of sequences of data points rather than a single step
        if self.training or pos_idxs.size(1) > 1:

            min_val = self.summary_length*80

            # we want to sample a segment length between 1-self.segment_lengths, with a bias towards the upper end, which is the segment length
            sampled_segment_length = min_val + (self.segment_lengths + 1 - min_val) * pow(rand(), 0.5)
            sampled_segment_length = int(sampled_segment_length)

            assert hidden_state is None

            h = hidden_state or [[None, None, None] for _ in range(self.n_layers)]

            traj_emb = self.inp(seq)
            traj_embs = []
            # ipdb.set_trace()
            seq_segments = torch.split(traj_emb, sampled_segment_length, dim=1)
            softprompt = None

            for i, segment in enumerate(seq_segments):
                
                # have one variable for whether there is a prev summary and one for whether this segment should be summarized or not
                if segment.size(1) < sampled_segment_length:
                    per_segment_summary_lens = 0
                else:
                    per_segment_summary_lens = self.summary_length

                prev_summ_and_next_seg_len = per_segment_summary_lens + (i>0)*self.summary_length

                # pos idxs for the whole input sequence
                segment_pos_idxs = torch.arange( segment.size(1) + prev_summ_and_next_seg_len, device=pos_idxs.device).unsqueeze(0).expand(segment.size(0), -1) ## chanhge, add one prev summary token set

                # summarisation tokens for the end
                summary_token_ids = torch.arange(per_segment_summary_lens, dtype=torch.long, device=traj_emb.device).unsqueeze(0).expand(traj_emb.size(0), -1)
                summary_token_embeds = self.embed_summary(summary_token_ids).to(traj_emb.dtype)
                
                if softprompt is None:
                    segment_embed = torch.cat([segment, summary_token_embeds], dim=1)
                else:
                    segment_embed = torch.cat([softprompt, segment, summary_token_embeds], dim=1)

                segment_embed = self.dropout(segment_embed + self.position_embedding(segment_pos_idxs))

                # self-attention
                for j, layer in enumerate(self.layers):
                    segment_embed = layer(segment_embed, *h[j])

                traj_emb_i = self.norm(segment_embed)
                # ignore the first summary_length tokens and get the middle segment embeddings, use the last summary_ln tokens as the softprompt

                if per_segment_summary_lens > 0:
                    new_softprompt = traj_emb_i[:, -self.summary_length:]
                    traj_emb_i = traj_emb_i[:, min(i, 1) * self.summary_length:-self.summary_length]
                    softprompt = new_softprompt
                else:
                    traj_emb_i = traj_emb_i[:, min(i, 1) * self.summary_length:]

                traj_embs.append(traj_emb_i)

            traj_emb = torch.cat(traj_embs, dim=1)

        else:
            rollout_idx = self.interaction_step  # pos_idxs[0]     
            h = hidden_state


            index_within_segment = pos_idxs % self.segment_lengths
            softprompt_pos_offset = (pos_idxs > self.segment_lengths) * self.summary_length
            segment_pos_idxs = index_within_segment + softprompt_pos_offset

            traj_emb = self.inp(seq)

            if (rollout_idx + 1) % self.segment_lengths == 0:
                # now we also want to gebnerate a new summary so we will add summary_length tokens to the end of the segment
                summary_token_ids = torch.arange(self.summary_length, dtype=torch.long, device=traj_emb.device).unsqueeze(0).expand(traj_emb.size(0), -1)
                summary_token_embeds = self.embed_summary(summary_token_ids).to(traj_emb.dtype)
                segment_pos_idxs = torch.cat([segment_pos_idxs, torch.arange(self.summary_length, device=pos_idxs.device).unsqueeze(0).expand(traj_emb.size(0), -1)], dim=1)
                traj_emb = torch.cat([traj_emb, summary_token_embeds], dim=1)

            traj_emb = self.dropout(traj_emb + self.position_embedding(segment_pos_idxs))

            # self-attention
            for i, layer in enumerate(self.layers):
                traj_emb = layer(traj_emb, *hidden_state[i])
            traj_emb = self.norm(traj_emb)

            if (rollout_idx + 1) % self.segment_lengths == 0:
                # ignore the last summary_length tokens and get the middle segment embeddings, use the last summary_ln tokens as the softprompt
                new_softprompt_inp_embeds = traj_emb[:, -self.summary_length:]
                traj_emb = traj_emb[:, :-self.summary_length]

                # now do a forward pass on just the embedding outputs from the summary tokens to get the new softprompt, use pos_idxs correspondng to how many softprompts vecs were collected previously == 0 in this case
                new_softprompt_pos_idxs = torch.arange(self.summary_length, device=pos_idxs.device).unsqueeze(0).expand(traj_emb.size(0), -1)

                # new_softprompt = self.inp(new_softprompt_inp_embeds)                                                      
                new_softprompt = self.dropout(new_softprompt_inp_embeds + self.position_embedding(new_softprompt_pos_idxs))

                # clear the cache
                hidden_state.reset_and_clear_all()

                for i, layer in enumerate(self.layers):
                    new_softprompt = layer(new_softprompt, *hidden_state[i])                                         # TODO assert that hidden state only has num_softprompts * summary_len entries    
                new_softprompt = self.norm(new_softprompt)

                hidden_state.update_n(
                    add=self.summary_length,
                )

            else:
                if hidden_state is not None:
                    # controls the sequence length of the k/v cache
                    hidden_state.update()
                else:
                    raise ValueError("Hidden state must be provided during inference")
            
            
            self.interaction_step += 1          


        return traj_emb, hidden_state