from abc import ABC, abstractmethod
from typing import Optional
import math

import torch
from torch import nn
import gin

from torch.nn import functional as F
from amago.nets.goal_embedders import FFGoalEmb, TokenGoalEmb
from amago.nets.utils import InputNorm, add_activation_log, symlog
from amago.nets import ff, cnn

import ipdb

# PositionalEmbedding2D(height, width, core_out_size, self.cfg.model.device, relative=cfg.env.relative_position)
class PositionalEmbedding2D(nn.Module):
    def __init__(self, height, width, dim, relative=False):
        super(PositionalEmbedding2D, self).__init__()
        self.relative = relative
        multiplier = 3 if relative else 1
        self.height = height * multiplier
        self.width = width * multiplier
        self.dim = dim


        if dim % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(dim))

        self.pe = torch.zeros(dim, self.height, self.width)

        # Each dimension use half of dim
        dim = int(dim / 2)
        div_term = torch.exp(torch.arange(0., dim, 2) * (-(math.log(10000.0)) / dim))
        pos_w = torch.arange(0., self.width).unsqueeze(1)
        pos_h = torch.arange(0., self.height).unsqueeze(1)
        self.pe[0:dim:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        self.pe[1:dim:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        self.pe[dim::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        self.pe[dim + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        self.pe = self.pe.to('cuda')
        # print(self.pe)

    def forward(self, x, position_info):
        # print(position_info)
        # sometimes position_info might be 3-dimensional, so we need to reshape it to 2-dimensional flattening the first two dims and later reshape the indexed pe back to 3-dimensional to match the input x
        if position_info.dim() == 3:
            position_info = position_info.view(-1, position_info.size(2))
        if self.relative:
            position_info[:, 0] += self.width / 2
            position_info[:, 1] += self.height / 2
        x_pos = torch.floor(position_info[:, 0]).long()
        y_pos = torch.floor(position_info[:, 1]).long()
        # print(x_pos.max(), y_pos.max(), x_pos.min(), y_pos.min())
        # print('.....')
        # print(x_pos.shape, y_pos.shape)
        pe = self.pe[:, x_pos, y_pos].transpose(0, 1)
        # print(pe.shape)
        # print(x.shape)
        pe = pe.view(x.size(0), x.size(1), -1) if x.dim() == 3 else pe
        x = x + pe
        return x


@gin.configurable
class TstepEncoder(nn.Module, ABC):
    def __init__(self, obs_space, goal_space, rl2_space, goal_emb_Cls=TokenGoalEmb):
        super().__init__()
        self.obs_space = obs_space
        self.goal_space = goal_space
        self.rl2_space = rl2_space
        goal_length, goal_dim = goal_space.shape
        self.goal_emb = goal_emb_Cls(goal_length=goal_length, goal_dim=goal_dim)
        self.goal_emb_dim = self.goal_emb.goal_emb_dim

    def forward(self, obs, goals, rl2s, log_dict: Optional[dict] = None):
        goal_rep = self.goal_emb(goals)
        out = self.inner_forward(obs, goal_rep, rl2s, log_dict=log_dict)
        return out

    @abstractmethod
    def inner_forward(self, obs, goal_rep, rl2s, log_dict: Optional[dict] = None):
        pass

    @property
    @abstractmethod
    def emb_dim(self):
        pass


@gin.configurable
class FFTstepEncoder(TstepEncoder):
    def __init__(
        self,
        obs_space,
        goal_space,
        rl2_space,
        n_layers: int = 2,
        d_hidden: int = 512,
        d_output: int = 256,
        norm: str = "layer",
        activation: str = "leaky_relu",
        hide_rl2s: bool = False,
        normalize_inputs: bool = True,
    ):
        super().__init__(
            obs_space=obs_space, goal_space=goal_space, rl2_space=rl2_space
        )
        flat_obs_shape = math.prod(self.obs_space["observation"].shape)
        in_dim = flat_obs_shape + self.goal_emb_dim + self.rl2_space.shape[-1]
        self.in_norm = InputNorm(
            flat_obs_shape + self.rl2_space.shape[-1], skip=not normalize_inputs
        )
        self.base = ff.MLP(
            d_inp=in_dim,
            d_hidden=d_hidden,
            n_layers=n_layers,
            d_output=d_output,
            activation=activation,
        )
        self.out_norm = ff.Normalization(norm, d_output)
        self._emb_dim = d_output
        self.hide_rl2s = hide_rl2s

    def inner_forward(self, obs, goal_rep, rl2s, log_dict: Optional[dict] = None):
        # multi-modal envs that do not use the default `observation` key need their own custom encoders.
        obs = obs["observation"]
        B, L, *_ = obs.shape
        if self.hide_rl2s:
            rl2s = rl2s * 0
        flat_obs_rl2 = torch.cat((obs.view(B, L, -1).float(), rl2s), dim=-1)
        if self.training:
            self.in_norm.update_stats(flat_obs_rl2)
        flat_obs_rl2 = self.in_norm(flat_obs_rl2)
        obs_rl2_goals = torch.cat((flat_obs_rl2, goal_rep), dim=-1)
        prenorm = self.base(obs_rl2_goals)
        out = self.out_norm(prenorm)
        return out

    @property
    def emb_dim(self):
        return self._emb_dim


@gin.configurable
class CNNTstepEncoder(TstepEncoder):
    def __init__(
        self,
        obs_space,
        goal_space,
        rl2_space,
        cnn_Cls=cnn.NatureishCNN,
        channels_first: bool = False,
        img_features: int = 384,
        rl2_features: int = 12,
        d_output: int = 384,
        out_norm: str = "layer",
        activation: str = "leaky_relu",
        skip_rl2_norm: bool = False,
        hide_rl2s: bool = False,
        drqv2_aug: bool = True,
    ):
        super().__init__(
            obs_space=obs_space, goal_space=goal_space, rl2_space=rl2_space
        )
        self.data_aug = (
            cnn.DrQv2Aug(4, channels_first=channels_first) if drqv2_aug else lambda x: x
        )
        obs_shape = self.obs_space["observation"].shape
        self.cnn = cnn_Cls(
            img_shape=obs_shape,
            channels_first=channels_first,
            activation=activation,
        )
        img_feature_dim = self.cnn(
            torch.zeros((1, 1) + obs_shape, dtype=torch.uint8)
        ).shape[-1]
        self.img_features = nn.Linear(img_feature_dim, img_features)

        self.rl2_norm = InputNorm(self.rl2_space.shape[-1], skip=skip_rl2_norm)
        self.rl2_features = nn.Linear(rl2_space.shape[-1], rl2_features)

        mlp_in = img_features + self.goal_emb_dim + rl2_features
        self.merge = nn.Linear(mlp_in, d_output)
        self.out_norm = ff.Normalization(out_norm, d_output)
        self.hide_rl2s = hide_rl2s
        self._emb_dim = d_output

    def inner_forward(self, obs, goal_rep, rl2s, log_dict: Optional[dict] = None):
        # multi-modal envs that do not use the default `observation` key need their own custom encoders.
        img = obs["observation"].float()
        B, L, *_ = img.shape
        if self.training:
            og_split = max(min(math.ceil(B * 0.25), B - 1), 0)
            aug = self.data_aug(img[og_split:, ...])
            img = torch.cat((img[:og_split, ...], aug), dim=0)
        img = (img / 128.0) - 1.0
        img_rep = self.cnn(img, flatten=True, from_float=True)
        add_activation_log("cnn_out", img_rep, log_dict)
        img_rep = self.img_features(img_rep)
        add_activation_log("img_features", img_rep, log_dict)

        rl2s = symlog(rl2s)
        rl2s_norm = self.rl2_norm(rl2s)
        if self.training:
            self.rl2_norm.update_stats(rl2s)
        if self.hide_rl2s:
            rl2s_norm = rl2s_norm * 0
        rl2s_rep = self.rl2_features(rl2s_norm)

        inp = torch.cat((img_rep, goal_rep, rl2s_rep), dim=-1)
        merge = self.merge(inp)
        add_activation_log("tstep_encoder_prenorm", merge, log_dict)
        out = self.out_norm(merge)
        return out

    @property
    def emb_dim(self):
        return self._emb_dim

@gin.configurable
class MultimodalCNNTstepEncoder(TstepEncoder):
    def __init__(
        self,
        obs_space,
        goal_space,
        rl2_space,
        cnn_Cls=cnn.NatureishCNN,
        channels_first: bool = False,
        img_features: int = 512,
        d_hidden: int = 512,
        n_layers: int = 2,
        d_output: int = 256,
        norm: str = "layer",
        activation: str = "leaky_relu",
        skip_rl2_norm: bool = False,
        hide_rl2s: bool = False,
    ):
        super().__init__(
            obs_space=obs_space, goal_space=goal_space, rl2_space=rl2_space
        )
        obs_shape = self.obs_space["image"].shape
        print(cnn_Cls)
        self.cnn = cnn_Cls(
            img_shape=obs_shape,
            channels_first=channels_first,
            activation=activation,
        )
        img_feature_dim = self.cnn(
            torch.zeros((1, 1) + obs_shape, dtype=torch.uint8)
        ).shape[-1]
        self.img_features = nn.Linear(img_feature_dim, img_features)
        self.img_norm = ff.Normalization(norm, img_features)
        self.rl2_norm = InputNorm(self.rl2_space.shape[-1], skip=skip_rl2_norm)
        mlp_in = img_features + self.goal_emb_dim + self.rl2_space.shape[-1]
        self.merge = ff.MLP(
            d_inp=mlp_in, d_hidden=d_hidden, n_layers=n_layers, d_output=d_output
        )
        self.out_norm = ff.Normalization(norm, d_output)
        self.hide_rl2s = hide_rl2s
        self._emb_dim = d_output

        # now we fuse the agent position and dir with the image features
        self.fuse = nn.Linear(img_features + 16, img_features)
        self.fuse_norm = ff.Normalization(norm, img_features)

        self.action_linear = nn.Linear(6, 16)
        self.action_norm = ff.Normalization(norm, 16)

    def forward(self, obs, goals, rl2s, log_dict: Optional[dict] = None):
        goal_rep = self.goal_emb(goals)
        out = self.inner_forward(obs, goal_rep, rl2s, log_dict=log_dict)
        return out

    def inner_forward(self, obs, goal_rep, rl2s, log_dict: Optional[dict] = None):
        bsz, seq_len = obs["image"].shape[:2]
        actions = obs["prev_action"]
        # convert this to one hot
        actions = F.one_hot(actions, num_classes=6).squeeze(2)
        acts = self.action_linear(actions.float().view(bsz*seq_len, 6)).view(bsz, seq_len, 16)
        acts = self.action_norm(acts)
                                                                        ## TODO: normalise and set from_float=true, why is it uint8 now

        obs_img = obs["image"]
        img_rep = self.img_features(self.cnn(obs_img))                      ## TODO: do the log dict thing here also
        img_rep = self.img_norm(img_rep)

        img_rep = self.fuse(torch.cat((img_rep, acts), dim=-1))
        img_rep = self.fuse_norm(img_rep)
        # except:
        #     ipdb.set_trace()

        rl2s_norm = self.rl2_norm(rl2s)
        if self.training:
            self.rl2_norm.update_stats(rl2s)
        if self.hide_rl2s:
            rl2s = rl2s * 0
        inp = torch.cat((img_rep, goal_rep, rl2s_norm), dim=-1)

        merge = self.merge(inp)
        add_activation_log("tstep_encoder_prenorm", merge, log_dict)
        out = self.out_norm(merge)
        return out

    @property
    def emb_dim(self):
        return self._emb_dim

@gin.configurable
class MultimodalPoseCNNTstepEncoder(TstepEncoder):
    def __init__(
        self,
        obs_space,
        goal_space,
        rl2_space,
        cnn_Cls=cnn.NatureishCNN,
        channels_first: bool = False,
        img_features: int = 512,
        d_hidden: int = 512,
        n_layers: int = 2,
        d_output: int = 256,
        norm: str = "layer",
        activation: str = "leaky_relu",
        skip_rl2_norm: bool = False,
        hide_rl2s: bool = False,
    ):
        super().__init__(
            obs_space=obs_space, goal_space=goal_space, rl2_space=rl2_space
        )
        obs_shape = self.obs_space["image"].shape
        self.cnn = cnn_Cls(
            img_shape=obs_shape,
            channels_first=channels_first,
            activation=activation,
        )
        print(cnn_Cls)
        img_feature_dim = self.cnn(
            torch.zeros((1, 1) + obs_shape, dtype=torch.uint8)
        ).shape[-1]
        self.img_features = nn.Linear(img_feature_dim, img_features)
        self.img_norm = ff.Normalization(norm, img_features)
        self.rl2_norm = InputNorm(self.rl2_space.shape[-1], skip=skip_rl2_norm)
        mlp_in = img_features + self.goal_emb_dim + self.rl2_space.shape[-1]
        self.merge = ff.MLP(
            d_inp=mlp_in, d_hidden=d_hidden, n_layers=n_layers, d_output=d_output
        )
        self.out_norm = ff.Normalization(norm, d_output)
        self.hide_rl2s = hide_rl2s
        self._emb_dim = d_output

        add_dims = 0
        print('obs space: /////', self.obs_space)
        self.agent_pos_encoding = PositionalEmbedding2D(15, 15, img_features)

        # now we fuse the agent position and dir with the image features
        self.fuse = nn.Linear(img_features + add_dims + 16, img_features)
        self.fuse_norm = ff.Normalization(norm, img_features)

        self.action_linear = nn.Linear(6, 16)
        self.action_norm = ff.Normalization(norm, 16)

    def forward(self, obs, goals, rl2s, log_dict: Optional[dict] = None):
        goal_rep = self.goal_emb(goals)
        out = self.inner_forward(obs, goal_rep, rl2s, log_dict=log_dict)
        return out

    @torch.compile
    def inner_forward(self, obs, goal_rep, rl2s, log_dict: Optional[dict] = None):
        bsz, seq_len = obs["image"].shape[:2]


        actions = obs["prev_action"]
        # print('actions: ', actions)
        # convert this to one hot
        actions = F.one_hot(actions, num_classes=6).squeeze(2)
        acts = self.action_linear(actions.float().view(bsz*seq_len, 6)).view(bsz, seq_len, 16)
        acts = self.action_norm(acts)
                                                                        ## TODO: normalise and set from_float=true, why is it uint8 now

        obs_img = obs["image"]
        img_rep = self.img_features(self.cnn(obs_img))                      ## TODO: do the log dict thing here also
        img_rep = self.img_norm(img_rep)

        img_rep = self.agent_pos_encoding(img_rep, obs['agent_pos'])

        img_rep = self.fuse(torch.cat((img_rep, acts), dim=-1))
        img_rep = self.fuse_norm(img_rep)


        rl2s_norm = self.rl2_norm(rl2s)
        if self.training:
            self.rl2_norm.update_stats(rl2s)
        if self.hide_rl2s:
            rl2s = rl2s * 0
        inp = torch.cat((img_rep, goal_rep, rl2s_norm), dim=-1)

        merge = self.merge(inp)
        add_activation_log("tstep_encoder_prenorm", merge, log_dict)
        out = self.out_norm(merge)
        return out

    @property
    def emb_dim(self):
        return self._emb_dim