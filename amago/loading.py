import os
import random
import shutil
import pickle
from dataclasses import dataclass
from operator import itemgetter
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tensordict import MemoryMappedTensor, TensorDict
import numpy as np
import ipdb
from .hindsight import Trajectory, Relabeler, FrozenTraj


def load_traj_from_disk(path: str) -> Trajectory | FrozenTraj:
    _, ext = os.path.splitext(path)
    if ext == ".traj":
        with open(path, "rb") as f:
            disk = pickle.load(f)
        traj = Trajectory(max_goals=disk.max_goals, timesteps=disk.timesteps)
        return traj
    elif ext == ".npz":
        disk = FrozenTraj.from_dict(np.load(path))
        return disk
    else:
        raise ValueError(
            f"Unrecognized trajectory file extension `{ext}` for path `{path}`."
        )


class TrajDset(Dataset):
    """
    Load trajectory files from disk in parallel with pytorch Dataset/DataLoader
    pipeline.
    """

    def __init__(
        self,
        relabeler: Relabeler,
        dset_root: str = None,
        dset_name: str = None,
        dset_split: str = "train",
        items_per_epoch: int = None,
        max_seq_len: int = None,
        warmstart: bool = False,
    ):
        assert dset_split in ["train", "val", "test"]
        assert dset_root is not None and os.path.exists(dset_root)
        self.max_seq_len = max_seq_len
        self.dset_split = dset_split
        self.dset_path = (
            os.path.join(dset_root, dset_name, dset_split) if dset_name else None
        )
        self.length = items_per_epoch if dset_name else None
        self.filenames = []
        self.external_filenames = []
        self.warmstart = warmstart
        if warmstart:
            self.external_file_path =  ""
            self.external_filenames = os.listdir(self.external_file_path)
        else:
            self.external_file_path = None
        self.refresh_files()
        if len(self.filenames) > 0 and warmstart and len(self.external_filenames) > 0:
            # raise ValueError("Cannot have both warmstart and already existing files - this looks like a run being resumed.")
            self.warmstart = False
            self.external_filenames = []
            self.external_file_path = None

        print('filenames: ', self.filenames)
        if len(self.external_filenames) > 0:
            print('externals: ', self.external_filenames)
        self.relabeler = relabeler

    def __len__(self):
        # this length is used by DataLoaders to end an epoch
        if self.length is None:
            return self.count_trajectories()
        else:
            return self.length

    @property
    def disk_usage(self):
        bytes = sum(
            os.path.getsize(os.path.join(self.dset_path, f)) for f in self.filenames
        )
        return bytes * 1e-9

    def clear(self):
        # remove files on disk
        if os.path.exists(self.dset_path):
            shutil.rmtree(self.dset_path)
            os.makedirs(self.dset_path)

    def refresh_files(self):
        # find the new .traj files from the previous rollout
        if self.dset_path is not None and os.path.exists(self.dset_path):
            self.filenames = os.listdir(self.dset_path)
            self.file_len = len(self.filenames)


    def count_trajectories(self) -> int:
        # get the real dataset size
        return len(self.filenames)

    def filter(self, delete_pct: float):
        """
        Imitates fixed-size replay buffers by clearing .traj files on disk.
        """
        assert delete_pct <= 1.0 and delete_pct >= 0.0
        print(delete_pct)

        if self.warmstart and len(self.external_filenames) > 0:
            # then we first want toremove filenames from the external buffer
            items_to_delete = round((len(self.external_filenames) + len(self.filenames)) * delete_pct)
            # we only want to remove the path from the list - not delete the actual trajectory file because these are shared across multiple datasets
            # we also don't care about sampling randomly, just make sure we don't get an error if we try to delete more than we have
            to_delete = self.external_filenames[:min(items_to_delete, len(self.external_filenames))]
            self.external_filenames = self.external_filenames[len(to_delete):]
        else:
            traj_infos = []
            for traj_filename in self.filenames:
                env_name, rand_id, unix_time = os.path.splitext(traj_filename)[0].split("_")
                time, _ = unix_time.split(".")
                traj_infos.append(
                    {
                        "env": env_name,
                        "rand": rand_id,
                        "time": int(time),
                        "filename": traj_filename,
                    }
                )
            traj_infos = sorted(traj_infos, key=lambda d: d["time"])
            num_to_remove = round(len(traj_infos) * delete_pct)
            to_delete = list(map(itemgetter("filename"), traj_infos[:num_to_remove]))
            for file_to_delete in to_delete:
                os.remove(os.path.join(self.dset_path, file_to_delete))

    def __getitem__(self, i):
        # if warmstarting, we want to sample from the external buffer as well in proportion to its size
        if self.warmstart and len(self.external_filenames) > 0:
            if random.random() < len(self.filenames) / (len(self.filenames) + len(self.external_filenames)):
                filename = random.choice(self.filenames)
                # print('filename: ', filename)
                traj = load_traj_from_disk(os.path.join(self.dset_path, filename))
            else:
                try:
                    filename = random.choice(self.external_filenames)
                    # print('external filename: ', filename)
                    traj = load_traj_from_disk(os.path.join(self.external_file_path, filename))
                except:
                    print(len(self.external_filenames))
                    print(self.external_file_path)
                    print(filename)
                    raise ValueError("Could not load from external buffer.")
        else:
            use_i = i%self.file_len         
            filename = self.filenames[use_i]
            # print('filename: ', filename)
            traj = load_traj_from_disk(os.path.join(self.dset_path, filename))

        if isinstance(traj, Trajectory):
            traj = self.relabeler(traj)
        data = RLData(traj)
        if self.max_seq_len is not None:
            data = data.random_slice(length=self.max_seq_len)
        return data


class RLData:
    def __init__(self, traj: Trajectory | FrozenTraj):
        if isinstance(traj, Trajectory):
            traj = traj.freeze()
        assert isinstance(traj, FrozenTraj)
        # self.obs = {k: torch.from_numpy(v) for k, v in traj.obs.items()}
        self.obs = TensorDict(
            {k: torch.from_numpy(v) for k, v in traj.obs.items()},
            batch_size=traj.time_idxs.shape,
        )
        self.goals = torch.from_numpy(traj.goals).float()
        self.rl2s = torch.from_numpy(traj.rl2s).float()
        self.time_idxs = torch.from_numpy(traj.time_idxs).long()
        self.rews = torch.from_numpy(traj.rews).float()
        self.dones = torch.from_numpy(traj.dones).bool()
        self.actions = torch.from_numpy(traj.actions).float()

    def __len__(self):
        return len(self.actions)

    def random_slice(self, length: int):
        i = random.randrange(0, max(len(self) - length + 1, 1))
        # the causal RL loss requires these off-by-one lengths
        self.obs = self.obs[i : i + length + 1].contiguous() #{k: v[i : i + length + 1] for k, v in self.obs.items()}
        # self.obs = self.obs.apply(lambda x: x.contiguous())
        self.goals = self.goals[i : i + length + 1].contiguous()
        self.rl2s = self.rl2s[i : i + length + 1].contiguous()
        self.time_idxs = self.time_idxs[i : i + length + 1].contiguous()
        self.dones = self.dones[i : i + length].contiguous()
        self.rews = self.rews[i : i + length].contiguous()
        self.actions = self.actions[i : i + length].contiguous()
        return self


MAGIC_PAD_VAL = 0
pad = partial(pad_sequence, batch_first=True, padding_value=MAGIC_PAD_VAL)


@dataclass
class Batch:
    """
    Keeps data organized during training step
    """

    obs: dict[torch.Tensor]
    goals: torch.Tensor
    rl2s: torch.Tensor
    rews: torch.Tensor
    dones: torch.Tensor
    actions: torch.Tensor
    time_idxs: torch.Tensor

    def to(self, device):
        self.obs = self.obs.to(device, non_blocking_pin=True, num_threads=8)
        # self.obs = {k: v.to(device, non_blocking=True) for k, v in self.obs.items()}
        self.goals = self.goals.to(device, non_blocking=True)
        self.rl2s = self.rl2s.to(device, non_blocking=True)
        self.rews = self.rews.to(device, non_blocking=True)
        self.dones = self.dones.to(device, non_blocking=True)
        self.actions = self.actions.to(device, non_blocking=True)
        self.time_idxs = self.time_idxs.to(device, non_blocking=True)
        return self

def RLData_pad_collate(samples: list[RLData]) -> Batch:
    assert samples[0].obs.keys() == samples[-1].obs.keys()

    time_idxs = pad([s.time_idxs for s in samples])
    # obs = pad([s.obs for s in samples])
    obs = TensorDict(
        {k: pad([s.obs[k] for s in samples]).contiguous() for k in samples[0].obs.keys()},
        batch_size=time_idxs.shape,
    )
    goals = pad([s.goals for s in samples]).contiguous()
    rl2s = pad([s.rl2s for s in samples]).contiguous()
    rews = pad([s.rews for s in samples]).contiguous()
    dones = pad([s.dones for s in samples]).contiguous()
    actions = pad([s.actions for s in samples]).contiguous()
    return Batch(
        obs=obs,
        goals=goals,
        rl2s=rl2s,
        rews=rews,
        dones=dones,
        actions=actions,
        time_idxs=time_idxs,
    )
