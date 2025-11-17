import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


class SingleViewSequenceDataset(Dataset):
    """
    Fast dataset for a single HDF5 sequence.
    Each HDF5 file = one trajectory (len_traj, C, H, W).
    Later you can wrap many of these in ConcatDataset.
    """

    def __init__(
        self,
        hd5_file_path: str,
        traj_len: int,
        non_overlapping: bool = False,
        load_to_ram: bool = True,
        nu: int = 2,
        rate: int = 1,
    ):
        """
        Args:
            hd5_file_path: path to HDF5 file for a single sequence.
            traj_len: number of time steps in a chunk.
            non_overlapping: if True, segments do not overlap.
            load_to_ram: if True, load entire sequence into RAM as torch tensors.
            nu: number of action/state dims to keep.
            rate: temporal downsampling factor.
        """
        super().__init__()

        self.hd5_file_path = hd5_file_path
        self.traj_len = traj_len
        self.non_overlapping = non_overlapping
        self.load_to_ram = load_to_ram
        self.nu = nu
        self.rate = rate

        # We only open once here to inspect shape & optionally load-to-RAM.
        with h5py.File(self.hd5_file_path, "r") as f:
            cam1 = f["cam1"]
            self.len_traj, self.C, self.H, self.W = cam1.shape

            if self.traj_len * self.rate > self.len_traj:
                raise ValueError(
                    f"traj_len * rate = {self.traj_len} * {self.rate} "
                    f"> len_traj = {self.len_traj}"
                )

            self.has_states = "states" in f.keys()

            if self.load_to_ram:
                # Load everything once and convert to tensors
                self.cam1 = torch.from_numpy(cam1[:]).float()          # (T, C, H, W)
                self.actions = torch.from_numpy(f["actions"][:]).float()  # (T, A)
                if self.has_states:
                    self.states = torch.from_numpy(f["states"][:]).float()  # (T, S)
                else:
                    self.states = None

        # For on-disk mode, we'll open lazily per worker in __getitem__.
        if not self.load_to_ram:
            self.cam1 = None
            self.actions = None
            self.states = None
            self._h5 = None  # will be opened lazily

        # Compute how many segments and their start indices
        self._compute_starts()

    def _compute_starts(self):
        # Max possible number of overlapping segments
        max_overlap_len = self.len_traj - self.traj_len * self.rate + 1

        if self.non_overlapping:
            # Each segment spans traj_len * rate time steps
            n_segments = self.len_traj // (self.traj_len * self.rate)
            # Starts: 0, traj_len*rate, 2*traj_len*rate, ...
            starts = torch.arange(
                0,
                n_segments * self.traj_len * self.rate,
                self.traj_len * self.rate,
                dtype=torch.long,
            )
        else:
            # Fully overlapping sliding window
            n_segments = max_overlap_len
            starts = torch.arange(0, n_segments, dtype=torch.long)

        self.num_segments = int(n_segments)
        self.starts = starts  # (num_segments,)

    def __len__(self):
        return self.num_segments

    def _lazy_open_h5(self):
        """
        Open HDF5 file lazily per worker process.
        """
        if self._h5 is None:
            self._h5 = h5py.File(self.hd5_file_path, "r")
        return self._h5

    def _get_from_disk(self, sl):
        """
        Fetch a slice from disk and convert to tensors.
        """
        f = self._lazy_open_h5()
        cam1 = f["cam1"][sl]        # (traj_len, C, H, W)
        actions = f["actions"][sl]  # (traj_len, A)

        imgs = torch.from_numpy(cam1).float()
        actions = torch.from_numpy(actions).float()

        if "states" in f.keys():
            states_np = f["states"][sl]
            states = torch.from_numpy(states_np).float()
        else:
            states = torch.zeros_like(actions)

        return imgs, actions, states

    def _get_from_ram(self, sl):
        """
        Fetch a slice from preloaded tensors in RAM.
        """
        imgs = self.cam1[sl]        # (traj_len, C, H, W)
        actions = self.actions[sl]  # (traj_len, A)
        if self.states is not None:
            states = self.states[sl]
        else:
            states = torch.zeros_like(actions)
        return imgs, actions, states

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.num_segments:
            raise IndexError(f"Index {idx} out of range 0..{self.num_segments - 1}")

        start = int(self.starts[idx].item())
        end = start + self.traj_len * self.rate
        sl = slice(start, end, self.rate)  # stride by rate

        # Read data either from RAM or from disk
        if self.load_to_ram:
            imgs, actions, states = self._get_from_ram(sl)
        else:
            imgs, actions, states = self._get_from_disk(sl)

        # Crop state/action dims to nu
        Dout = {
            "observation.image": imgs,                                   # (T, C, H, W)
            "observation.state": states[:, : self.nu],                   # (T, nu)
            "action": actions[:, : self.nu],                             # (T, nu)
        }
        return Dout