import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
from torch.utils.data import Dataset, DistributedSampler
import h5py
import numpy as np
from pathlib import Path
import json

class SingleViewSequenceDataset(Dataset):
    """
    Fast dataset for a single HDF5 sequence.
    Each HDF5 file = one trajectory (len_traj, C, H, W).
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
            if 'dreamer-tokens' in f:
                self.tokens = f['dreamer-tokens']
                T1, self.C, self.H, self.W = cam1.shape
                T2 = self.tokens.shape[0]
                self.len_traj = min(T1, T2)

            else:
                self.tokens = None
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
                if self.tokens is not None:
                    self.tokens = torch.from_numpy(self.tokens[:]).float()          # (T, C, H, W)
                self.actions = torch.from_numpy(f["actions"][:]).float()  # (T, A)
                if self.has_states:
                    self.states = torch.from_numpy(f["states"][:]).float()  # (T, S)
                else:
                    self.states = None

        # For on-disk mode, we'll open lazily per worker in __getitem__.
        if not self.load_to_ram:
            self.cam1 = None
            self.tokens = None
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
        tokens = f["dreamer-tokens"][sl] if "dreamer-tokens" in f else None

        imgs = torch.from_numpy(cam1).float()
        actions = torch.from_numpy(actions).float()

        if "states" in f.keys():
            states_np = f["states"][sl]
            states = torch.from_numpy(states_np).float()
        else:
            states = torch.zeros_like(actions)

        return imgs, actions, states, tokens

    def _get_from_ram(self, sl):
        """
        Fetch a slice from preloaded tensors in RAM.
        """
        imgs = self.cam1[sl]        # (traj_len, C, H, W)
        actions = self.actions[sl]  # (traj_len, A)
        if self.tokens is not None:
            tokens = self.tokens[sl]
        else:
            tokens = None
        if self.states is not None:
            states = self.states[sl]
        else:
            states = torch.zeros_like(actions)
        return imgs, actions, states, tokens

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.num_segments:
            raise IndexError(f"Index {idx} out of range 0..{self.num_segments - 1}")

        start = int(self.starts[idx].item())
        end = start + self.traj_len * self.rate
        sl = slice(start, end, self.rate)  # stride by rate

        # Read data either from RAM or from disk
        if self.load_to_ram:
            imgs, actions, states, tokens = self._get_from_ram(sl)
        else:
            imgs, actions, states, tokens = self._get_from_disk(sl)

        # Crop state/action dims to nu
        Dout = {
            "observation.image": imgs,                                   # (T, C, H, W)
            "observation.state": states[:, : self.nu],                   # (T, nu)
            "action": actions[:, : self.nu],                             # (T, nu)
        }
        if tokens is not None:
            Dout["observation.tokens"] = tokens                         # (T, C, H, W)

        return Dout
    
    def close(self):
        if self.load_to_ram:
            return
        if self._h5 is not None:
            try:
                self._h5.close()
            except:
                pass
            self._h5 = None
    
class ShardedHDF5Dataset(Dataset):
    """
    Dataset for sharded HDF5 files optimized for multi-node training.
    Each worker preferentially reads from local shards when available.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
        split: str = "train",          # "train" or "test"
        train_fraction: float = 0.9,   # fraction of episodes in train
        split_seed: int = 42,          # seed for reproducible split
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.split = split
        self.train_fraction = train_fraction
        self.split_seed = split_seed

        # Load metadata
        with open(self.data_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self.num_shards = self.metadata['num_shards']
        self.shard_files = [
            self.data_dir / f"shard_{i:04d}.h5"
            for i in range(self.num_shards)
        ]

        # Build window index across all shards
        self.windows = []
        self.episode_lengths = []  # Store all episode lengths for analysis

        for shard_idx, shard_file in enumerate(self.shard_files):
            with h5py.File(shard_file, 'r') as f:
                num_episodes = f.attrs['num_episodes']
                lengths = f['episode_lengths'][:]

                # Store episode lengths for statistics
                self.episode_lengths.extend(lengths.tolist())

                for ep_idx, ep_length in enumerate(lengths):
                    for start in range(0, ep_length - window_size + 1, stride):
                        self.windows.append((shard_idx, ep_idx, start))

        # Collect all (shard_idx, ep_idx) pairs
        all_episodes = sorted({(shard_idx, ep_idx) for shard_idx, ep_idx, _ in self.windows})

        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(len(all_episodes))

        num_train_eps = int(self.train_fraction * len(all_episodes))
        train_eps = {all_episodes[i] for i in perm[:num_train_eps]}
        test_eps  = {all_episodes[i] for i in perm[num_train_eps:]}

        self.split_info = {
            "train_episodes": sorted(list(train_eps)),
            "test_episodes": sorted(list(test_eps)),
        }

        if self.split == "train":
            keep = train_eps
        elif self.split == "test":
            keep = test_eps
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Filter windows based on chosen split
        self.windows = [w for w in self.windows if (w[0], w[1]) in keep]
        print(f"{self.split.capitalize()} split: {len(self.windows)} windows "
              f"from {len(keep)} episodes")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        shard_idx, ep_idx, start = self.windows[idx]
        end = start + self.window_size

        shard_file = self.shard_files[shard_idx]

        # Open HDF5 file (each worker maintains its own handle)
        with h5py.File(shard_file, 'r') as f:
            images = f['images'][ep_idx, start:end]
            actions = f['actions'][ep_idx, start:end]

        # Convert to PyTorch
        images = torch.from_numpy(images).float() / 255.0
        images = images.permute(0, 3, 1, 2)
        actions = torch.from_numpy(actions)
        
        return {'image': images, 'action': actions}


    def get_episode_length_statistics(self):
        """
        Calculate comprehensive statistics about episode lengths.
        
        Returns:
            dict with statistics about episode lengths
        """
        lengths = np.array(self.episode_lengths)
        
        stats = {
            'total_episodes': len(lengths),
            'total_timesteps': int(np.sum(lengths)),
            'min_length': int(np.min(lengths)),
            'max_length': int(np.max(lengths)),
            'mean_length': float(np.mean(lengths)),
            'median_length': float(np.median(lengths)),
            'std_length': float(np.std(lengths)),
            'percentile_25': float(np.percentile(lengths, 25)),
            'percentile_75': float(np.percentile(lengths, 75)),
            'percentile_90': float(np.percentile(lengths, 90)),
            'percentile_95': float(np.percentile(lengths, 95)),
            'percentile_99': float(np.percentile(lengths, 99)),
        }
        
        return stats
