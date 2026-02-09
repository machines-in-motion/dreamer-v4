import h5py
import torch
import random
import json
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class PushTDataset(Dataset):
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

# To be depricated
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
        shuffle_windows: bool = True,    
    ):
        self.shuffle_windows = shuffle_windows
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
        if self.shuffle_windows:
            random.shuffle(self.windows)

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

class HDF5SequenceDataset(Dataset):
    """
    Dataset that creates sliding windows from a directory of independent .h5 files.
    It automatically parses 'dones' to ensure windows do not cross episode boundaries.
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        stride: int = 1,
    ):
        """
        Args:
            data_dir: Directory containing .h5 files.
            window_size: Sequence length (batch time dimension).
            stride: Step size between windows.
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        
        # Find all H5 files
        self.h5_files = sorted([
            f for f in self.data_dir.glob("*.h5") 
            if "shard" not in f.name # Exclude shard files if mixed
        ])
        
        if not self.h5_files:
            raise ValueError(f"No .h5 files found in {data_dir}")

        # Indexing structure: list of (file_index, start_frame_index)
        self.windows = []
        
        print(f"Scanning {len(self.h5_files)} files for valid episodes...")
        
        total_frames = 0
        total_episodes = 0
        
        for file_idx, file_path in enumerate(self.h5_files):
            try:
                with h5py.File(file_path, 'r') as f:
                    if 'dones' not in f or 'images' not in f:
                        print(f"Skipping {file_path.name}: missing datasets")
                        continue
                        
                    n_frames = len(f['images'])
                    dones = f['dones'][:]
                    
                    # Identify Episode Boundaries
                    # An episode ends where done=1.
                    # We need start and end indices for every continuous segment.
                    
                    # Indices where done == 1
                    done_indices = np.where(dones > 0.5)[0]
                    
                    # Episode Starts: [0] + [idx+1 for idx in done_indices if idx+1 < n_frames]
                    # Episode Ends:   [idx+1 for idx in done_indices] + [n_frames] (if last frame isn't done)
                    
                    # Simpler logic: Iterate through done indices to carve out chunks
                    start_idx = 0
                    
                    # Add a synthetic done at the very end to close the last loop
                    all_boundaries = list(done_indices)
                    if len(all_boundaries) == 0 or all_boundaries[-1] != n_frames - 1:
                        all_boundaries.append(n_frames - 1)
                        
                    for end_idx in all_boundaries:
                        # The episode is valid from start_idx to end_idx (inclusive)
                        # Length = end_idx - start_idx + 1
                        
                        # Generate windows for this segment
                        # Valid starts: range(segment_start, segment_end - window_size + 2, stride)
                        # Example: Ep len 50, window 50. range(0, 0+1) -> [0]. Window 0:50.
                        
                        # Note: HDF5 slicing [start:end] is exclusive at end, so we use end_idx + 1
                        segment_len = (end_idx - start_idx) + 1
                        
                        if segment_len >= self.window_size:
                            num_windows_in_ep = (segment_len - self.window_size) // self.stride + 1
                            
                            for k in range(num_windows_in_ep):
                                global_start = start_idx + (k * self.stride)
                                self.windows.append((file_idx, global_start))
                            
                            total_episodes += 1
                        
                        # Next episode starts after this done
                        start_idx = end_idx + 1
                        
                    total_frames += n_frames
                    
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")

        print(f"Found {len(self.windows)} windows across {total_episodes} valid episodes.")
        print(f"Total raw frames processed: {total_frames}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_idx, start_frame = self.windows[idx]
        file_path = self.h5_files[file_idx]
        end_frame = start_frame + self.window_size

        with h5py.File(file_path, 'r') as f:
            images = f['images'][start_frame:end_frame]  # [T, H, W, 3] (BGR)
            commands = f['commands'][start_frame:end_frame]
            dones = f['dones'][start_frame:end_frame]

            # --- Handle optional is_demo field ---
            if 'is_demo' in f:
                is_demo = f['is_demo'][start_frame:end_frame]
            else:
                # Create zeros matching the sequence length
                is_demo = np.zeros((self.window_size, 1), dtype=np.float32)

        # --- Convert BGR to RGB ---
        # Numpy array slicing is the fastest way to do this
        images = images[..., ::-1].copy()

        # Convert to Torch
        images = torch.from_numpy(images).float() / 255.0
        images = images.permute(0, 3, 1, 2)  # [T, C, H, W]
        
        commands = torch.from_numpy(commands).float()
        dones = torch.from_numpy(dones).float()

        # Ensure is_demo is a tensor
        if isinstance(is_demo, np.ndarray):
            is_demo = torch.from_numpy(is_demo).float()

        return {
            'image': images,
            'action': commands,
            'done': dones,
            'is_demo': is_demo
        }

def create_distributed_dataloader(
    data_dir: str,
    window_size: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    stride: int = 1,
    seed: int = 42,
    split: str = "train",
    train_fraction: float = 0.9,
    split_seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """
    Create DataLoader with DistributedSampler for sharded HDF5 dataset.
    """
    # Create the dataset with a fixed split
    dataset = ShardedHDF5Dataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        split=split,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )

    # Create DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader, sampler, dataset

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Analyze ShardedHDF5Dataset and print statistics"
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/scratch/ja5009/soar_data_sharded/',
        help='Path to sharded HDF5 directory'
    )
    parser.add_argument(
        '--window_size',
        type=int,
        default=96,
        help='Window size for sliding windows'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=1,
        help='Stride for sliding windows'
    )
    parser.add_argument(
        '--show_histogram',
        action='store_true',
        help='Show histogram of episode lengths (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("ShardedHDF5Dataset Analysis")
    print("="*70)
    print(f"\nLoading dataset from: {args.data_dir}")
    print(f"Window size: {args.window_size}")
    print(f"Stride: {args.stride}\n")
    
    # Create dataset
    dataset = ShardedHDF5Dataset(
        data_dir=args.data_dir,
        window_size=args.window_size,
        stride=args.stride,
    )
    
    # Get statistics
    stats = dataset.get_episode_length_statistics()
    
    print("\n" + "="*70)
    print("Episode Length Statistics")
    print("="*70)
    print(f"Total Episodes:           {stats['total_episodes']:,}")
    print(f"Total Timesteps:          {stats['total_timesteps']:,}")
    print(f"Total Windows:            {len(dataset):,}")
    print(f"\nLength Statistics:")
    print(f"  Min:                    {stats['min_length']:.0f} steps")
    print(f"  Max:                    {stats['max_length']:.0f} steps")
    print(f"  Mean:                   {stats['mean_length']:.2f} steps")
    print(f"  Median:                 {stats['median_length']:.2f} steps")
    print(f"  Std Dev:                {stats['std_length']:.2f} steps")
    print(f"\nPercentiles:")
    print(f"  25th percentile:        {stats['percentile_25']:.0f} steps")
    print(f"  75th percentile:        {stats['percentile_75']:.0f} steps")
    print(f"  90th percentile:        {stats['percentile_90']:.0f} steps")
    print(f"  95th percentile:        {stats['percentile_95']:.0f} steps")
    print(f"  99th percentile:        {stats['percentile_99']:.0f} steps")
    
    # Calculate storage efficiency
    avg_windows_per_episode = len(dataset) / stats['total_episodes']
    print(f"\nDataset Efficiency:")
    print(f"  Avg windows per episode: {avg_windows_per_episode:.2f}")
    print(f"  Window coverage:         {avg_windows_per_episode * args.stride / stats['mean_length'] * 100:.1f}%")
    
    # Shard information
    print(f"\nShard Information:")
    print(f"  Number of shards:        {dataset.num_shards}")
    print(f"  Avg episodes per shard:  {stats['total_episodes'] / dataset.num_shards:.1f}")
    
    # Calculate approximate memory usage per batch
    if 'image_shape' in dataset.metadata:
        img_shape = dataset.metadata['image_shape']
        bytes_per_window = (
            args.window_size * img_shape[0] * img_shape[1] * img_shape[2] * 4  # float32
        )
        print(f"\nMemory Usage (per window):")
        print(f"  Image shape:             {img_shape}")
        print(f"  Bytes per window:        {bytes_per_window / (1024**2):.2f} MB")
        print(f"  Batch of 5 windows:      {5 * bytes_per_window / (1024**2):.2f} MB")
    
    print("\n" + "="*70)
    
    # Optional: Show histogram
    if args.show_histogram:
        try:
            import matplotlib.pyplot as plt
            
            lengths = np.array(dataset.episode_lengths)
            
            plt.figure(figsize=(12, 6))
            
            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(lengths, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(stats['mean_length'], color='r', linestyle='--', 
                       label=f"Mean: {stats['mean_length']:.1f}")
            plt.axvline(stats['median_length'], color='g', linestyle='--', 
                       label=f"Median: {stats['median_length']:.1f}")
            plt.xlabel('Episode Length (timesteps)')
            plt.ylabel('Frequency')
            plt.title('Episode Length Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Box plot
            plt.subplot(1, 2, 2)
            plt.boxplot(lengths, vert=True)
            plt.ylabel('Episode Length (timesteps)')
            plt.title('Episode Length Box Plot')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            output_file = Path(args.data_dir) / 'episode_length_analysis.png'
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"\nHistogram saved to: {output_file}")
            
            plt.show()
            
        except ImportError:
            print("\nWarning: matplotlib not installed. Cannot show histogram.")
            print("Install with: pip install matplotlib")
    
    # Test loading a sample
    print("\nTesting data loading...")
    try:
        sample = dataset[0]
        print(f"  Sample shapes:")
        print(f"    Images: {sample['image'].shape}")
        print(f"    Actions: {sample['action'].shape}")
        print(f"  Sample dtypes:")
        print(f"    Images: {sample['image'].dtype}")
        print(f"    Actions: {sample['action'].dtype}")
        print(f"  Image value range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
        print("  ✓ Data loading successful!")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
    
    print("\n" + "="*70)
