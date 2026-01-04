#!/usr/bin/env python3
"""
Convert RLDS to multiple HDF5 shards for optimal multi-node training.
Each shard can be read independently, eliminating contention.
"""

import os
import argparse
import numpy as np
import h5py
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from tqdm import tqdm


def convert_rlds_to_sharded_hdf5(
    rlds_dir: str,
    output_dir: str,
    split: str = "success+failure",
    episodes_per_shard: int = 500,
    max_episodes: int = None,
):
    """
    Convert RLDS to multiple HDF5 shards for distributed training.
    
    Each shard is independent, allowing nodes to read different shards
    without contention on shared storage.
    
    Args:
        episodes_per_shard: Number of episodes per HDF5 file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading RLDS dataset from: {rlds_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Episodes per shard: {episodes_per_shard}")
    
    # Load RLDS dataset
    builder = tfds.builder_from_directory(rlds_dir)
    ds = builder.as_dataset(split=split)
    
    if max_episodes is not None:
        ds = ds.take(max_episodes)
    
    # First pass: analyze structure
    print("\nAnalyzing dataset structure...")
    image_shape = None
    action_shape = None
    
    for episode in ds.take(1):
        steps_ds = episode['steps']
        for step in steps_ds:
            img = step['observation']['image_0'].numpy()
            image_shape = img.shape
            action_shape = step['action'].numpy().shape
            break
    
    print(f"Image shape: {image_shape}")
    print(f"Action shape: {action_shape}")
    
    # Process episodes into shards
    print("\nConverting to sharded HDF5...")
    current_shard = []
    shard_idx = 0
    global_episode_idx = 0
    
    for episode in tqdm(ds, desc="Processing"):
        steps_ds = episode['steps']
        
        # Collect episode data
        images = []
        actions = []
        
        for step in steps_ds:
            image = step['observation']['image_0'].numpy()
            images.append(image)
            actions.append(step['action'].numpy())
        
        if len(images) == 0:
            continue
        
        # Stack episode
        images = np.stack(images, axis=0).astype(np.uint8)
        actions = np.stack(actions, axis=0).astype(np.float32)
        
        current_shard.append({
            'images': images,
            'actions': actions,
            'length': len(images),
        })
        
        global_episode_idx += 1
        
        # Write shard when full
        if len(current_shard) >= episodes_per_shard:
            write_shard(current_shard, output_path, shard_idx, image_shape, action_shape)
            current_shard = []
            shard_idx += 1
    
    # Write remaining episodes
    if len(current_shard) > 0:
        write_shard(current_shard, output_path, shard_idx, image_shape, action_shape)
        shard_idx += 1
    
    # Write metadata
    metadata = {
        'num_shards': shard_idx,
        'episodes_per_shard': episodes_per_shard,
        'total_episodes': global_episode_idx,
        'image_shape': image_shape,
        'action_shape': action_shape,
        'split': split,
    }
    
    import json
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    total_size = sum(
        os.path.getsize(output_path / f"shard_{i:04d}.h5")
        for i in range(shard_idx)
    )
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Total shards: {shard_idx}")
    print(f"  Total episodes: {global_episode_idx}")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


def write_shard(episodes, output_path, shard_idx, image_shape, action_shape):
    """Write a shard of episodes to HDF5."""
    num_episodes = len(episodes)
    max_length = max(ep['length'] for ep in episodes)
    
    shard_file = output_path / f"shard_{shard_idx:04d}.h5"
    
    with h5py.File(shard_file, 'w') as f:
        # Create datasets
        images_ds = f.create_dataset(
            'images',
            shape=(num_episodes, max_length, *image_shape),
            dtype=np.uint8,
            chunks=(1, max_length, *image_shape),
            compression='gzip',
            compression_opts=4,
        )
        
        actions_ds = f.create_dataset(
            'actions',
            shape=(num_episodes, max_length, *action_shape),
            dtype=np.float32,
            chunks=(1, max_length, *action_shape),
            compression='gzip',
            compression_opts=4,
        )
        
        lengths = []
        
        # Write episodes
        for ep_idx, episode in enumerate(episodes):
            ep_length = episode['length']
            lengths.append(ep_length)
            
            images = episode['images']
            actions = episode['actions']
            
            # Pad if needed
            if ep_length < max_length:
                pad_images = np.zeros(
                    (max_length - ep_length, *image_shape),
                    dtype=np.uint8
                )
                images = np.concatenate([images, pad_images], axis=0)
                
                pad_actions = np.zeros(
                    (max_length - ep_length, *action_shape),
                    dtype=np.float32
                )
                actions = np.concatenate([actions, pad_actions], axis=0)
            
            images_ds[ep_idx] = images
            actions_ds[ep_idx] = actions
        
        # Store lengths
        f.create_dataset('episode_lengths', data=np.array(lengths, dtype=np.int32))
        
        # Metadata
        f.attrs['num_episodes'] = num_episodes
        f.attrs['max_length'] = max_length


def main():
    parser = argparse.ArgumentParser(
        description="Convert RLDS to sharded HDF5 for multi-node training"
    )
    parser.add_argument('--rlds_dir', type=str, default='/scratch/ja5009/soar_data/')
    parser.add_argument('--output_dir', type=str, default='/scratch/ja5009/soar_data_sharded/')
    parser.add_argument('--split', type=str, default='success+failure')
    parser.add_argument('--episodes_per_shard', type=int, default=500)
    parser.add_argument('--max_episodes', type=int, default=None)
    
    args = parser.parse_args()
    
    convert_rlds_to_sharded_hdf5(
        rlds_dir=args.rlds_dir,
        output_dir=args.output_dir,
        split=args.split,
        episodes_per_shard=args.episodes_per_shard,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
