#!/usr/bin/env python3
"""
Convert RLDS dataset to NumPy memmap format for efficient PyTorch loading.
This saves each episode as a separate .npy file on disk.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
from tqdm import tqdm
import json

def process_episode(episode, episode_idx, output_dir, verbose=False):
    """
    Process a single episode and save to disk.
    
    Returns:
        dict with episode metadata (length, shapes, etc.)
    """
    steps_ds = episode['steps']
    
    # Collect all steps
    images = []
    actions = []
    
    for step in steps_ds:
        # Decode image
        image = step['observation']['image_0'].numpy()
        images.append(image)
        
        # Get action
        action = step['action'].numpy()
        actions.append(action)
    
    if len(images) == 0:
        if verbose:
            print(f"Warning: Episode {episode_idx} has no steps, skipping")
        return None
    
    # Stack into arrays
    images = np.stack(images, axis=0)  # (T, H, W, C)
    actions = np.stack(actions, axis=0)  # (T, action_dim)
    
    # Save episode data
    episode_file = output_dir / f"episode_{episode_idx:06d}.npz"
    np.savez_compressed(
        episode_file,
        images=images.astype(np.uint8),  # Keep as uint8 to save space
        actions=actions.astype(np.float32),
    )
    
    # Return metadata
    return {
        'episode_idx': episode_idx,
        'length': len(images),
        'image_shape': images.shape,
        'action_shape': actions.shape,
        'file': str(episode_file),
    }

def convert_rlds_to_numpy(
    rlds_dir: str,
    output_dir: str,
    split: str = "success+failure",
    max_episodes: int = None
):
    """
    Convert RLDS dataset to NumPy format.
    
    Args:
        rlds_dir: Path to RLDS dataset directory
        output_dir: Output directory for NumPy files
        split: Dataset split to convert (e.g., "success", "failure", "success+failure")
        max_episodes: Maximum number of episodes to convert (None = all)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading RLDS dataset from: {rlds_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Split: {split}")
    
    # Load RLDS dataset
    builder = tfds.builder_from_directory(rlds_dir)
    ds = builder.as_dataset(split=split)
    
    # Count total episodes (if possible)
    try:
        total_episodes = builder.info.splits[split].num_examples
        print(f"Total episodes in split: {total_episodes}")
    except:
        total_episodes = None
        print("Total episode count unknown")
    
    if max_episodes is not None:
        ds = ds.take(max_episodes)
        total_episodes = min(max_episodes, total_episodes) if total_episodes else max_episodes
    
    # Process episodes
    metadata_list = []
    processed_count = 0
    skipped_count = 0
    
    print("\nProcessing episodes...")
    with tqdm(total=total_episodes, desc="Converting") as pbar:
        for episode_idx, episode in enumerate(ds):
            metadata = process_episode(
                episode, 
                episode_idx, 
                output_path,
                verbose=(episode_idx % 100 == 0)
            )
                
            if metadata is not None:
                metadata_list.append(metadata)
                processed_count += 1
            else:
                skipped_count += 1
                
            pbar.update(1)
                
            # Print progress every 100 episodes
            if (episode_idx + 1) % 100 == 0:
                pbar.set_postfix({
                    'processed': processed_count,
                    'skipped': skipped_count
                })
    
    # Save metadata
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump({
            'num_episodes': processed_count,
            'skipped_episodes': skipped_count,
            'split': split,
            'source_dir': rlds_dir,
            'episodes': metadata_list,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"  Processed: {processed_count} episodes")
    print(f"  Skipped: {skipped_count} episodes")
    print(f"  Output directory: {output_dir}")
    print(f"  Metadata file: {metadata_file}")
    
    # Calculate disk usage
    total_size = sum(
        os.path.getsize(output_path / f"episode_{i:06d}.npz") 
        for i in range(processed_count)
    )
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(
        description="Convert RLDS dataset to NumPy format"
    )
    parser.add_argument(
        '--rlds_dir',
        type=str,
        default='/scratch/ja5009/soar_data/',
        help='Path to RLDS dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/scratch/ja5009/soar_data_numpy/',
        help='Output directory for NumPy files'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='success+failure',
        help='Dataset split to convert (e.g., "success", "failure", "success+failure")'
    )
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=None,
        help='Maximum number of episodes to convert (default: all)'
    )
    
    args = parser.parse_args()
    
    convert_rlds_to_numpy(
        rlds_dir=args.rlds_dir,
        output_dir=args.output_dir,
        split=args.split,
        max_episodes=args.max_episodes,
    )

if __name__ == "__main__":
    main()
