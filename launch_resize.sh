#!/bin/bash
input_dir="/scratch/ja5009/soar_data_sharded"
output_dir="/scratch/ja5009/soar_data_sharded_128x128"
target_size="128 128"
num_shards=62  # Adjust if needed
n_procs=62
shards_per_proc=$(( (num_shards + n_procs - 1) / n_procs ))  # Ceiling division

for i in $(seq 0 $((n_procs-1))); do
  start=$((i * shards_per_proc))
  end=$(((i + 1) * shards_per_proc - 1))
  end=$((end < num_shards ? end : num_shards - 1))
  echo "Launching process $i: shards $start to $end"
  python resize_dataset.py --input_dir "$input_dir" --output_dir "$output_dir" \
    --target_size $target_size --shard_start $start --shard_end $end &
done
wait
echo "All processes complete."
