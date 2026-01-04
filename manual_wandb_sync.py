import wandb
from tbparse import SummaryReader
import time
import pandas as pd

# CONFIGURATION
# USE A NEW ID so you don't mess up the original run with duplicates
WANDB_RUN_ID = "6x2ee8g2_01"
PROJECT_NAME = "dreamer-v4-tokenizer"
LOG_DIR = "./logs/dreamer_v4_dynamics/2025-11-30_08-05-07/tensorboard/"

wandb.init(project=PROJECT_NAME, id=WANDB_RUN_ID, resume="allow")
wandb.define_metric("global_step")
wandb.define_metric("*", step_metric="global_step")

last_uploaded_step = -1

print("Starting live sync monitor...")

print(f"Scanning {LOG_DIR}...")
reader = SummaryReader(LOG_DIR)
    
# Optimisation: tbparse reads everything, so this gets slower as training grows.
# Ideally, you'd only read new files, but tbparse simplifies this complexity.
df = reader.scalars
    
# Filter for ONLY new steps
new_data = df[df['step'] > last_uploaded_step]
    
if not new_data.empty:
    print(f"Found {len(new_data)} new scalar points. Syncing...")
    
    # Group by step to create valid wandb packets
    for step, group in new_data.groupby('step'):
        metrics = {row['tag']: row['value'] for _, row in group.iterrows()}
        metrics['global_step'] = step
        wandb.log(metrics)
    
    last_uploaded_step = new_data['step'].max()
    print(f"Synced up to step {last_uploaded_step}")
else:
    print("No new data found.")

wandb.finish()
