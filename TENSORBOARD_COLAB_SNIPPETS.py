"""
QUICK TENSORBOARD COMMANDS FOR GOOGLE COLAB

Copy and paste these commands into your Colab notebook cells.
Run them in the order that makes sense for your workflow.
"""

# ============================================================================
# DURING TRAINING: View TensorBoard Live
# ============================================================================
# Run this in a NEW CELL while your training cell is running in another cell
# This will create a live dashboard that updates as training progresses

%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs'


# ============================================================================
# AFTER TRAINING: View Final Results
# ============================================================================
# If training has already completed, view the final TensorBoard

%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs'


# ============================================================================
# COMPARE MULTIPLE TRAINING RUNS
# ============================================================================
# If you trained with different hyperparameters and have multiple log directories:

%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/'  # View all runs in parent dir


# ============================================================================
# CLEAR OLD LOGS AND START FRESH
# ============================================================================
# If you want to train again and want a clean logs directory:

import shutil
import os

log_dir = '/content/drive/My Drive/LICENTA_COLAB/logs'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print(f"✓ Cleared {log_dir}")
else:
    print(f"Log directory not found: {log_dir}")


# ============================================================================
# DOWNLOAD TENSORBOARD LOGS TO YOUR COMPUTER
# ============================================================================
# After training, save logs to your computer for offline analysis

from google.colab import files

log_dir = '/content/drive/My Drive/LICENTA_COLAB/logs'
print(f"Downloading logs from {log_dir}...")
files.download_from_directory(log_dir)
print("✓ Download complete. Logs saved to your computer.")


# ============================================================================
# DOWNLOAD BEST MODEL CHECKPOINT
# ============================================================================
# Download the trained model to your computer

from google.colab import files

model_path = '/content/drive/My Drive/LICENTA_COLAB/results/best_model.pt'
print(f"Downloading model from {model_path}...")
files.download(model_path)
print("✓ Model downloaded successfully!")


# ============================================================================
# DOWNLOAD PATIENT SPLIT FILE
# ============================================================================
# Download the file showing which patients were used for train/val

from google.colab import files

split_path = '/content/drive/My Drive/LICENTA_COLAB/results/patient_split.json'
print(f"Downloading split file from {split_path}...")
files.download(split_path)
print("✓ Split file downloaded successfully!")


# ============================================================================
# VIEW TENSORBOARD EVENT FILES DIRECTLY
# ============================================================================
# Inspect what's actually in the TensorBoard logs (advanced)

import os
from pathlib import Path

log_dir = Path('/content/drive/My Drive/LICENTA_COLAB/logs')
event_files = list(log_dir.glob('events.out.tfevents.*'))

print(f"TensorBoard event files in {log_dir}:")
for f in event_files:
    size_mb = f.stat().st_size / (1024*1024)
    print(f"  - {f.name} ({size_mb:.2f} MB)")

if not event_files:
    print("  ⚠️  No event files found. Check the log directory path.")


# ============================================================================
# CHECK TRAINING STATUS
# ============================================================================
# Quick check of training progress without TensorBoard

import json
from pathlib import Path

results_dir = Path('/content/drive/My Drive/LICENTA_COLAB/results')
split_file = results_dir / 'patient_split.json'
model_file = results_dir / 'best_model.pt'

print("Training Status Check")
print("=" * 60)

if split_file.exists():
    with open(split_file) as f:
        split = json.load(f)
    print(f"✓ Patients split: {len(split['train'])} train, {len(split['val'])} val")
else:
    print("✗ patient_split.json not found")

if model_file.exists():
    size_mb = model_file.stat().st_size / (1024*1024)
    print(f"✓ Model checkpoint: {size_mb:.2f} MB")
else:
    print("✗ best_model.pt not found")


# ============================================================================
# LOAD AND INSPECT TRAINED MODEL
# ============================================================================
# After training, load your model to check it works

import torch
import sys
sys.path.insert(0, '/content/LungTumorTrackingSystem')

from src.models.unet import UNet2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/content/drive/My Drive/LICENTA_COLAB/results/best_model.pt'

checkpoint = torch.load(checkpoint_path, map_location=device)

model = UNet2D(in_channels=1, out_channels=1, base_channels=32).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✓ Model loaded successfully!")
print(f"  Trained for {checkpoint.get('epoch', '?')} epochs")
print(f"  Best Dice score: {checkpoint.get('val_dice', '?'):.4f}")
print(f"  Device: {device}")

# Test inference on a random tensor
with torch.no_grad():
    test_input = torch.randn(1, 1, 512, 512).to(device)
    test_output = model(test_input)
    print(f"  Test inference shape: {test_output.shape}")
    print(f"  ✓ Model ready for inference!")


# ============================================================================
# ANALYZE TRAINING METRICS FROM EVENT FILES (Advanced)
# ============================================================================
# Read metrics directly from TensorBoard event files without TensorBoard

from torch.utils.tensorboard.summary import Summary
from tensorboard.compat.proto.event_pb2 import Event

import os
from pathlib import Path

log_dir = Path('/content/drive/My Drive/LICENTA_COLAB/logs')

print("Loading TensorBoard metrics...")
print("=" * 60)

metrics = {}

for event_file in log_dir.glob('events.out.tfevents.*'):
    print(f"Reading {event_file.name}...")
    for event in tf.compat.v1.train.summary_iterator(str(event_file)):
        for value in event.summary.value:
            if value.tag not in metrics:
                metrics[value.tag] = []
            if value.HasField('simple_value'):
                metrics[value.tag].append(value.simple_value)

print(f"\nFound {len(metrics)} metrics:")
for tag in sorted(metrics.keys()):
    values = metrics[tag]
    print(f"  {tag}: {len(values)} values (last={values[-1]:.4f})")


# ============================================================================
# EXPORT METRICS TO CSV FOR ANALYSIS
# ============================================================================
# Save metrics as CSV for plotting in Excel/Python

import pandas as pd
import torch
from tensorboard.compat.proto.event_pb2 import Event

log_dir = '/content/drive/My Drive/LICENTA_COLAB/logs'

metrics_data = []

for event_file in os.listdir(log_dir):
    if event_file.startswith('events.out.tfevents'):
        event_path = os.path.join(log_dir, event_file)
        for event in tf.compat.v1.train.summary_iterator(event_path):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    metrics_data.append({
                        'tag': value.tag,
                        'step': event.step,
                        'value': value.simple_value,
                        'wall_time': event.wall_time,
                    })

df = pd.DataFrame(metrics_data)

# Save to CSV
csv_path = '/content/drive/My Drive/LICENTA_COLAB/results/metrics.csv'
df.to_csv(csv_path, index=False)
print(f"✓ Saved metrics to {csv_path}")

# Show summary
print("\nMetrics Summary:")
for tag in df['tag'].unique():
    tag_data = df[df['tag'] == tag]
    print(f"  {tag}: {len(tag_data)} entries, last value = {tag_data['value'].iloc[-1]:.4f}")

from google.colab import files
files.download(csv_path)
