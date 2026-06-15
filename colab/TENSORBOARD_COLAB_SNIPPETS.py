%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs'


import shutil
import os

log_dir = '/content/drive/My Drive/LICENTA_COLAB/logs'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print(f"Cleared {log_dir}")
else:
    print(f"Log directory not found: {log_dir}")


from google.colab import files

log_dir = '/content/drive/My Drive/LICENTA_COLAB/logs'
files.download_from_directory(log_dir)


from google.colab import files

model_path = '/content/drive/My Drive/LICENTA_COLAB/results/best_model.pt'
files.download(model_path)


import os
from pathlib import Path

log_dir = Path('/content/drive/My Drive/LICENTA_COLAB/logs')
event_files = list(log_dir.glob('events.out.tfevents.*'))

for f in event_files:
    size_mb = f.stat().st_size / (1024*1024)
    print(f"  {f.name} ({size_mb:.2f} MB)")

if not event_files:
    print("No event files found.")


import json
from pathlib import Path

results_dir = Path('/content/drive/My Drive/LICENTA_COLAB/results')
split_file  = results_dir / 'patient_split.json'
model_file  = results_dir / 'best_model.pt'

if split_file.exists():
    with open(split_file) as f:
        split = json.load(f)
    print(f"Patients: {len(split['train'])} train, {len(split['val'])} val")

if model_file.exists():
    print(f"Checkpoint: {model_file.stat().st_size / 1e6:.2f} MB")
