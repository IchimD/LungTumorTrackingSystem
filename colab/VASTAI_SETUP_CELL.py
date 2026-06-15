import os, subprocess

print("=" * 60)
print("Installing dependencies...")
print("=" * 60)

os.system("pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
os.system("pip install -q SimpleITK scipy tqdm tensorboard matplotlib")
print("✓ Dependencies installed")

print("\n" + "=" * 60)
print("Installing rclone...")
print("=" * 60)

os.system("curl https://rclone.org/install.sh | sudo bash -s 2>/dev/null")
result = os.popen("rclone version").read()
print(f"✓ {result.splitlines()[0] if result else 'rclone installed'}")

print("\n" + "=" * 60)
print("Configuring Google Drive access...")
print("=" * 60)

RCLONE_TOKEN = 'PASTE_YOUR_TOKEN_HERE'

rclone_config = f"""[gdrive]
type = drive
scope = drive
token = {RCLONE_TOKEN}
"""

os.makedirs(os.path.expanduser("~/.config/rclone"), exist_ok=True)
with open(os.path.expanduser("~/.config/rclone/rclone.conf"), "w") as f:
    f.write(rclone_config)

# Test the connection
test = os.popen("rclone lsd gdrive: 2>&1").read()
if "ERROR" in test or "NOTICE" in test.upper() and "PASTE" in RCLONE_TOKEN:
    print("⚠ Token not set — edit RCLONE_TOKEN above and re-run this cell")
else:
    print("✓ Google Drive connected")
    print(f"  Drive root folders: {test[:200]}")

print("\n" + "=" * 60)
print("Copying dataset from Drive to local SSD...")
print("=" * 60)

os.makedirs("/workspace/images", exist_ok=True)
os.makedirs("/workspace/masks",  exist_ok=True)
os.makedirs("/workspace/results", exist_ok=True)
os.makedirs("/workspace/logs",   exist_ok=True)

# Copy images  (note: folder name has a trailing space)
print("\nCopying images...")
os.system("rclone copy 'gdrive:LICENTA_COLAB ' /workspace/images/ --progress --transfers=8 --checkers=16")

# Copy masks
print("\nCopying masks...")
os.system("rclone copy 'gdrive:LICENTA_COLAB /masks' /workspace/masks/ --progress --transfers=8 --checkers=16")

# Copy existing checkpoint so we can resume
print("\nCopying checkpoint (if any)...")
os.system("rclone copy 'gdrive:LICENTA_COLAB/results' /workspace/results/ --progress")

print("\n✓ Dataset copy complete")

# Quick check
n_images = len([f for f in os.listdir("/workspace/images") if f.endswith(".npy")])
n_masks  = len([f for f in os.listdir("/workspace/masks")  if f.endswith(".npy")])
checkpoint_exists = os.path.exists("/workspace/results/best_model.pt")
print(f"  Images : {n_images}")
print(f"  Masks  : {n_masks}")
print(f"  Checkpoint: {'found — will resume' if checkpoint_exists else 'not found — fresh start'}")

print("\n" + "=" * 60)
print("Cloning repository...")
print("=" * 60)

REPO_URL = "https://github.com/IchimD/LungTumorTrackingSystem"
REPO_DIR = "/workspace/LungTumorTrackingSystem"

if os.path.exists(REPO_DIR):
    os.system(f"git -C {REPO_DIR} pull")
    print("✓ Repository updated")
else:
    os.system(f"git clone {REPO_URL} {REPO_DIR}")
    print(f"✓ Repository cloned")

import sys
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

print("\n" + "=" * 60)
print("GPU info")
print("=" * 60)

import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    print(f"✓ GPU  : {gpu.name}")
    print(f"  VRAM : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
else:
    print("⚠ No GPU — make sure you selected a GPU instance on Vast.ai")

print("\n" + "=" * 60)
print("SETUP COMPLETE")
print("=" * 60)
