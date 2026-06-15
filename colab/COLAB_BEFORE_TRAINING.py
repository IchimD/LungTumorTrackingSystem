from IPython.display import Javascript, display

display(Javascript("""
setInterval(() => {
    document.querySelector('body').click();
    console.log('Keep-alive ping: ' + new Date().toLocaleTimeString());
}, 30000);
console.log('Keep-alive started (every 30 s)');
"""))

import gc
import torch

_old_vars = [
    "model", "optimizer", "scheduler", "scaler",
    "criterion", "writer", "train_loader", "val_loader",
    "train_ds", "val_ds",
]
for _v in _old_vars:
    if _v in globals():
        del globals()[_v]

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {free/1e9:.1f} GB free / {total/1e9:.1f} GB total")
else:
    print("No GPU detected — check Runtime > Change runtime type > GPU")

from google.colab import drive
drive.mount("/content/drive", force_remount=False)
print("Drive mounted at /content/drive")
