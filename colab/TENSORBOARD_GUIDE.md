# TensorBoard Logging Guide for Google Colab

## What Gets Logged

Your enhanced training script logs the following to TensorBoard:

### 📊 **Metrics (Every Epoch)**
- `train/loss` - Training loss
- `val/loss` - Validation loss
- `val/dice` - Validation Dice score
- `val/iou` - Intersection over Union
- `val/sensitivity` - True positive rate
- `val/precision` - Positive predictive value
- `train/lr` - Learning rate

### 🖼️ **Images (Every 5 Epochs)**
- `val_samples` - 4 sample predictions with [image | ground truth | prediction] side-by-side

### 📈 **Weight Histograms (Every 5 Epochs)**
- All model weight and bias histograms
- Gradient histograms showing how weights are changing

### ⚙️ **Hyperparameters (Start of Training)**
- Batch size, learning rate, optimizer, loss function, augmentation settings
- These appear in TensorBoard's "HPARAMS" tab

---

## Viewing TensorBoard in Google Colab

### **Option 1: Live Viewing During Training (Recommended)**

While training is running, in a **separate new cell**, execute:

```python
%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs'
```

This will display a live TensorBoard dashboard that updates as training progresses. You can:
- ✅ Watch loss curves in real-time
- ✅ Monitor Dice score improving
- ✅ See sample predictions at each 5-epoch checkpoint
- ✅ Track weight histogram changes

### **Option 2: View After Training Completes**

After training finishes, run in a new cell:

```python
%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs'
```

---

## TensorBoard Interface Overview

When you run the above command, you'll see tabs at the top:

### **SCALARS Tab** (Default)
Shows all training metrics:
- **Left panel**: Select which metrics to display (loss, dice, lr, etc.)
- **Right panel**: Interactive line graphs
- **Hover** to see exact values at each epoch
- **Click and drag** to zoom into specific epoch ranges

Example metrics you'll see:
```
Epoch 1:  train_loss=0.8234, val_loss=0.7891, val_dice=0.1523
Epoch 5:  train_loss=0.6234, val_loss=0.5891, val_dice=0.3201
Epoch 30: train_loss=0.2341, val_loss=0.2156, val_dice=0.4823
```

### **IMAGES Tab**
Shows sample predictions:
- **Green boxes** = Ground truth masks
- **Red boxes** = Model predictions
- **Images displayed every 5 epochs**
- Scroll horizontally to see progression

### **DISTRIBUTIONS Tab**
Shows how model weights change over training:
- Weight distributions narrow as training improves
- Gradient distributions show learning magnitude

### **HISTOGRAMS Tab**
Shows weight/gradient distributions at specific epochs:
- Can compare shapes at different training stages
- Helps detect vanishing/exploding gradients

### **HPARAMS Tab**
Summary of all hyperparameters used for this training run:
- Batch size, learning rate, optimizer, loss function
- Useful for comparing different runs

---

## Useful TensorBoard Features

### **Comparing Multiple Runs**
If you train multiple times with different hyperparameters, TensorBoard can compare them:

```python
# Run 1: baseline
%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs'

# Run 2: different batch size
%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs2'
```

### **Download Logs to Your Computer**
After training, save logs to analyze offline:

```python
from google.colab import files
files.download('/content/drive/My Drive/LICENTA_COLAB/logs')
```

Then on your computer, run:
```bash
tensorboard --logdir=logs
```

### **Dark Mode (Optional)**
TensorBoard respects your system's dark mode setting.

---

## Interpreting Your Results

### **Expected Training Curve**
With your lung nodule dataset, you should see:

**Loss (should decrease):**
```
Epoch 1:  train_loss ≈ 0.80
Epoch 10: train_loss ≈ 0.50
Epoch 30: train_loss ≈ 0.25
Epoch 60: train_loss ≈ 0.15
```

**Dice Score (should increase):**
```
Epoch 1:  val_dice ≈ 0.15
Epoch 10: val_dice ≈ 0.30
Epoch 30: val_dice ≈ 0.45
Epoch 60: val_dice ≈ 0.50
```

**Red Flags to Watch For:**
- ❌ Loss not decreasing = learning rate too low or data issue
- ❌ Dice stuck at 0 = data loading problem (check mask validation output)
- ❌ Loss jumping/spiking = learning rate too high
- ❌ Training loss decreasing but val loss increasing = overfitting

---

## Troubleshooting TensorBoard

### **"No scalar data to display"**
- TensorBoard needs at least one epoch to complete
- Try refreshing the page or waiting a moment
- Check that log directory path is correct

### **Images not showing**
- Sample logging happens every 5 epochs
- Wait until epoch 5 to see the first sample
- Check that `val_dataset` isn't empty

### **Want more frequent updates?**
Modify the logging frequency in the training loop:

```python
# Change this line:
if epoch % 5 == 0:  # Logs every 5 epochs
    
# To this for more frequent logging:
if epoch % 1 == 0:  # Logs every epoch (more data, slightly slower)
```

---

## Quick Copy-Paste Commands

### **View training in real-time (during training):**
```python
%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs'
```

### **View after training completes:**
```python
%tensorboard --logdir '/content/drive/My Drive/LICENTA_COLAB/logs'
```

### **Download logs to desktop:**
```python
from google.colab import files
files.download('/content/drive/My Drive/LICENTA_COLAB/logs')
```

### **Clear old logs (start fresh run):**
```python
import shutil
import os
log_dir = '/content/drive/My Drive/LICENTA_COLAB/logs'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    print(f"Cleared {log_dir}")
```

---

## Pro Tips

1. **Run TensorBoard in a separate cell** - keeps it running even if your training cell times out
2. **Take screenshots of key graphs** - save them for your thesis/report
3. **Check weight histograms** - if they stop changing, learning might have stalled
4. **Compare val/train loss** - if training loss keeps dropping but validation plateaus, you're overfitting
5. **Use sample images** - visual inspection of predictions is often more informative than raw metrics

---

## Next Steps

After reviewing your TensorBoard logs:
1. ✅ Verify Dice score reaches ~0.50
2. ✅ Check sample predictions look reasonable
3. ✅ Download best_model.pt from Google Drive
4. ✅ Evaluate on local test set if available
5. ✅ Consider hyperparameter tuning if metrics plateau early

Good luck with your training! 🚀
