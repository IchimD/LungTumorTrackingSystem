# U-Net Training Module

This package contains everything needed to train a 2D U-Net segmentation network on preprocessed lung CT volumes.

## Components

### `train.py`
Main training loop with checkpointing, validation, and metrics tracking.

```bash
python -m src.training.train \
  --image_dir data/processed/images \
  --mask_dir data/processed/masks \
  --output_dir checkpoints \
  --epochs 40 \
  --batch_size 8 \
  --lr 0.001 \
  --augment
```

**Arguments:**
- `--image_dir`: Path to preprocessed image volumes (.npy files)
- `--mask_dir`: Path to binary mask volumes (.npy files)
- `--output_dir`: Where to save checkpoints and logs
- `--epochs`: Number of training epochs (default: 40)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 0.001)
- `--val_fraction`: Fraction of patients for validation (default: 0.15)
- `--augment`: Enable light data augmentation (flip, intensity jitter)
- `--seed`: Random seed for reproducible splits (default: 42)

**Output:**
- `best_model.pt`: Best model checkpoint (selected by validation Dice score)
- `patient_split.json`: Train/val patient IDs used

### `evaluate.py`
Load a checkpoint and compute metrics (Dice, IoU, Sensitivity, Precision) on a dataset.

```bash
python -m src.training.evaluate \
  --checkpoint checkpoints/best_model.pt \
  --image_dir data/processed/images \
  --mask_dir data/processed/masks \
  --output_dir results
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--image_dir`: Image directory
- `--mask_dir`: Mask directory
- `--output_dir`: Where to save results JSON
- `--patient_ids`: Optional list of specific patient IDs to evaluate
- `--threshold`: Probability threshold (default: 0.5)

**Output:**
- `evaluation_results.json`: Per-slice metrics and summary statistics

### `inference.py`
Run inference on a preprocessed volume and generate predictions.

```bash
python -m src.training.inference \
  --checkpoint checkpoints/best_model.pt \
  --image_path data/processed/images/patient_001.npy \
  --output_path predictions/patient_001_mask.npy
```

**Arguments:**
- `--checkpoint`: Path to model checkpoint
- `--image_path`: Path to preprocessed volume
- `--output_path`: Where to save predicted mask
- `--threshold`: Probability threshold (default: 0.5)
- `--batch_size`: Inference batch size (default: 8)

### `dataset.py`
`VolumeSliceDataset` — PyTorch Dataset that extracts 2D axial slices from preprocessed volumes.

- Supports patient filtering
- Class balance: keeps all lesion slices, samples a subset of background slices
- Optional augmentation (flip, intensity jitter)
- Works with .npy, .npz, and .mhd files

### `loss.py`
- `DiceLoss`: Standard Dice coefficient loss for segmentation
- `BCEDiceLoss`: Weighted combination of BCE and Dice losses (default: 50/50)

### `metrics.py`
- `dice_score()`: Dice coefficient
- `iou_score()`: Intersection over Union
- `sensitivity_score()`: True Positive Rate (Recall)
- `precision_score()`: Positive Predictive Value

All metrics use a configurable probability threshold (default: 0.5).

### `src/models/unet.py`
2D U-Net architecture with:
- 5 encoder/decoder levels
- Skip connections
- BatchNorm after each convolution
- Bilinear upsampling
- No activation in final layer (handled by loss function)

### `src/data/splits.py`
Generate and save train/val/test patient splits by patient ID.

```bash
python -m src.data.splits \
  --image_dir data/processed/images \
  --mask_dir data/processed/masks \
  --output_dir data/splits \
  --train_fraction 0.7 \
  --val_fraction 0.15 \
  --seed 42
```

**Output:**
- `data/splits/patient_splits.json`: Train/val/test patient IDs

### `src/data/augmentation.py`
Light augmentation for training:
- Random horizontal/vertical flip (50% probability)
- Random intensity jitter (20% probability, ±5% gain, ±0.05 shift)

## Typical Workflow

1. **Preprocess data** (create image and mask .npy files)
   ```bash
   python -m src.data.preprocess \
     --subset D:/LICENTA2/DATASET/subset0 \
     --csv D:/LICENTA2/DATASET/annotations.csv \
     --output_dir data/processed
   ```

2. **Create patient splits**
   ```bash
   python -m src.data.splits \
     --image_dir data/processed/images \
     --mask_dir data/processed/masks \
     --output_dir data/splits
   ```

3. **Train the model**
   ```bash
   python -m src.training.train \
     --image_dir data/processed/images \
     --mask_dir data/processed/masks \
     --output_dir checkpoints \
     --epochs 40 \
     --augment
   ```

4. **Evaluate on test set**
   ```bash
   python -m src.training.evaluate \
     --checkpoint checkpoints/best_model.pt \
     --image_dir data/processed/images \
     --mask_dir data/processed/masks \
     --output_dir results
   ```

5. **Generate predictions**
   ```bash
   python -m src.training.inference \
     --checkpoint checkpoints/best_model.pt \
     --image_path data/processed/images/some_patient.npy \
     --output_path predictions/some_patient_mask.npy
   ```

## Expected Results

- **Validation Dice > 0.7** (good threshold for medical imaging)
- **Train/val Dice gap < 0.1** (no severe overfitting)
- **Training time ~2-5 mins per epoch** on modern GPU, ~30 mins on CPU

## Key Features

✅ 2D axial slices (512×512) extracted from 3D volumes  
✅ Class balance: keep positive slices, sample background slices  
✅ Train/val/test split by patient (no patient leakage)  
✅ Dice + BCE loss for imbalanced segmentation  
✅ Light augmentation (flip + intensity jitter)  
✅ Checkpoint saving (best model by validation Dice)  
✅ Per-slice and aggregate metrics  
✅ No overfitting (BatchNorm regularization)  

## References

- U-Net: [Ronneberger et al. (2015)](https://arxiv.org/abs/1505.04597)
- Dice Loss: [Milletari et al. (2016)](https://arxiv.org/abs/1606.06650)
- LUNA16 dataset: [Setio et al. (2016)](https://arxiv.org/abs/1607.06021)
