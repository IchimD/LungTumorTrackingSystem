# Colab Training Issues - Debugging Guide

## Root Cause: Memory-Mapped Arrays in Multi-Worker DataLoader

### The Problem
- **Local Training (CPU)**: Works fine because file handles are consistent and local filesystem is reliable
- **Colab Training (GPU)**: Memory-mapped arrays fail silently on networked Google Drive
  - Worker processes get stale file handles
  - mmap returns zeros instead of crashing (silent failure)
  - Multi-worker serialization breaks

### The Fix Applied
✅ Changed `np.load(path, mmap_mode='r')` → `np.load(path)` in `VolumeSliceDataset.__getitem__`

This ensures the full array is loaded into memory and properly serialized to worker processes.

---

## Colab-Specific Recommendations

### 1. **Reduce num_workers if Memory is Constrained**
```python
# If you get OOM errors, try:
train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    num_workers=2,  # Try reducing from 4
    pin_memory=True,
)
```

### 2. **Add Data Validation in Colab**
Add this debugging cell to your Colab notebook BEFORE training:

```python
# Test dataset loading
dataset = SafeVolumeSliceDataset(image_dir, mask_dir, patient_ids=train_ids)
print(f"Dataset size: {len(dataset)}")

# Validate 3 random samples
for i in range(3):
    idx = np.random.randint(0, len(dataset))
    img, mask, patient_id = dataset[idx]
    print(f"Sample {i}: Shape={img.shape}, Mask sum={mask.sum():.1f}, Patient={patient_id}")
    
    # Check if masks are actually non-zero
    if mask.sum() == 0:
        print(f"  WARNING: All-zero mask for patient {patient_id}!")
```

### 3. **Monitor DataLoader in Training Loop**
```python
# Add this to your train_one_epoch function
for batch_idx, (images, masks, patient_ids) in enumerate(train_loader):
    if batch_idx == 0:  # First batch
        print(f"Batch mask statistics:")
        print(f"  Min: {masks.min():.4f}, Max: {masks.max():.4f}")
        print(f"  Sum: {masks.sum():.1f}, Mean: {masks.mean():.4f}")
        print(f"  Patient IDs: {patient_ids}")
```

### 4. **Alternative: Use Compressed .npz Files**
If memory is tight, compress masks to .npz (they compress well):
```python
# One-time conversion
import os
mask_dir = "/content/drive/My Drive/LICENTA_COLAB/masks_rclone"
for fname in os.listdir(mask_dir):
    if fname.endswith('.npy'):
        path = os.path.join(mask_dir, fname)
        arr = np.load(path)
        np.savez_compressed(path.replace('.npy', '.npz'), arr)
        os.remove(path)
```

Then `np.load()` will auto-detect .npz and decompress.

---

## Optional: Optimize for Large Datasets

If you have memory issues, implement **lazy loading with caching**:

```python
# In dataset.py - add to VolumeSliceDataset.__init__
self._volume_cache = {}  # patient_id -> (image_volume, mask_volume)
self._cache_max_size = 5  # Keep last 5 patients in memory

def __getitem__(self, index: int):
    image_path, mask_path, patient_id = self._items[index]
    
    # Check cache first
    if patient_id not in self._volume_cache:
        mask_volume = normalize_mask(np.load(mask_path))
        image_volume = np.load(image_path)
        
        self._volume_cache[patient_id] = (image_volume, mask_volume)
        
        # Evict oldest if cache full
        if len(self._volume_cache) > self._cache_max_size:
            self._volume_cache.pop(next(iter(self._volume_cache)))
    else:
        image_volume, mask_volume = self._volume_cache[patient_id]
    
    # Rest of __getitem__ logic...
```

---

## Verification Checklist

After applying the fix, verify in Colab:

- [ ] Dataset loads without errors
- [ ] First batch has non-zero masks: `masks.sum() > 0`
- [ ] Training loss decreases (not constant)
- [ ] Validation Dice > 0 (not 0.0000)
- [ ] Check TensorBoard image logs show correct GT masks

---

## If Still Having Issues

1. **Check file system sync**: Colab may cache Google Drive. Force sync:
   ```python
   !pkill -f 'gvfsd-fuse' || true  # Remount Google Drive
   ```

2. **Verify file integrity** in Colab:
   ```python
   import os
   mask_path = "/content/drive/My Drive/LICENTA_COLAB/masks_rclone/example.npy"
   arr = np.load(mask_path)
   print(f"Shape: {arr.shape}, dtype: {arr.dtype}, min: {arr.min()}, max: {arr.max()}, sum: {arr.sum()}")
   ```

3. **Use rclone directly** (faster/more reliable than Google Drive mount):
   ```bash
   !rclone copy gdrive:LICENTA_COLAB /content/LICENTA_LOCAL --filter "*.npy"
   ```

4. **Enable debug logging** in train_colab.py:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
