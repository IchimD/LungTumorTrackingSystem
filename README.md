# Lung Tumor Tracking System in Radiotherapy

**Bachelor Degree Project**

---

## Project Overview

This project builds a deep-learning pipeline to automatically **segment** and **track** lung tumors across different breathing phases from medical imaging data (CT scans). In radiotherapy, the tumor moves with each breath; knowing its exact position at every phase is critical for precise dose delivery and minimizing damage to healthy tissue.

**Pipeline summary:**
1. Train a U-Net from scratch to semantically segment the tumor in each image slice/phase.
2. Use the segmentation output to track the tumor's 3D position across the full breathing cycle.

---

## Project Plan

### Phase 0 — Background Research & Setup
**Goal:** Understand the domain and establish the development environment.

| Task | Details |
|------|---------|
| Literature review | Study U-Net (Ronneberger et al., 2015), 4D-CT concepts, respiratory motion models, and relevant radiotherapy segmentation papers |
| Dataset survey | Identify publicly available datasets: DIR-Lab (4D CT with expert landmarks), TCIA RIDER Lung CT, LUNA16 |
| Environment setup | Python 3.10+, PyTorch, SimpleITK, nibabel, NumPy, OpenCV, Matplotlib, scikit-learn, MLflow (experiment tracking) |
| Repository structure | Define folder layout (see below) |

**Deliverable:** Literature summary, selected dataset, working Python environment, initial repo structure.

---

### Phase 1 — Data Acquisition & Preprocessing
**Goal:** Produce clean, normalised, annotated image/mask pairs ready for training.

| Task | Details |
|------|---------|
| 1.1 Data collection | Download 4D-CT dataset (e.g., DIR-Lab: 10 patients × 10 breathing phases) with doctor-provided tumor contours |
| 1.2 DICOM → NumPy | Convert DICOM/NIfTI files to NumPy arrays using SimpleITK; preserve voxel spacing metadata |
| 1.3 Mask generation | Convert doctor-annotated contours (RTStruct / landmark files) into binary segmentation masks |
| 1.4 Intensity normalisation | Apply window/level (lung window: W=1500, L=-600) and min-max or z-score normalisation |
| 1.5 Resampling | Resample all volumes to a uniform voxel spacing (e.g., 1×1×1 mm) |
| 1.6 Patch / slice extraction | Extract 2D slices or 3D patches centred on lung region; handle class imbalance (tumor << background) |
| 1.7 Data augmentation | Random flips, rotations (±15°), elastic deformations, intensity jitter, Gaussian noise |
| 1.8 Dataset split | 70% train / 15% validation / 15% test — split by **patient**, not by slice |

**Deliverable:** Preprocessed dataset saved as `.npy` or HDF5; dataset statistics report; sample visualisations.

---

### Phase 2 — U-Net Semantic Segmentation (from Scratch)
**Goal:** Train a U-Net to produce a binary mask (tumor vs. background) for each input image.

#### 2.1 Architecture Design
- Standard encoder–decoder U-Net with skip connections
- Encoder: 4 down-sampling blocks (Conv→BN→ReLU ×2, MaxPool)
- Bottleneck: 2× Conv block
- Decoder: 4 up-sampling blocks (Bilinear upsample / TransposedConv, skip concat, Conv→BN→ReLU ×2)
- Output: 1×H×W sigmoid map (binary segmentation)
- All weights initialised from scratch (no pretrained backbone)

#### 2.2 Loss Function
- **Primary:** Dice Loss + Binary Cross-Entropy (combined)
- **Optional:** Focal Loss to handle extreme class imbalance

#### 2.3 Training Setup
| Hyperparameter | Starting value |
|----------------|---------------|
| Input size | 256×256 (2D slices) |
| Batch size | 8–16 |
| Optimiser | Adam (lr=1e-4) |
| LR scheduler | ReduceLROnPlateau |
| Epochs | 100–150 with early stopping |
| Hardware | GPU (CUDA) |

#### 2.4 Evaluation Metrics
- **Dice Similarity Coefficient (DSC)** — primary metric
- **Intersection over Union (IoU)**
- **Hausdorff Distance (HD95)** — boundary accuracy
- **Sensitivity / Specificity**

#### 2.5 Experiments & Ablations
- Baseline 2D U-Net vs. 2.5D (multi-slice input) vs. 3D U-Net
- Loss function comparison
- Augmentation impact study

**Deliverable:** Trained model checkpoint; training curves; quantitative results table; qualitative segmentation visualisations.

---

### Phase 3 — Tumor Tracking Across Breathing Phases
**Goal:** Use segmentation masks from all phases to track the tumor's position through the breathing cycle.

| Task | Details |
|------|---------|
| 3.1 Apply segmentation | Run trained U-Net on all 10 breathing phases for each patient → 10 binary mask volumes |
| 3.2 Centroid extraction | Compute 3D centroid (x, y, z) and bounding box from each phase mask using connected-component analysis |
| 3.3 Trajectory construction | Build a trajectory curve: centroid position as a function of breathing phase index |
| 3.4 Motion modelling | Fit a smooth parametric model (B-spline or sinusoidal) to the trajectory for interpolation between phases |
| 3.5 Tracking evaluation | Compare predicted centroids to doctor-provided ground-truth landmarks; compute **Target Registration Error (TRE)** in mm |
| 3.6 Visualisation | Animate tumor position overlay across all breathing phases; plot 3D motion trajectory |

**Deliverable:** Per-patient tracking trajectories; TRE statistics; motion visualisation videos/GIFs.

---

### Phase 4 — Integration & System Pipeline
**Goal:** Connect all components into a single end-to-end pipeline.

```
Input 4D-CT (DICOM)
        │
        ▼
 Preprocessing module
        │
        ▼
  U-Net segmentation
  (per-phase masks)
        │
        ▼
 Centroid extraction
        │
        ▼
 Motion model fitting
        │
        ▼
 Tracked trajectory → report / visualisation
```

- Write a CLI/script (`run_pipeline.py`) that accepts a patient folder and outputs masks + trajectory
- Logging with MLflow or TensorBoard

**Deliverable:** Working end-to-end script; example output report for one patient.

---

### Phase 5 — Evaluation, Writing & Presentation
**Goal:** Validate results, write the thesis, prepare the defence.

| Task | Details |
|------|---------|
| 5.1 Full test-set evaluation | Report DSC, IoU, HD95, TRE on held-out test patients |
| 5.2 Comparison | Compare against a simple baseline (threshold-based segmentation, or centroid from largest blob) |
| 5.3 Error analysis | Identify failure cases (low-contrast tumors, large motion artifacts) |
| 5.4 Thesis writing | Introduction, Related Work, Methodology, Experiments, Conclusions |
| 5.5 Defence preparation | Slides, demo |

---

## Proposed Repository Structure

```
LungTumorTrackingSystem/
├── data/
│   ├── raw/                  # Original DICOM / NIfTI files
│   ├── processed/            # Resampled volumes and masks (.npy / HDF5)
│   └── splits/               # train/val/test patient lists
├── src/
│   ├── data/
│   │   ├── preprocess.py     # DICOM→NumPy, normalisation, resampling
│   │   ├── dataset.py        # PyTorch Dataset class
│   │   └── augmentation.py   # Augmentation pipeline
│   ├── models/
│   │   └── unet.py           # U-Net architecture (from scratch)
│   ├── training/
│   │   ├── train.py          # Training loop
│   │   ├── loss.py           # Dice + BCE loss
│   │   └── metrics.py        # DSC, IoU, HD95
│   ├── tracking/
│   │   ├── extract_centroid.py
│   │   └── motion_model.py   # B-spline / sinusoidal fitting
│   └── visualisation/
│       └── plot.py
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_tracking_results.ipynb
├── configs/
│   └── config.yaml           # All hyperparameters in one place
├── run_pipeline.py           # End-to-end script
├── requirements.txt
└── README.md
```

---

## Technology Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Deep learning | PyTorch + torchvision |
| Medical imaging I/O | SimpleITK, nibabel, pydicom |
| Data handling | NumPy, pandas, h5py |
| Visualisation | Matplotlib, seaborn, napari |
| Experiment tracking | MLflow / TensorBoard |
| Evaluation | scikit-learn, surface-distance |

---

## Key Milestones

```
Week 1–2   Phase 0  — Research & environment setup
Week 3–5   Phase 1  — Data preprocessing pipeline
Week 6–10  Phase 2  — U-Net design, training & evaluation
Week 11–13 Phase 3  — Tracking system
Week 14    Phase 4  — Integration
Week 15–16 Phase 5  — Evaluation, thesis writing, defence prep
```

---

## References (starting point)
- Ronneberger, O. et al. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI.
- Castillo, R. et al. (2009). *DIR-Lab 4D CT dataset.* Medical Physics.
- Milletari, F. et al. (2016). *V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation.*
- Fu, Y. et al. (2021). *A review of deep learning-based methods for medical image segmentation.* Neurocomputing.
