# TRAIN.md: MOTIP Curriculum Learning Plan

This document outlines the 4-stage transfer learning strategy to train MOTIP from generic DanceTrack weights up to a specialized aerial tracker (VisDrone), using BDD100k as the foundational "ground view" bridge.

## 📋 Overview

Times are based on an **EC2 g5.2xlarge** (Nvidia A10G / 24 GB VRAM):

| Phase | Stage | Dataset | Size (Videos) | Est. Time | Goal | Base Weights |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **I** | **BDD-Foundation** | BDD-Mini | 56 | ~10 hrs | Learn basic road classes & shapes | `motip_dancetrack.pth` |
| **I** | **BDD-Refinement** | BDD-Mini | ~150 | ~24 hrs | Improve Recall & Tracking Stability | `output/bdd_stage1/...` |
| **II** | **Vis-Adaptation** | VisDrone | ~80 | ~12 hrs | Domain Transfer (Ground → Air) | `output/bdd_stage2/...` |
| **II** | **Vis-Specialist** | VisDrone | ~150 | ~24 hrs | Aerial Crowds & Tiny Objects | `output/visdrone_stage1/...` |

---

## 🛠️ Prerequisites

To train successfully on 24GB cards, the codebase must have:
1.  **OOM Protection**: The `try...except RuntimeError` block in `train.py`.
2.  **Memory Flag**: `MEMORY_EFFICIENT: True` in configs.
3.  **Symlink Setup**: `datasets/DanceTrack` should symlink to your dataset build folder (e.g., `output/bdd-mini-dataset`) so strict path checking passes.

---

## 🏁 Phase 1: BDD100k (Ground View)

### Stage 1: BDD Foundation
*Status: Complete ✅*

*Goal: Teach the model "What is a car/pedestrian?" vs background.*

<img width="1500" height="1000" alt="dashboard" src="https://github.com/user-attachments/assets/0f5c8e04-29e3-4f44-b29e-e6caff1ab67e" />

*Results Summary*

Stage 1 successfully established a high-precision baseline on the 56-video mini-dataset (~10 hours). The model demonstrates excellent trustworthiness with DetPr: 89.1% and minimal false positives, proving it has learned to distinguish vehicles from background clutter. However, the tracker remains "shy" and "forgetful," characterized by low recall (DetRe: 46.8%) and frequent ID switching (IDF1: 33.7%). Visual validation confirms this behavior, showing distinct "Blue Box" misses (recall failure) and "Orange Box" identity swaps (association failure), highlighting the critical need for Stage 2 to expand data volume and temporal tracking stability.

1.  **Generate Data**:
    * Set `bdd_num = 56` in `builder.py`.
    * Run: `python builder.py`
2.  **Config**: `configs/pretrain_r50_deformable_detr_bdd_mini.yaml`
    ```yaml
    # Base Architecture (Inherit from DanceTrack)
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    root_path: "."

    # Dataset & Classes
    NUM_CLASSES: 2  # Override default 1 (DanceTrack) to 2 (Pedestrian + Car)

    # Weights Initialization
    DETR_PRETRAIN: ./pretrains/motip_dancetrack.pth
    RESUME_OPTIMIZER: False
    RESUME_SCHEDULER: False

    # Training Strategy
    EPOCHS: 20
    LR: 2.0e-5 
    LR_DROP: 15       # Late drop to allow convergence on new heads
    LR_BACKBONE: 2.0e-6
    LR_WARMUP_EPOCHS: 0

    # Data Sampling (Robustness)
    sampler_lengths: [5]
    sample_intervals: [2, 3] # Skip frames to simulate faster motion

    # Hardware Safety (A10G)
    BATCH_SIZE: 1
    ACCUMULATE_STEPS: 4
    NUM_WORKERS: 4
    MEMORY_EFFICIENT: True
    USE_DECODER_CHECKPOINT: True

    # Memory Optimization: Force smaller images
    AUG_MAX_SIZE: 1000
    AUG_RESIZE_SCALES: [480, 512, 544, 576, 608, 640]
    AUG_RANDOM_RESIZE: [400, 500, 600]

    # Automatic Timestamped Output Directory
    OUTPUT_DIR: null
    OUTPUTS_DIR: null

    val_config:
      GT_FOLDER: "./datasets/DanceTrack/val"
      SEQMAP_FILE: "./datasets/DanceTrack/val_seqmap.txt"
      SPLIT_TO_EVAL: "val"
      CLASSES_TO_EVAL: ['pedestrian', 'car'] 
      CLASS_NAME_TO_ID: 
        pedestrian: 1
        car: 2
    ```
3.  **Run**: `./train-start.sh configs/pretrain_r50_deformable_detr_bdd_mini.yaml`

---

### Stage 2: BDD Refinement
*Status: Planned 🗓️*

*Goal: Scale up data volume to reduce ID switching and improve generalization.*

1.  **Generate Data**:
    * Edit `builder.py`: Set `bdd_num = 150`.
    * Run: `python builder.py` (This adds ~100 new videos).
2.  **Create Config**: `configs/bdd_stage2.yaml`
    ```yaml
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    root_path: "."

    # ⚠️ CRITICAL: Point to the actual result file from Stage 1
    # Example: ./outputs/pretrain_r50_deformable_detr_bdd_mini_20260112_163328/checkpoint_19.pth
    DETR_PRETRAIN: "PATH_TO_STAGE_1_CHECKPOINT.pth"
    RESUME_OPTIMIZER: False
    RESUME_SCHEDULER: False

    NUM_CLASSES: 2

    # Training Strategy (Fine-tuning)
    EPOCHS: 20
    LR: 1.0e-5        # Lower LR for refinement
    LR_DROP: 10       # Drop halfway to settle weights
    LR_BACKBONE: 1.0e-6

    # Hardware & Memory
    BATCH_SIZE: 1
    ACCUMULATE_STEPS: 4
    NUM_WORKERS: 4
    MEMORY_EFFICIENT: True
    USE_DECODER_CHECKPOINT: True

    # Sampling: Shorter clips for speed, but varied intervals for robustness
    sampler_lengths: [3]
    sample_intervals: [1, 2]

    # Memory Constraints
    AUG_MAX_SIZE: 1000
    AUG_RESIZE_SCALES: [480, 512, 544, 576, 608, 640]

    OUTPUT_DIR: null  # Let script generate timestamped folder

    val_config:
      GT_FOLDER: "./datasets/DanceTrack/val"
      SEQMAP_FILE: "./datasets/DanceTrack/val_seqmap.txt"
      SPLIT_TO_EVAL: "val"
      CLASSES_TO_EVAL: ['pedestrian', 'car'] 
      CLASS_NAME_TO_ID: 
        pedestrian: 1
        car: 2
    ```
3.  **Run**: `./train-start.sh configs/bdd_stage2.yaml`

---

## 🚁 Phase 2: VisDrone (Aerial View)

### Stage 3: VisDrone Adaptation
*Goal: Transfer ground-level features to top-down aerial views.*

1.  **Generate Data**:
    * Edit `builder.py`: 
        * `bdd_cfg = {'enabled': False}`
        * `vis_cfg = {'enabled': True, 'num_videos': 80}`
    * Run: `python builder.py` (Replaces BDD data with VisDrone).
2.  **Create Config**: `configs/visdrone_stage1.yaml`
    ```yaml
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    
    # ⚠️ Load from Best BDD Stage 2 Checkpoint
    DETR_PRETRAIN: "PATH_TO_STAGE_2_CHECKPOINT.pth"
    RESUME_OPTIMIZER: False
    
    NUM_CLASSES: 2
    EPOCHS: 20
    LR: 2.0e-5  # Reset LR to learn new domain features
    
    BATCH_SIZE: 1
    ACCUMULATE_STEPS: 4
    sampler_lengths: [2]
    
    MEMORY_EFFICIENT: True
    USE_DECODER_CHECKPOINT: True
    
    # VisDrone objects are tiny; keep resolution higher if memory permits
    AUG_MAX_SIZE: 1100 
    
    OUTPUT_DIR: null
    
    val_config:
      GT_FOLDER: "./datasets/DanceTrack/val"
      SEQMAP_FILE: "./datasets/DanceTrack/val_seqmap.txt"
    ```
3.  **Run**: `./train-start.sh configs/visdrone_stage1.yaml`

---

## 📊 Verification & Visualization

### 1. Qualitative: Render a Video (Test Set)
Visualizing a **Test** sequence (unseen data) is the best sanity check.

```bash
# Example: Render a specific video using the Stage 1 result
python run_video.py \
  --config_path configs/pretrain_r50_deformable_detr_bdd_mini.yaml \
  --checkpoint outputs/pretrain_r50_deformable_detr_bdd_mini_TIMESTAMP/checkpoint_19.pth \
  --video_path "datasets/DanceTrack/val/0224ccfa-4551648a/img1/%08d.jpg" \
  --output_path output/viz_stage1_final.mp4 \
  --score_thresh 0.4 \
  --longterm_patience 30
