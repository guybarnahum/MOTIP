# TRAIN.md: MOTIP Curriculum Learning Plan

This document outlines the 4-stage transfer learning strategy to train MOTIP from generic DanceTrack weights up to a specialized aerial tracker (VisDrone), using BDD100k as the foundational "ground view" bridge.



## 📋 Overview

Times are on EC2 g5.2xlarge / Nvidia A10G / 24 GB GPU:

| Phase | Stage | Dataset | Size (Videos) | Est. Time | Goal | Base Weights |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **I** | **BDD-Foundation** | BDD100k | ~80 | 300 min | Learn basic road classes | `motip_dancetrack.pth` |
| **I** | **BDD-Refinement** | BDD100k | ~150 | 300 min | Robustness & Convergence | `output/bdd_stage1` |
| **II** | **Vis-Adaptation** | VisDrone | ~80 | 300 min | Domain Transfer (Ground → Air) | `output/bdd_stage2` |
| **II** | **Vis-Specialist** | VisDrone | ~150 | 300 min | Aerial Crowds & Tiny Objects | `output/visdrone_stage1` |

---

## 🛠️ Prerequisites

There are important Memory optimizations in train.py that must be included for traning to work on 24GB cards, the original repro did not have them:

1.  **OOM Protection**: Ensure `train.py` has the OOM `try...except` block installed.
2.  **Memory Flag**: Ensure `train.py` reads `MEMORY_EFFICIENT` from config.

---

## 🏁 Phase 1: BDD100k (Ground View)

### Stage 1: BDD Foundation
*Teach the model "What is a car/pedestrian?"*

1.  **Generate Data**:
    * Edit `builder.py`: Set `bdd_num = 80`.
    * Run: `python builder.py`
2.  **Create Config**: `configs/bdd_stage1.yaml`
    ```yaml
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    DETR_PRETRAIN: ./pretrains/motip_dancetrack.pth
    RESUME_OPTIMIZER: False
    NUM_CLASSES: 2
    EPOCHS: 20
    LR: 2.0e-5
    BATCH_SIZE: 1
    ACCUMULATE_STEPS: 4
    sampler_lengths: [2]
    MEMORY_EFFICIENT: True
    USE_DECODER_CHECKPOINT: True
    OUTPUT_DIR: output/bdd_stage1
    val_config:
      GT_FOLDER: "./output/bdd-mini-dataset/val"
      SEQMAP_FILE: "./output/bdd-mini-dataset/val_seqmap.txt"
    ```
3.  **Run**: `./train-start.sh configs/bdd_stage1.yaml`

### Stage 2: BDD Refinement
*Improve accuracy and reduce ID switching.*

1.  **Generate Data**:
    * Edit `builder.py`: Set `bdd_num = 150`.
    * Run: `python builder.py` (It will add the new videos).
2.  **Create Config**: `configs/bdd_stage2.yaml`
    ```yaml
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    # ⚠️ Load from Stage 1 Result
    DETR_PRETRAIN: ./output/bdd_stage1/checkpoint.pth
    RESUME_OPTIMIZER: False
    NUM_CLASSES: 2
    EPOCHS: 20
    LR: 1.0e-5  # Lower LR for fine-tuning
    BATCH_SIZE: 1
    ACCUMULATE_STEPS: 4
    sampler_lengths: [2]
    MEMORY_EFFICIENT: True
    USE_DECODER_CHECKPOINT: True
    OUTPUT_DIR: output/bdd_stage2
    val_config:
      GT_FOLDER: "./output/bdd-mini-dataset/val"
      SEQMAP_FILE: "./output/bdd-mini-dataset/val_seqmap.txt"
    ```
3.  **Run**: `./train-start.sh configs/bdd_stage2.yaml`

---

## 🚁 Phase 2: VisDrone (Aerial View)

### Stage 3: VisDrone Adaptation
*Transfer knowledge from Ground View to Top-Down View.*

1.  **Generate Data**:
    * Edit `builder.py`: 
        * `bdd_cfg = {'enabled': False}`
        * `vis_cfg = {'enabled': True, 'num_videos': 80}`
    * Run: `python builder.py` (Replaces data with VisDrone).
2.  **Create Config**: `configs/visdrone_stage1.yaml`
    ```yaml
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    # ⚠️ Load from BDD Stage 2 Result (Best Ground Model)
    DETR_PRETRAIN: ./output/bdd_stage2/checkpoint.pth
    RESUME_OPTIMIZER: False
    NUM_CLASSES: 2
    EPOCHS: 20
    LR: 2.0e-5  # Reset LR to learn new domain features
    BATCH_SIZE: 1
    ACCUMULATE_STEPS: 4
    sampler_lengths: [2]
    MEMORY_EFFICIENT: True
    USE_DECODER_CHECKPOINT: True
    # VisDrone objects are tiny; keep resolution high if memory permits
    train_transform:
      max_size: 1000
    OUTPUT_DIR: output/visdrone_stage1
    val_config:
      GT_FOLDER: "./output/bdd-mini-dataset/val"
      SEQMAP_FILE: "./output/bdd-mini-dataset/val_seqmap.txt"
    ```
3.  **Run**: `./train-start.sh configs/visdrone_stage1.yaml`

### Stage 4: VisDrone Specialist
*Maximize performance on tiny aerial objects.*

1.  **Generate Data**:
    * Edit `builder.py`: Set `vis_num = 150`.
    * Run: `python builder.py`.
2.  **Create Config**: `configs/visdrone_stage2.yaml`
    ```yaml
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    # ⚠️ Load from VisDrone Stage 1 Result
    DETR_PRETRAIN: ./output/visdrone_stage1/checkpoint.pth
    RESUME_OPTIMIZER: False
    NUM_CLASSES: 2
    EPOCHS: 20
    LR: 1.0e-5  # Low LR for final polish
    BATCH_SIZE: 1
    ACCUMULATE_STEPS: 4
    sampler_lengths: [2]
    MEMORY_EFFICIENT: True
    USE_DECODER_CHECKPOINT: True
    OUTPUT_DIR: output/visdrone_stage2
    val_config:
      GT_FOLDER: "./output/bdd-mini-dataset/val"
      SEQMAP_FILE: "./output/bdd-mini-dataset/val_seqmap.txt"
    ```
3.  **Run**: `./train-start.sh configs/visdrone_stage2.yaml`

---

## 📊 Verification & Visualization

After any stage, verify performance in two ways: quantitatively (Scores) and qualitatively (Video).

### 1. Quantitative: Get HOTA/MOTA Scores (Validation Set)
Run the evaluation script to see how well the model is performing on the validation split. This uses the `val_config` settings already defined in your YAML files.

```bash
# Example: Evaluate Stage 2 BDD on Validation Set
python train.py \
  --config-path configs/bdd_stage2.yaml \
  --exp-name eval_bdd_stage2 \
  --resume output/bdd_stage2/checkpoint.pth \
  --eval
```
*Check the console logs or `output/eval_bdd_stage2/` for the HOTA/MOTA summary.*

### 2. Qualitative: Render a Video (Test Set)
Visualizing a **Test** sequence (unseen data) is the best sanity check to ensure the tracker works on new data.

1.  **Find a Test Video ID:**
    ```bash
    ls output/bdd-mini-dataset/test
    # Copy one ID, e.g., '0184a4be-c7a32fbc'
    ```
2.  **Render the Video:**
    ```bash
    python render_video.py \
      --config_path configs/bdd_stage2.yaml \
      --checkpoint output/bdd_stage2/checkpoint.pth \
      --video_path "./output/bdd-mini-dataset/test/<PASTE_ID_HERE>/img1/%08d.jpg" \
      --output_path output/viz_bdd_stage2.mp4 \
      --longterm_patience 300 \
      --fp16
    ```
