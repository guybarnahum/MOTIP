# TRAIN.md: MOTIP Training Plan

We employ **Joint Training Strategy** designed to prevent catastrophic forgetting. Unlike the previous attempt—which flooded the model with cars and erased its person-tracking knowledge—this plan uses the **Frame Budget** system to maintain a balanced diet of "People" and "Vehicles" throughout all stages.

## 📋 Strategic Overview

**Hardware Reality:** EC2 g5.2xlarge (A10G). Observed speed is **~2.0 hours/epoch** (Stage 1) to **~6.5 hours/epoch** (Stage 2) due to dataset scaling.
**Strategy:** We utilize "sleeping compute" (overnight/weekend runs) with optimized epoch counts to ensure convergence.
**Cost Basis:** ~$1.212 / hr (On-Demand).

| Phase | Stage | Dataset Mix (Frame Budget) | Epochs | Est. Time | Est. Cost | Goal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **I** | **Foundation** | 5k BDD (Cars)<br>5k DanceTrack (People) | **10** | **~20.5 hrs** | ~$25.00 | Rapidly learn "Vehicle" class without forgetting "Person". |
| **II** | **Refinement** | 15k BDD (Cars)<br>15k DanceTrack (People) | **10** | **~65.0 hrs** | ~$79.00 | Scale up volume (3x) to fix "Ghost Boxes" and stabilize tracking. |
| **III** | **Aerial-Adaptation** | 15k VisDrone (Air)<br>5k DT/BDD (Replay) | **10** | **~45.0 hrs** | ~$55.00 | Adapt to aerial view while retaining ground object features. |

### 🛡️ Dataset Integrity: Fixed Val/Test Splits
To ensure valid comparisons between stages, we modified `builder.py` to enforce a **Manifest Lock**.

* **The Mechanism:** The builder loads a `manifest.json` file defining specific video IDs for Validation and Test sets. These IDs are "locked" and excluded from the random selection pool before the training budget is filled.
* **The Benefit:** Even when we triple the training data (Stage 1 $\to$ Stage 2) or change the random seed, **the Validation Set remains bit-exact identical**.
* **Why it matters:** This prevents "Data Leakage" (training on validation data) and guarantees that any improvement in metrics (MOTA/IDF1) is due to model learning, not easier evaluation data.
  
---

## 🛠️ Prerequisites

1.  **Codebase:** Ensure `builder.py` has the new **Frame Budget** logic.
2.  **Config:** Ensure `MEMORY_EFFICIENT: True` is set for 24GB VRAM.
3.  **Data:** Ensure `datasets/DanceTrack` symlinks to your built dataset output.

---

## 🏁 Phase 1: Ground-Level Universality

### Stage 1: Balanced Foundation
*Goal: Multi-class tracking (Person + Vehicle) with equal exposure.*

Instead of showing the model 100% cars, we show it a 50/50 mix. This forces the gradients to optimize for both classes simultaneously.

1.  **Generate Data (Balanced Mix)**:
    * Edit `config.toml` to set equal budgets:
        ```toml
        [bdd]
        enabled = true
        frame_budget = 5000  # ~1000 training samples

        [dancetrack]
        enabled = true
        frame_budget = 5000  # ~1000 training samples
        ```
    * Run: `python builder.py`

2.  **Config**: `configs/stage1_universal_foundation.yaml`
    ```yaml
    # Inherit from DanceTrack baseline
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    root_path: "."

    # ⚠️ CRITICAL: 2 Classes (1=Person, 2=Vehicle)
    NUM_CLASSES: 2

    # Weights: Start from strong Person tracker
    DETR_PRETRAIN: ./pretrains/motip_dancetrack.pth
    RESUME_OPTIMIZER: False
    RESUME_SCHEDULER: False

    # Training Strategy (Overnight Run)
    EPOCHS: 10
    LR: 2.0e-5
    LR_DROP: 7      # Drop late (70% of training) to allow settling
    LR_BACKBONE: 2.0e-6

    # Hardware Safety (A10G)
    BATCH_SIZE: 1
    ACCUMULATE_STEPS: 4
    NUM_WORKERS: 4
    MEMORY_EFFICIENT: True
    USE_DECODER_CHECKPOINT: True

    # Memory Optimization
    AUG_MAX_SIZE: 1000
    AUG_RESIZE_SCALES: [480, 512, 544, 576, 608, 640]

    # Output
    OUTPUT_DIR: null

    # Validation: Evaluate on DanceTrack to ensure NO REGRESSION
    val_config:
      GT_FOLDER: "./datasets/DanceTrack/val"
      SEQMAP_FILE: "./datasets/DanceTrack/val_seqmap.txt"
      SPLIT_TO_EVAL: "val"
      # We check 'pedestrian' specifically to monitor forgetting
      CLASSES_TO_EVAL: ['pedestrian'] 
      CLASS_NAME_TO_ID: 
        pedestrian: 1
    ```

3.  **Run**: `./train-start.sh configs/stage1_universal_foundation.yaml`

---

### Stage 2: Scale & Refine
*Goal: Improved association stability via larger dataset volume.*

1.  **Generate Data (Scale Up)**:
    * Edit `config.toml`: Double the budgets.
        ```toml
        [bdd]
        frame_budget = 15000
        [dancetrack]
        frame_budget = 15000
        ```
    * Run: `python builder.py`

2.  **Config**: `configs/stage2_universal_refine.yaml`
    ```yaml
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    
    # Load Stage 1 Result
    DETR_PRETRAIN: "./pretrains/stage1_v2.pth"
    RESUME_OPTIMIZER: False

    NUM_CLASSES: 2
    
    # Refinement Strategy
    EPOCHS: 10
    LR: 1.0e-5  # Lower LR for refinement
    LR_DROP: 7
    ```

3.  **Run**: `./train-start.sh configs/stage2_universal_refine.yaml`

---

## 🚁 Phase 2: Aerial Adaptation (With Replay)

### Stage 3: VisDrone + Ground Replay
*Goal: Learn aerial features (VisDrone) without losing the ground-level capabilities learned in Phase 1.*

If we switch to 100% VisDrone now, the model will forget what a "normal" car looks like. We use a **Replay Buffer** (30% ground data) to anchor the weights.

1.  **Generate Data (Aerial Heavy Mix)**:
    * Edit `config.toml`:
        ```toml
        [bdd]
        enabled = true
        frame_budget = 2000  # Ground Replay (Cars)

        [dancetrack]
        enabled = true
        frame_budget = 2000  # Ground Replay (People)

        [visdrone]
        enabled = true
        frame_budget = 10000 # Primary Task (Aerial)
        ```
    * Run: `python builder.py`

2.  **Config**: `configs/stage3_aerial_adapt.yaml`
    ```yaml
    SUPER_CONFIG_PATH: ./configs/r50_deformable_detr_motip_dancetrack.yaml
    
    # Load Stage 2 Result
    DETR_PRETRAIN: "./pretrains/stage2_v2.pth"
    
    NUM_CLASSES: 2
    
    # VisDrone objects are tiny; Increase resolution if memory permits
    AUG_MAX_SIZE: 1100
    EPOCHS: 10
    LR_DROP: 7
    
    # Validation: Now we validate on VisDrone (or BDD if you want to check regression)
    val_config:
      GT_FOLDER: "datasets/DanceTrack/val" # Or VisDrone val path
      SEQMAP_FILE: "datasets/DanceTrack/val_seqmap.txt"
    ```

3.  **Run**: `./train-start.sh configs/stage3_aerial_adapt.yaml`

---

## 📊 Verification & QA

We use a two-pronged validation approach: **Hard Numbers** (Dashboard) and **Visual Truth** (Video Viz).

### 1. Quantitative Heartbeat (Dashboard)
Use `plot_dashboard.py` to ensure HOTA/IDF1 are rising and Loss is falling.

**When to run:** During or after training.

```bash
# Generates dashboard.png in the output folder
python plot_dashboard.py outputs/stage1_universal_foundation_TIMESTAMP/train.log
```



**Success Criteria:**
* **HOTA/IDF1:** Should be > 54.0 (If it drops to 40s, we have regression).
* **Grad Norm:** Should be stable (not spiking to Infinity).

### 2. Qualitative Stress Test (Visual Viz)
Use `viz.py` to render "Ground Truth vs Prediction" comparisons. This reveals if the model is ignoring people (Blue Boxes) or hallucinating cars (Red Boxes).

**When to run:** After training completes.

```bash
# Example: Visualize Stage 1 results on DanceTrack Val
python viz.py \
    --config configs/stage1_universal_foundation.yaml \
    --checkpoint outputs/stage1_universal_foundation_TIMESTAMP/checkpoint_best_idf1.pth \
    --dataset_root datasets/DanceTrack/val \
    --output_dir outputs/stage1_viz \
    --score_thresh 0.4
```

**What to look for in the video:**
* **Green Boxes:** Stable tracking (Good).
* **Orange Boxes:** ID Switches (Needs more Stage 2 refinement).
* **Blue Dashed Boxes:** Missed detections (Regression - model forgot people).
