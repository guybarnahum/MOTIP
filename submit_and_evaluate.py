# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import time
import torch
import random
import sys
import gc
import psutil
import numpy as np

from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment

from data.joint_dataset import dataset_classes
from data.seq_dataset import SeqDataset
from models.runtime_tracker import RuntimeTracker
from log.log import Metrics

try:
    from models.longterm_memory import LongTermMemory
except ImportError:
    LongTermMemory = None

# --- DIAGNOSTIC HELPERS ---
diag_logs = False

def diag_log(msg: str):
    """Bypasses standard logging to flush messages immediately to terminal."""
    if diag_logs:
        sys.stderr.write(f"🚩 [DIAG] {msg}\n")
        sys.stderr.flush()
    else:
        print(msg)

# ------------------------------------------------------------------------
# LITE EVAL ENGINE: Fast Metrics Calculation
# ------------------------------------------------------------------------

def get_iou(bboxes1, bboxes2):
    """Calculates Intersection over Union for two sets of boxes."""
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]))
    b1, b2 = bboxes1.copy(), bboxes2.copy()
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    b1[:, 2:] += b1[:, :2]
    b2[:, 2:] += b2[:, :2]
    lt = np.maximum(b1[:, None, :2], b2[:, :2])
    rb = np.minimum(b1[:, None, 2:], b2[:, 2:])
    wh = np.maximum(rb - lt, 0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-7)


def lite_mot_eval(results_dir, gt_root, iou_thresh=0.5):
    t_gt_dets, t_fp, t_fn, t_idsw, t_idtp = 0, 0, 0, 0, 0
    seq_files = [f for f in os.listdir(results_dir) if f.endswith('.txt')]
    
    for seq_file in seq_files:
        seq_name = seq_file.replace('.txt', '')
        res_path = os.path.join(results_dir, seq_file)
        gt_path = os.path.join(gt_root, seq_name, 'gt', 'gt.txt')
        if not os.path.exists(gt_path): continue
        
        # 🚨 THE FIX: Force 2D even for single rows or empty files
        res_data = np.atleast_2d(np.loadtxt(res_path, delimiter=','))
        gt_data = np.atleast_2d(np.loadtxt(gt_path, delimiter=','))
        
        if res_data.size == 0 or gt_data.size == 0:
            # Handle empty sequences gracefully
            if gt_data.size > 0: t_gt_dets += len(gt_data); t_fn += len(gt_data)
            if res_data.size > 0: t_fp += len(res_data)
            continue

        # Filter GT by visibility (column 6)
        gt_data = gt_data[gt_data[:, 6] > 0] 
        frames = np.unique(gt_data[:, 0]).astype(int)
        last_matches = {} 
        
        for f in frames:
            f_res = res_data[res_data[:, 0] == f]
            f_gt = gt_data[gt_data[:, 0] == f]
            t_gt_dets += len(f_gt)
            
            if len(f_gt) == 0: t_fp += len(f_res); continue
            if len(f_res) == 0: t_fn += len(f_gt); continue
                
            ious = get_iou(f_res[:, 2:6], f_gt[:, 2:6])
            res_idx, gt_idx = linear_sum_assignment(-ious)
            matched_count = 0
            
            for r, g in zip(res_idx, gt_idx):
                if ious[r, g] >= iou_thresh:
                    gt_id, res_id = int(f_gt[g, 1]), int(f_res[r, 1])
                    if gt_id in last_matches and last_matches[gt_id] != res_id:
                        t_idsw += 1
                    last_matches[gt_id] = res_id
                    t_idtp += 1
                    matched_count += 1
            t_fn += (len(f_gt) - matched_count)
            t_fp += (len(f_res) - matched_count)
            
    mota = 1 - (t_fp + t_fn + t_idsw) / max(1, t_gt_dets)
    idf1 = (2 * t_idtp) / max(1, (2 * t_idtp + t_fp + t_fn))
    return {"MOTA": mota, "IDF1": idf1}

# ------------------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# ------------------------------------------------------------------------

def submit_and_evaluate_one_model(
        is_evaluate: bool,
        accelerator,
        state,
        logger,
        model,
        data_root: str,
        dataset: str,
        data_split: str,
        outputs_dir: str,
        val_config: dict = None,
        image_max_shorter: int = 800,
        image_max_longer: int = 800,
        size_divisibility: int = 0,
        use_sigmoid: bool = False,
        assignment_protocol: str = "hungarian",
        miss_tolerance: int = 30,
        det_thresh: float = 0.4,
        newborn_thresh: float = 0.4,
        id_thresh: float = 0.1,
        area_thresh: int = 0,
        inference_only_detr: bool = False,
        dtype: str = "FP32",
        limit_frames: int = None,
        limit_seqs: int = None,  # Set to None to evaluate all sequences
        **kwargs
):
    # 1. Resolve Frame and Seq Limit immediately
    if limit_frames is None and val_config is not None:
        limit_frames = val_config.get("LIMIT_VAL_FRAMES", None)

    if limit_seqs is None and val_config is not None:
        limit_seqs = val_config.get("LIMIT_VAL_SEQ", None)

    # 2. FIX GEOMETRY: Ensure max_longer > max_shorter for torchvision v2.Resize
    if image_max_longer <= image_max_shorter:
        image_max_longer = 1333 
        
    diag_log(f"📊 [EVAL] Frame Limit: {limit_frames} | Resize: {image_max_shorter}x{image_max_longer}")

    # 3. Setup Dataset
    inf_ds = dataset_classes[dataset](data_root=data_root, split=data_split, load_annotation=False)
    torch_dtype = torch.float32 if dtype == "FP32" else torch.float16
    all_seq_names = sorted(list(inf_ds.sequence_infos.keys()))
    
    # 🎲 RANDOMIZE: Shuffle with a fixed seed so all DDP processes 
    # agree on the same random order before splitting.
    random.Random(42).shuffle(all_seq_names)

    # 🚨 SEQUENCE FILTERING: Trimming happens AFTER shuffle to get a diverse mix
    if limit_seqs is not None and len(all_seq_names) > limit_seqs:
        all_seq_names = all_seq_names[:limit_seqs]
        diag_log(f"✂️ [EVAL] Trimming evaluation to first {limit_seqs} sequences for speed.")

    # Properly split across DDP processes (if any)
    my_seq_names = [name for i, name in enumerate(all_seq_names) if i % state.num_processes == state.process_index]

    eval_rng = random.Random(42) 
    num_seqs = len(my_seq_names)

    # 4. Main Streaming Loop
    for s_idx, sequence_name in enumerate(my_seq_names):
        diag_log(f"▶️ [{s_idx+1}/{num_seqs}] Seq: {sequence_name} (RAM: {psutil.virtual_memory().percent}%)")

        seq_ds = SeqDataset(
            inf_ds.sequence_infos[sequence_name], inf_ds.image_paths[sequence_name],
            image_max_shorter, image_max_longer, size_divisibility, torch_dtype
        )
        
        # num_workers=0 is mandatory to prevent process-cloning RAM spikes
        loader = DataLoader(seq_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
        
        tracker = RuntimeTracker(
            model, seq_ds.seq_hw(), use_sigmoid, assignment_protocol, 
            miss_tolerance, det_thresh, newborn_thresh, id_thresh, 
            area_thresh, inference_only_detr, torch_dtype
        )
        memory = LongTermMemory(patience=900) if LongTermMemory else None

        # Random Window Selection
        total_frames = len(loader)
        start_f = eval_rng.randint(0, max(0, total_frames - (limit_frames or 1))) if limit_frames else 0
        actual_limit = min(limit_frames, total_frames) if limit_frames else total_frames
        end_f = start_f + actual_limit

        tracker_path = os.path.join(outputs_dir, "tracker")
        os.makedirs(tracker_path, exist_ok=True)
        txt_path = os.path.join(tracker_path, f"{sequence_name}.txt")

        with open(txt_path, "w") as f:
            start_time = time.time()
            # We track the number of frames actually processed for FPS calculation
            frames_processed = 0
            
            for t, (image, _) in enumerate(loader):
                if t < start_f: continue
                if t >= end_f: break 
                
                image.tensors, image.mask = image.tensors.cuda(), image.mask.cuda()
                tracker.update(image=image)
                res = tracker.get_track_results()
                
                if memory and "embeddings" in res and len(res["id"]) > 0:
                    id_map = memory.update(t, res["id"].tolist(), res["embeddings"])
                    res["id"] = torch.tensor([id_map.get(rid, rid) for rid in res["id"].tolist()], dtype=torch.int64)

                # Write results in MOT format
                for obj_id, bbox in zip(res["id"], res["bbox"]):
                    f.write(f"{t+1},{obj_id.item()},{bbox[0].item():.2f},{bbox[1].item():.2f},{bbox[2].item():.2f},{bbox[3].item():.2f},1,-1,-1,-1\n")
                
                frames_processed += 1
                
                # Progress heartbeat
                if frames_processed % 25 == 0:
                    curr_ram = psutil.virtual_memory().percent
                    diag_log(f"   ∟ Progress: {frames_processed}/{actual_limit} | RAM: {curr_ram}% | Swap: {psutil.swap_memory().used/(1024**3):.1f}GB")

                del res
                if t % 50 == 0: 
                    torch.cuda.empty_cache()

            fps = frames_processed / max(1e-5, (time.time() - start_time))
            diag_log(f"✅ Finished {sequence_name} | {fps:.1f} FPS")

        # NUCLEAR PURGE per sequence to keep baseline RAM at ~10%
        del loader, seq_ds, tracker, memory
        gc.collect() 

    accelerator.wait_for_everyone()
    if not is_evaluate: 
        return None

    # 5. Global Aggregation
    metrics = Metrics()
    if accelerator.is_main_process:
        diag_log("📊 [EVAL] Calculating Global MOTA/IDF1...")
        gt_root = val_config.get("GT_FOLDER") if val_config else os.path.join(data_root, dataset, data_split)
        
        # Ensure lite_mot_eval uses the robust version with np.atleast_2d
        lite_res = lite_mot_eval(os.path.join(outputs_dir, "tracker"), gt_root)
        
        metrics["MOTA"].update(lite_res["MOTA"])
        metrics["IDF1"].update(lite_res["IDF1"])
        metrics["HOTA"].update(0.0) 
        
        logger.success(f"Final Eval -> MOTA: {lite_res['MOTA']:.4f} | IDF1: {lite_res['IDF1']:.4f}")

    return metrics
