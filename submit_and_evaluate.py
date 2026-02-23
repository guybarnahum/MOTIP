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
from tqdm import tqdm  

from data.joint_dataset import dataset_classes
from data.seq_dataset import SeqDataset
from models.runtime_tracker import RuntimeTracker
from log.log import Metrics
from collections import defaultdict

# --- DIAGNOSTIC HELPERS ---
diag_logs = False

def diag_log(msg: str):
    """Bypasses standard logging to flush messages immediately to terminal."""
    if diag_logs:
        sys.stderr.write(f"🚩 [DIAG] {msg}\n")
        sys.stderr.flush()
    else:
        print(msg, flush=True)


def get_iou(bboxes1, bboxes2):
    """ Helper for IoU calculation [x, y, w, h] """
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]))
    
    # Convert to x1, y1, x2, y2
    b1 = np.copy(bboxes1); b1[:, 2:] += b1[:, :2]
    b2 = np.copy(bboxes2); b2[:, 2:] += b2[:, :2]
    
    iou = np.zeros((len(b1), len(b2)))
    for i, bc1 in enumerate(b1):
        for j, bc2 in enumerate(b2):
            iw = min(bc1[2], bc2[2]) - max(bc1[0], bc2[0])
            ih = min(bc1[3], bc2[3]) - max(bc1[1], bc2[1])
            if iw > 0 and ih > 0:
                area1 = (bc1[2]-bc1[0])*(bc1[3]-bc1[1])
                area2 = (bc2[2]-bc2[0])*(bc2[3]-bc2[1])
                iou[i, j] = (iw*ih) / (area1 + area2 - iw*ih)
    return iou

# ------------------------------------------------------------------------
# LITE EVAL ENGINE: Fast Metrics Calculation
# ------------------------------------------------------------------------

def lite_mot_eval(results_dir, gt_root, classes=[1, 2], iou_thresh=0.5):
    stats = {
        'overall': {'tp': 0, 'fp': 0, 'fn': 0, 'idsw': 0, 'gt_count': 0},
        'per_class': {c: {'tp': 0, 'fp': 0, 'fn': 0, 'gt_count': 0} for c in classes}
    }
    
    global_matches = defaultdict(lambda: defaultdict(int))
    gt_id_existence = defaultdict(int)
    pred_id_existence = defaultdict(int)
    
    seq_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.txt')])
    
    print(f"🔍 [EVAL] Starting Geometric Matching for {len(seq_files)} sequences...")
    
    for seq_file in seq_files:
        seq_name = seq_file.replace('.txt', '')
        res_path = os.path.join(results_dir, seq_file)
        gt_path = os.path.join(gt_root, seq_name, 'gt', 'gt.txt')
        if not os.path.exists(gt_path): continue
        
        # Patch: Handle empty files or loading errors gracefully
        try:
            res_data = np.atleast_2d(np.loadtxt(res_path, delimiter=','))
        except Exception:
            res_data = np.array([]).reshape(0, 10) # Create empty 2D array if file is unreadable

        gt_data = np.atleast_2d(np.loadtxt(gt_path, delimiter=','))
        if gt_data.size == 0: continue

        gt_data = gt_data[gt_data[:, 6] > 0]
        frames = np.unique(gt_data[:, 0]).astype(int)
        last_matches = {} 

        # Add a nested progress bar for frames in the current sequence
        for f in tqdm(frames, desc=f"   ∟ Processing {seq_name[:15]}...", leave=False):
            f_gt = gt_data[gt_data[:, 0] == f]
            
            # Patch: Check size of res_data before indexing axis 0
            if res_data.size > 0 and res_data.shape[1] > 0:
                f_res = res_data[res_data[:, 0] == f]
            else:
                f_res = np.array([])
            
            stats['overall']['gt_count'] += len(f_gt)
            for g_row in f_gt:
                cls_id = int(g_row[7])
                if cls_id in stats['per_class']:
                    stats['per_class'][cls_id]['gt_count'] += 1
                    gt_id_existence[f"{seq_name}_{int(g_row[1])}"] += 1

            if len(f_res) > 0:
                for p_id in f_res[:, 1]:
                    pred_id_existence[f"{seq_name}_{int(p_id)}"] += 1

            if len(f_res) == 0:
                stats['overall']['fn'] += len(f_gt)
                for g_row in f_gt:
                    cls_id = int(g_row[7])
                    if cls_id in stats['per_class']: stats['per_class'][cls_id]['fn'] += 1
                continue

            ious = get_iou(f_res[:, 2:6], f_gt[:, 2:6])
            res_idx, gt_idx = linear_sum_assignment(-ious)
            
            matched_res = set()
            matched_gt = set()
            
            for r, g in zip(res_idx, gt_idx):
                if ious[r, g] >= iou_thresh:
                    gt_id, res_id = int(f_gt[g, 1]), int(f_res[r, 1])
                    cls_id = int(f_gt[g, 7])
                    global_matches[f"{seq_name}_{gt_id}"][f"{seq_name}_{res_id}"] += 1
                    
                    if gt_id in last_matches and last_matches[gt_id] != res_id:
                        stats['overall']['idsw'] += 1
                    last_matches[gt_id] = res_id
                    
                    stats['overall']['tp'] += 1
                    if cls_id in stats['per_class']: stats['per_class'][cls_id]['tp'] += 1
                    matched_res.add(r)
                    matched_gt.add(g)

            stats['overall']['fn'] += (len(f_gt) - len(matched_gt))
            stats['overall']['fp'] += (len(f_res) - len(matched_res))
            
            # Per-class FP/FN attribution
            for idx, g_row in enumerate(f_gt):
                if idx not in matched_gt:
                    c_id = int(g_row[7])
                    if c_id in stats['per_class']: stats['per_class'][c_id]['fn'] += 1
            for idx, r_row in enumerate(f_res):
                if idx not in matched_res:
                    p_cls = int(r_row[7]) if r_row.shape[0] > 7 and r_row[7] != -1 else None
                    if p_cls in stats['per_class']: stats['per_class'][p_cls]['fp'] += 1

    # --- THE "HANG" ZONE: Global IDTP ---
    print(f"📊 [EVAL] Initializing Global Association Matrix ({len(gt_id_existence)}x{len(pred_id_existence)})...")
    
    gt_uids = list(gt_id_existence.keys())
    pr_uids = list(pred_id_existence.keys())
    idtp_global = 0
    if gt_uids and pr_uids:
        # Safety check for stability test scenarios
        if len(gt_uids) * len(pr_uids) > 25_000_000: # Threshold for ~5000x5000 matrix
            print("⚠️ [WARNING] Matrix size extreme. Solving will be slow...")
            
        c_mat = np.zeros((len(gt_uids), len(pr_uids)), dtype=np.int32)
        u_gt_map = {uid: i for i, uid in enumerate(gt_uids)}
        u_pr_map = {uid: i for i, uid in enumerate(pr_uids)}
        
        for g_uid, p_dict in global_matches.items():
            for p_uid, count in p_dict.items():
                c_mat[u_gt_map[g_uid], u_pr_map[p_uid]] = count
        
        print("🧠 [EVAL] Solving Global Identity Matching (Hungarian Algorithm)...")
        r_i, c_i = linear_sum_assignment(-c_mat)
        idtp_global = c_mat[r_i, c_i].sum()
        print("✅ [EVAL] Identity Matching Complete.")

    # Derived Metrics
    detA = stats['overall']['tp'] / max(1, stats['overall']['tp'] + stats['overall']['fp'] + stats['overall']['fn'])
    idf1 = (2 * idtp_global) / max(1, sum(gt_id_existence.values()) + sum(pred_id_existence.values()))
    
    # AssA Calculation
    assa_scores = []
    for g_uid in gt_uids:
        matches = global_matches[g_uid]
        if matches:
            best_p_uid = max(matches, key=matches.get)
            match_count = matches[best_p_uid]
            assa_scores.append(match_count / (gt_id_existence[g_uid] + pred_id_existence[best_p_uid] - match_count))
        else: assa_scores.append(0)
    assA = np.mean(assa_scores) if assa_scores else 0

    return {
        "DetA": detA * 100,
        "AssA": assA * 100,
        "IDF1": idf1 * 100,
        "MOTA": (1 - (stats['overall']['fp'] + stats['overall']['fn'] + stats['overall']['idsw']) / max(1, stats['overall']['gt_count'])) * 100,
        "Classes": {c: {
            "Prec": (v['tp'] / max(1, v['tp'] + v['fp'])) * 100,
            "Rec": (v['tp'] / max(1, v['gt_count'])) * 100
        } for c, v in stats['per_class'].items()}
    }

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
        del loader, seq_ds, tracker
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
           
        # Populate standard Metrics object
        metrics["MOTA"].update(lite_res["MOTA"] / 100.0) # Convert % back to decimal for logging
        metrics["IDF1"].update(lite_res["IDF1"] / 100.0)
        metrics["DetA"].update(lite_res["DetA"] / 100.0)
        metrics["AssA"].update(lite_res["AssA"] / 100.0)
        metrics["HOTA"].update(0.0) # Placeholder

        # --- PRETTY PRINTING THE RESULTS ---
        logger.success(f"Final Eval -> MOTA: {lite_res['MOTA']:.2f}% | IDF1: {lite_res['IDF1']:.2f}%")
        logger.info(f"Association Diagnostics -> DetA: {lite_res['DetA']:.2f}% | AssA: {lite_res['AssA']:.2f}%")
        
        # Per-Class Breakdown Table
        print("\n" + "="*50)
        print(f"{'Class ID':<10} | {'Precision':<15} | {'Recall':<15}")
        print("-" * 50)
        for cls_id, cls_stats in lite_res["Classes"].items():
            name = "Person" if cls_id == 1 else "Vehicle" if cls_id == 2 else f"ID {cls_id}"
            print(f"{name:<10} | {cls_stats['Prec']:>13.2f}% | {cls_stats['Rec']:>13.2f}%")
        print("="*50 + "\n")
        
    return metrics
