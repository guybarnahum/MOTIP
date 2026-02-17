# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: Submit or evaluate using Streaming LiteEval (Ultra-Low RAM).

import os
import json
import time
import torch
import random
import sys
import gc
import psutil

import numpy as np
from scipy.optimize import linear_sum_assignment
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader

from runtime_option import runtime_option
from utils.misc import yaml_to_dict
from configs.util import load_super_config, update_config
from log.logger import Logger
from data.joint_dataset import dataset_classes
from data.seq_dataset import SeqDataset
from models.runtime_tracker import RuntimeTracker
from log.log import Metrics
from models.motip import build as build_motip
from models.misc import load_checkpoint

try:
    from models.longterm_memory import LongTermMemory
except ImportError:
    LongTermMemory = None

# ------------------------------------------------------------------------
# LITE EVAL CORE: Fast NumPy IoU + Hungarian Matching
# ------------------------------------------------------------------------

def get_iou(bboxes1, bboxes2):
    if bboxes1.shape[0] == 0 or bboxes2.shape[0] == 0:
        return np.zeros((bboxes1.shape[0], bboxes2.shape[0]))
    b1, b2 = bboxes1.copy(), bboxes2.copy()
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
        res_data = np.loadtxt(res_path, delimiter=',')
        gt_data = np.loadtxt(gt_path, delimiter=',')
        gt_data = gt_data[gt_data[:, 6] > 0] 
        frames = np.unique(gt_data[:, 0]).astype(int)
        last_matches = {} 
        for f in frames:
            f_res = res_data[res_data[:, 0] == f]
            f_gt = gt_data[gt_data[:, 0] == f]
            t_gt_dets += len(f_gt)
            if len(f_gt) == 0:
                t_fp += len(f_res); continue
            if len(f_res) == 0:
                t_fn += len(f_gt); continue
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
    return {"MOTA": 1 - (t_fp + t_fn + t_idsw) / max(1, t_gt_dets), 
            "IDF1": (2 * t_idtp) / max(1, (2 * t_idtp + t_fp + t_fn))}

# ------------------------------------------------------------------------

def submit_and_evaluate(config: dict):
    accelerator = Accelerator()
    state = PartialState()
    mode = config.get("INFERENCE_MODE", "evaluate")
    outputs_dir = config.get("OUTPUTS_DIR", "./outputs")
    inf_model = config.get("INFERENCE_MODEL", "model")
    _name = os.path.split(inf_model)[-1].rsplit('.', 1)[0]
    out = os.path.join(outputs_dir, mode, config.get("INFERENCE_GROUP", "default"), 
                       config.get("INFERENCE_DATASET", "DanceTrack"), 
                       config.get("INFERENCE_SPLIT", "val"), _name)
    accelerator.wait_for_everyone()
    os.makedirs(out, exist_ok=True)
    logger = Logger(logdir=str(out), use_wandb=False, config=config)
    model, _ = build_motip(config=config)
    if os.path.exists(inf_model): load_checkpoint(model, path=inf_model)
    model = accelerator.prepare(model)

    metrics = submit_and_evaluate_one_model(
        is_evaluate=(mode == "evaluate"), accelerator=accelerator, state=state, logger=logger,
        model=model, data_root=config.get("DATA_ROOT", "./datasets"),
        dataset=config.get("INFERENCE_DATASET", "DanceTrack"), data_split=config.get("INFERENCE_SPLIT", "val"),
        outputs_dir=out, val_config=config.get("val_config", None),
        image_max_longer=config.get("INFERENCE_MAX_LONGER", 1536),
        det_thresh=config.get("DET_THRESH", 0.5), newborn_thresh=config.get("NEWBORN_THRESH", 0.5),
        id_thresh=config.get("ID_THRESH", 0.1)
    )
    if metrics:
        metrics.sync()
        logger.metrics(log="Eval Result: ", metrics=metrics, fmt="{global_average:.4f}")


def submit_and_evaluate_one_model(
        is_evaluate: bool,
        accelerator: Accelerator,
        state: PartialState,
        logger: Logger,
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
        **kwargs
):
    # 1. Resolve Frame Limit immediately
    if limit_frames is None and val_config is not None:
        limit_frames = val_config.get("LIMIT_VAL_FRAMES", None)

    sys.stderr.write(f"\n📊 Running LiteEval limit_frames:{limit_frames}\n")
    sys.stderr.flush()

    # 2. Setup Dataset
    inf_ds = dataset_classes[dataset](data_root=data_root, split=data_split, load_annotation=False)
    torch_dtype = torch.float32 if dtype == "FP32" else torch.float16
    _seq_names = sorted(list(inf_ds.sequence_infos.keys()))
    
    # Filter sequences for DDP
    for i, name in enumerate(_seq_names):
        if i % state.num_processes != state.process_index:
            inf_ds.sequence_infos.pop(name)
            inf_ds.image_paths.pop(name)

    eval_rng = random.Random(42) 
    num_seqs = len(inf_ds.sequence_infos)

    # 📢 STARTUP PRINT (Unbuffered)
    sys.stderr.write(f"\n🚀 STARTING EVALUATION: {num_seqs} sequences detected.\n")
    sys.stderr.flush()

    # 3. Main Loop
    for s_idx, sequence_name in enumerate(inf_ds.sequence_infos.keys()):
        # 📢 SEQUENCE HEADER
        sys.stderr.write(f"\n▶️ [{s_idx+1}/{num_seqs}] Seq: {sequence_name}\n")
        sys.stderr.flush()

        seq_ds = SeqDataset(
            inf_ds.sequence_infos[sequence_name], inf_ds.image_paths[sequence_name],
            image_max_shorter, image_max_longer, size_divisibility, torch_dtype
        )
        
        # ⚠️ FORCE 0 WORKERS: Prevents RAM fragmentation in long loops
        loader = DataLoader(seq_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x[0])
        
        tracker = RuntimeTracker(
            model, seq_ds.seq_hw(), use_sigmoid, assignment_protocol, 
            miss_tolerance, det_thresh, newborn_thresh, id_thresh, 
            area_thresh, inference_only_detr, torch_dtype
        )
        memory = LongTermMemory(patience=900) if LongTermMemory else None

        # Determine Window
        total_frames = len(loader)
        start_f = eval_rng.randint(0, max(0, total_frames - (limit_frames or 1))) if limit_frames else 0
        actual_limit = limit_frames if limit_frames else total_frames
        end_f = start_f + actual_limit

        tracker_path = os.path.join(outputs_dir, "tracker")
        os.makedirs(tracker_path, exist_ok=True)
        txt_path = os.path.join(tracker_path, f"{sequence_name}.txt")

        with open(txt_path, "w") as f:
            start_time = time.time()
            for t, (image, _) in enumerate(loader):
                if t < start_f: continue
                if t >= end_f: break 
                
                image.tensors, image.mask = image.tensors.cuda(), image.mask.cuda()
                tracker.update(image=image)
                res = tracker.get_track_results()
                
                if memory and "embeddings" in res and len(res["id"]) > 0:
                    id_map = memory.update(t, res["id"].tolist(), res["embeddings"])
                    res["id"] = torch.tensor([id_map.get(rid, rid) for rid in res["id"].tolist()], dtype=torch.int64)

                for obj_id, bbox in zip(res["id"], res["bbox"]):
                    f.write(f"{t+1},{obj_id.item()},{bbox[0].item()},{bbox[1].item()},{bbox[2].item()},{bbox[3].item()},1,-1,-1,-1\n")
                
                # 📢 HEARTBEAT PRINT (Every 20 frames)
                if (t + 1 - start_f) % 20 == 0:
                    curr_ram = psutil.virtual_memory().percent
                    sys.stderr.write(f"   ∟ Frame {t+1-start_f}/{actual_limit} | RAM: {curr_ram}% | Swap: {psutil.swap_memory().used/(1024**3):.1f}GB\n")
                    sys.stderr.flush()

                # CRITICAL: Clean up frame objects immediately
                del res
                if t % 100 == 0:
                    torch.cuda.empty_cache()

            fps = (t + 1 - start_f) / max(1e-5, (time.time() - start_time))
            sys.stderr.write(f"✅ Sequence Finished. Avg FPS: {fps:.1f}\n")
            sys.stderr.flush()

        # 🧹 SEQUENCE EXIT: Absolute Purge
        del loader, seq_ds, tracker, memory
        gc.collect() 

    accelerator.wait_for_everyone()
    if not is_evaluate: return None

    # 4. Final aggregation
    metrics = Metrics()
    if accelerator.is_main_process:
        sys.stderr.write("\n📊 Running LiteEval Aggregation...\n")
        sys.stderr.flush()
        gt_root = val_config.get("GT_FOLDER") if val_config else os.path.join(data_root, dataset, data_split)
        lite_res = lite_mot_eval(os.path.join(outputs_dir, "tracker"), gt_root)
        metrics["MOTA"].update(lite_res["MOTA"])
        metrics["IDF1"].update(lite_res["IDF1"])
        metrics["HOTA"].update(0.0) 
        
        logger.success(f"LiteEval Results -> MOTA: {lite_res['MOTA']:.4f} | IDF1: {lite_res['IDF1']:.4f}", only_main=True)
    return metrics


if __name__ == '__main__':
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)
    cfg = load_super_config(cfg, opt.super_config_path or cfg.get("SUPER_CONFIG_PATH"))
    submit_and_evaluate(update_config(cfg, opt))