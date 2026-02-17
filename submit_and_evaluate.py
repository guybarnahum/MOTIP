# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: Submit or evaluate using Streaming LiteEval (Ultra-Low RAM).

import os
import json
import time
import torch
import random
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
    # 1. Dataset & Type Setup
    inference_dataset = dataset_classes[dataset](data_root=data_root, split=data_split, load_annotation=False)
    torch_dtype = torch.float32 if dtype == "FP32" else torch.float16
    
    # 2. Multi-GPU Sequence Splitting (DDP Friendly)
    _seq_names = sorted(list(inference_dataset.sequence_infos.keys()))
    for i, name in enumerate(_seq_names):
        if i % state.num_processes != state.process_index:
            inference_dataset.sequence_infos.pop(name)
            inference_dataset.image_paths.pop(name)

    # 3. Reproducibility Seed for Windowing
    # Using a fixed seed ensures we pick the same "random" slice every epoch
    eval_rng = random.Random(42) 

    num_seqs = len(inference_dataset.sequence_infos)
    
    # 4. Main Streaming Loop
    for s_idx, sequence_name in enumerate(inference_dataset.sequence_infos.keys()):
        seq_ds = SeqDataset(
            seq_info=inference_dataset.sequence_infos[sequence_name],
            image_paths=inference_dataset.image_paths[sequence_name],
            max_shorter=image_max_shorter, max_longer=image_max_longer,
            size_divisibility=size_divisibility, dtype=torch_dtype
        )
        loader = DataLoader(dataset=seq_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=lambda x: x[0])
        tracker = RuntimeTracker(
            model=model, sequence_hw=seq_ds.seq_hw(), use_sigmoid=use_sigmoid,
            assignment_protocol=assignment_protocol, miss_tolerance=miss_tolerance,
            det_thresh=det_thresh, newborn_thresh=newborn_thresh, id_thresh=id_thresh,
            area_thresh=area_thresh, only_detr=inference_only_detr, dtype=torch_dtype
        )

        memory = LongTermMemory(patience=900) if LongTermMemory else None
        
        # Determine the Window [start_frame, end_frame)
        total_frames = len(loader)
        start_frame = 0
        if limit_frames and total_frames > limit_frames:
            # We use eval_rng.randint to stay reproducible across epochs
            start_frame = eval_rng.randint(0, total_frames - limit_frames)
        
        actual_limit = limit_frames if limit_frames else total_frames
        end_frame = start_frame + actual_limit

        tracker_path = os.path.join(outputs_dir, "tracker")
        os.makedirs(tracker_path, exist_ok=True)
        txt_path = os.path.join(tracker_path, f"{sequence_name}.txt")

        # Stream directly to disk to save RAM
        with open(txt_path, "w") as f:
            start_time = time.time()
            logger.info(f"▶️ [{s_idx+1}/{num_seqs}] Seq: {sequence_name} | Window: {start_frame}→{end_frame}", only_main=False)
            
            for t, (image, _) in enumerate(loader):
                # Skip to start of window
                if t < start_frame:
                    continue
                # Break at end of window
                if t >= end_frame:
                    break 
                
                image.tensors, image.mask = image.tensors.cuda(), image.mask.cuda()
                tracker.update(image=image)
                res = tracker.get_track_results()
                
                # Apply LongTermMemory re-ID mapping if active
                if memory and "embeddings" in res and len(res["id"]) > 0:
                    id_map = memory.update(t, res["id"].tolist(), res["embeddings"])
                    new_ids = [id_map.get(rid, rid) for rid in res["id"].tolist()]
                    res["id"] = torch.tensor(new_ids, dtype=torch.int64)

                # Format as MOTChallenge: <frame>, <id>, <x>, <y>, <w>, <h>, <conf>, -1, -1, -1
                for obj_id, bbox in zip(res["id"], res["bbox"]):
                    f.write(f"{t+1},{obj_id.item()},{bbox[0].item()},{bbox[1].item()},{bbox[2].item()},{bbox[3].item()},1,-1,-1,-1\n")
                
                # Progress logging every 50 frames
                processed_count = t + 1 - start_frame
                if processed_count % 50 == 0:
                    logger.info(f"   ∟ Progress: {processed_count}/{actual_limit} frames...", only_main=False)
                
                # Memory Cleanup
                del res
                if t % 100 == 0:
                    torch.cuda.empty_cache()
            
            fps = processed_count / max(1e-5, (time.time() - start_time))
            logger.success(f"✅ Finished {sequence_name} | FPS: {fps:.1f}", only_main=False)

    accelerator.wait_for_everyone()
    
    if not is_evaluate:
        return None

    # Final LiteMOT Metric Aggregation
    metrics = Metrics()
    if accelerator.is_main_process:
        logger.info("📊 LiteEval: Calculating epoch vitals...", only_main=True)
        tracker_dir = os.path.join(outputs_dir, "tracker")
        gt_root = val_config.get("GT_FOLDER") if val_config else os.path.join(data_root, dataset, data_split)
        
        lite_res = lite_mot_eval(tracker_dir, gt_root)
        
        metrics["MOTA"].update(lite_res["MOTA"])
        metrics["IDF1"].update(lite_res["IDF1"])
        metrics["HOTA"].update(0.0) # Placeholder for standalone TrackEval audit
        
        logger.success(f"MOTA: {lite_res['MOTA']:.4f} | IDF1: {lite_res['IDF1']:.4f}", only_main=True)
        
    return metrics


if __name__ == '__main__':
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)
    cfg = load_super_config(cfg, opt.super_config_path or cfg.get("SUPER_CONFIG_PATH"))
    submit_and_evaluate(update_config(cfg, opt))