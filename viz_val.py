import argparse
import cv2
import torch
import os
import sys
import yaml
import numpy as np
import glob
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Import your existing modules
from memory_manager import LongTermMemory
# Added convert_to_h264 to imports
from utils_inference import recover_embeddings, convert_to_h264
from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker

# -------------------------------------------------------------------------
# Monkey Patch
# -------------------------------------------------------------------------
original_get_activate_detections = RuntimeTracker._get_activate_detections
def patched_get_activate_detections(self, detr_out):
    self.output = detr_out
    return original_get_activate_detections(self, detr_out)
RuntimeTracker._get_activate_detections = patched_get_activate_detections

# -------------------------------------------------------------------------
# Helper: Ground Truth Parsing (Correct as verified)
# -------------------------------------------------------------------------
def load_mot_gt(gt_path):
    gt_dict = {}
    if not os.path.exists(gt_path):
        return gt_dict

    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            frame = int(parts[0])
            obj_id = int(parts[1])
            # GT is xywh (TopLeft)
            x, y, w, h = map(float, parts[2:6])
            # Convert to xyxy (Corner)
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            if frame not in gt_dict:
                gt_dict[frame] = []
            gt_dict[frame].append({'bbox': [x1, y1, x2, y2], 'id': obj_id})
    return gt_dict

# -------------------------------------------------------------------------
# Helper: Prediction Format Conversion (THE FIX 🔧)
# -------------------------------------------------------------------------
def convert_tlwh_to_xyxy(boxes):
    """
    Converts TopLeft-Size [x1, y1, w, h] to Corner-Corner [x1, y1, x2, y2]
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    w  = boxes[:, 2]
    h  = boxes[:, 3]
    
    x2 = x1 + w
    y2 = y1 + h
    
    return np.stack([x1, y1, x2, y2], axis=1)

def compute_iou_matrix(preds, gts):
    if len(preds) == 0 or len(gts) == 0:
        return np.zeros((len(preds), len(gts)))

    iou_matrix = np.zeros((len(preds), len(gts)))
    for i, p_box in enumerate(preds):
        p_x1, p_y1, p_x2, p_y2 = p_box
        p_area = (p_x2 - p_x1) * (p_y2 - p_y1)
        for j, g_box in enumerate(gts):
            g_x1, g_y1, g_x2, g_y2 = g_box
            g_area = (g_x2 - g_x1) * (g_y2 - g_y1)
            ix1 = max(p_x1, g_x1); iy1 = max(p_y1, g_y1)
            ix2 = min(p_x2, g_x2); iy2 = min(p_y2, g_y2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = p_area + g_area - inter
                iou_matrix[i, j] = inter / union
    return iou_matrix

# -------------------------------------------------------------------------
# Main Processor
# -------------------------------------------------------------------------
def process_sequence(seq_path, gt_path, output_path, model, device, args):
    cap = cv2.VideoCapture(os.path.join(seq_path, 'img1/%08d.jpg'))
    if not cap.isOpened(): return

    W, H = int(cap.get(3)), int(cap.get(4))
    fps = 10 
    
    tracker = RuntimeTracker(
        model=model, sequence_hw=(H, W),
        miss_tolerance=30, det_thresh=args.score_thresh,
        dtype=torch.float32
    )
    tracker.bbox_unnorm = torch.tensor([W, H, W, H], device=device).float()
    
    gt_data = load_mot_gt(gt_path)
    
    # Use a temp file for OpenCV writing (mp4v), convert to H.264 later
    temp_out = output_path.replace(".mp4", "_temp.mp4")
    out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    gt_id_to_pred_id = {} 
    frame_idx = 1 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. Inference
        t_img = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255.0
        h, w = t_img.shape[1:]
        scale = 800 / min(h, w)
        if max(h, w) * scale > 1333: scale = 1333 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = torch.nn.functional.interpolate(t_img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
        img_norm = (img_resized - mean) / std
        
        tracker.update(img_norm)
        res = tracker.get_track_results()
        
        raw_pred_boxes = res['bbox'].cpu().numpy() # This is [cx, cy, w, h]
        pred_ids = res['id'].tolist()
        
        # --- FIX: Convert Predictions to XYXY ---
        pred_boxes = convert_tlwh_to_xyxy(raw_pred_boxes)
        
        current_gt = gt_data.get(frame_idx, [])
        gt_boxes = [g['bbox'] for g in current_gt]
        gt_ids_frame = [g['id'] for g in current_gt]
        
        # 3. Match Predictions to GT
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)
        
        matched_gt_indices = set()
        matched_pred_indices = set()
        matches = [] 
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= 0.5:
                matches.append((r, c))
                matched_gt_indices.add(c)
                matched_pred_indices.add(r)

        # 4. Draw & Analyze
        
        # A. Misses (Blue Dashed)
        for i, gbox in enumerate(gt_boxes):
            if i not in matched_gt_indices:
                x1, y1, x2, y2 = map(int, gbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                for d in range(x1, x2, 10): cv2.line(frame, (d, y1), (d+5, y1), (255,0,0), 2)
                for d in range(y1, y2, 10): cv2.line(frame, (x1, d), (x1, d+5), (255,0,0), 2)
                cv2.putText(frame, f"MISS G:{gt_ids_frame[i]}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # B. False Positives (Red)
        for i, pbox in enumerate(pred_boxes):
            if i not in matched_pred_indices:
                x1, y1, x2, y2 = map(int, pbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"FP P:{pred_ids[i]}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # C. True Positives + ID CHECK (Green or Alert)
        for r, c in matches:
            pbox = pred_boxes[r]
            pid = pred_ids[r]
            gid = gt_ids_frame[c]
            
            x1, y1, x2, y2 = map(int, pbox)
            
            # --- ID LOGIC ---
            previous_pid = gt_id_to_pred_id.get(gid)
            
            # Case 1: New Car (First time seeing this GT ID)
            if previous_pid is None:
                color = (0, 255, 0) # Green
                text = f"P:{pid} G:{gid}"
                gt_id_to_pred_id[gid] = pid
            # Case 2: ID Switch (Same Car, Different Tracker ID)
            elif previous_pid != pid:
                color = (0, 165, 255) # Orange (BGR)
                text = f"SWITCH! {previous_pid}->{pid}"
                # Update map to accept the new ID as the current valid one
                gt_id_to_pred_id[gid] = pid
                # Draw thick border for emphasis
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            # Case 3: Stable (Same Car, Same Tracker ID)
            else:
                color = (0, 255, 0) # Green
                text = f"P:{pid} G:{gid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Legend
        cv2.rectangle(frame, (5, 5), (250, 105), (0,0,0), -1)
        cv2.putText(frame, "Green: Stable Match", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Orange: ID SWITCH", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, "Red Box: False Pos", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Blue: Missed GT", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out.write(frame)
        frame_idx += 1

    out.release()
    cap.release()
    
    # Convert to H.264
    if os.path.exists(temp_out):
        convert_to_h264(temp_out, output_path)
        if os.path.exists(output_path):
            os.remove(temp_out)
    else:
        print(f"✅ Saved: {output_path}")

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/pretrain_r50_deformable_detr_bdd_mini.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default='./datasets/DanceTrack/val')
    parser.add_argument('--output_dir', type=str, default='./outputs/stage1_id_viz')
    parser.add_argument('--score_thresh', type=float, default=0.4)
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    if "SUPER_CONFIG_PATH" in cfg:
        with open(cfg["SUPER_CONFIG_PATH"], 'r') as f_base:
            base_cfg = yaml.safe_load(f_base)
        base_cfg.update(cfg)
        cfg = base_cfg

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(cfg)[0].to(device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    seqs = sorted(glob.glob(os.path.join(args.dataset_root, '*')))
    
    for seq in tqdm(seqs):
        if not os.path.isdir(seq): continue
        seq_name = os.path.basename(seq)
        gt_path = os.path.join(seq, 'gt/gt.txt')
        out_path = os.path.join(args.output_dir, f"{seq_name}_id_viz.mp4")
        process_sequence(seq, gt_path, out_path, model, device, args)