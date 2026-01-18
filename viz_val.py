import argparse
import cv2
import torch
import os
import sys
import yaml
import numpy as np
import glob
import json
import shutil
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

# Import your existing modules
from memory_manager import LongTermMemory
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
# Helper: Ground Truth Parsing
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
# Helper: Prediction Format Conversion
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
    
    # Temp file for OpenCV writing
    temp_out = output_path.replace(".mp4", "_temp.mp4")
    out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    gt_id_to_pred_id = {} 
    frame_idx = 1 
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Inference
        t_img = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255.0
        h, w = t_img.shape[1:]
        scale = 800 / min(h, w)
        if max(h, w) * scale > 1333: scale = 1333 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = torch.nn.functional.interpolate(t_img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
        img_norm = (img_resized - mean) / std
        
        tracker.update(img_norm)
        res = tracker.get_track_results()
        
        raw_pred_boxes = res['bbox'].cpu().numpy()
        pred_ids = res['id'].tolist()
        pred_boxes = convert_tlwh_to_xyxy(raw_pred_boxes)
        
        current_gt = gt_data.get(frame_idx, [])
        gt_boxes = [g['bbox'] for g in current_gt]
        gt_ids_frame = [g['id'] for g in current_gt]
        
        # Match Predictions to GT
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

        # Draw
        # Misses (Blue)
        for i, gbox in enumerate(gt_boxes):
            if i not in matched_gt_indices:
                x1, y1, x2, y2 = map(int, gbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"MISS G:{gt_ids_frame[i]}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # False Positives (Red)
        for i, pbox in enumerate(pred_boxes):
            if i not in matched_pred_indices:
                x1, y1, x2, y2 = map(int, pbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"FP P:{pred_ids[i]}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Matches (Green/Orange)
        for r, c in matches:
            pbox = pred_boxes[r]
            pid = pred_ids[r]
            gid = gt_ids_frame[c]
            x1, y1, x2, y2 = map(int, pbox)
            
            previous_pid = gt_id_to_pred_id.get(gid)
            if previous_pid is None:
                color = (0, 255, 0)
                text = f"P:{pid} G:{gid}"
                gt_id_to_pred_id[gid] = pid
            elif previous_pid != pid:
                color = (0, 165, 255) # Orange
                text = f"SWITCH! {previous_pid}->{pid}"
                gt_id_to_pred_id[gid] = pid
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            else:
                color = (0, 255, 0)
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
    
    # H.264 Conversion
    if os.path.exists(temp_out):
        convert_to_h264(temp_out, output_path)
        if os.path.exists(output_path):
            os.remove(temp_out)

def generate_html_viewer(output_dir, video_files, template_path="viewer_template.html"):
    """Generates the viewer by creating video_data.js and copying the HTML template."""
    print(f"📝 Generating HTML viewer for {len(video_files)} videos...")
    
    filenames = [os.path.basename(f) for f in video_files]
    
    # 1. Write Data JS
    js_path = os.path.join(output_dir, "video_data.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(f"const videoList = {json.dumps(filenames, indent=2)};")

    # 2. Copy Template to index.html
    dest_path = os.path.join(output_dir, "index.html")
    
    if os.path.exists(template_path):
        shutil.copy(template_path, dest_path)
        print(f"✅ Viewer generated: {dest_path}")
        print(f"   Data File: {js_path}")
        print(f"   To view: cd {output_dir} && python -m http.server")
    else:
        print(f"⚠️  Template file '{template_path}' not found! Generated only data JS.")
        print(f"   Please place 'viewer_template.html' in the working directory.")

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    
    # ---------------------------------------------------------------------
    # 🛡️ VRAM SAFETY FUSE
    # ---------------------------------------------------------------------
    # We restrict this script to 15% of GPU memory (~3.5GB on A10G).
    # If it happens to use more, it kills itself before hurting the training run.
    SAFETY_LIMIT = 0.15 
    
    if torch.cuda.is_available():
        print(f"🔒 ACTIVATING VRAM SAFETY LIMITER: Max {SAFETY_LIMIT*100:.0f}% of GPU")
        try:
            torch.cuda.set_per_process_memory_fraction(SAFETY_LIMIT, 0)
        except RuntimeError:
            print("⚠️ Warning: Could not set memory limit (CUDA already initialized?)")
            sys.exit(1)
            
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/pretrain_r50_deformable_detr_bdd_mini.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--dataset_root', type=str, default='./datasets/DanceTrack/val')
    parser.add_argument('--output_dir', type=str, default='./outputs/stage1_id_viz')
    parser.add_argument('--score_thresh', type=float, default=0.4)
    # New argument to specify template location if needed
    parser.add_argument('--html_template', type=str, default='viz_viewer.html', 
                       help="Path to the HTML viewer template file")
    args = parser.parse_args()

    try:
        # Load Config
        with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
        if "SUPER_CONFIG_PATH" in cfg:
            with open(cfg["SUPER_CONFIG_PATH"], 'r') as f_base:
                base_cfg = yaml.safe_load(f_base)
            base_cfg.update(cfg)
            cfg = base_cfg

        # Init Model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = build_model(cfg)[0].to(device)
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)
        model.eval()

        os.makedirs(args.output_dir, exist_ok=True)
        seqs = sorted(glob.glob(os.path.join(args.dataset_root, '*')))
        
        generated_videos = []

        for seq in tqdm(seqs):
            if not os.path.isdir(seq): continue
            seq_name = os.path.basename(seq)
            gt_path = os.path.join(seq, 'gt/gt.txt')
            out_path = os.path.join(args.output_dir, f"{seq_name}_id_viz.mp4")
            
            process_sequence(seq, gt_path, out_path, model, device, args)
            
            if os.path.exists(out_path):
                generated_videos.append(out_path)

        # Final Step: Generate Viewer
        if generated_videos:
            generate_html_viewer(args.output_dir, generated_videos, args.html_template)
        else:
            print("⚠️ No videos generated. Check dataset path.")

    except torch.cuda.OutOfMemoryError:
        print("\n" + "🟥"*32)
        print("VRAM SAFETY FUSE BLOWN")
        print("🟥"*32)
        print(f"The script probbaly hit the {SAFETY_LIMIT*100:.0f}% memory limit and stopped itself.")
        print("\n✅ GOOD NEWS: any training run or other GPU work are SAFE.")
        print("❌ BAD NEWS: Visualization could not finish.")
        print("👉 ACTION: Wait for GPU availabiliy and run again.")
        print("="*64 + "\n")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        # Optional: Print traceback to debug
        import traceback; traceback.print_exc()
        sys.exit(1)