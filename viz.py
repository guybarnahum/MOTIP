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
import subprocess
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from collections import defaultdict

# Import your existing modules
from utils_inference import recover_embeddings
from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker

# --- Import Memory Manager ---
try:
    from models.longterm_memory import LongTermMemory
except ImportError:
    print("⚠️ Warning: models/longterm_memory.py not found. Viz will skip Re-ID.")
    LongTermMemory = None

# -------------------------------------------------------------------------
# NEW: Metric Accumulator (The Brain 🧠)
# -------------------------------------------------------------------------
class MetricAccumulator:
    def __init__(self):
        self.stats = {
            'GT': 0, 'TP': 0, 'FP': 0, 'FN': 0, 'IDSW': 0
        }
        # For Histogram
        self.tp_scores = []
        self.fp_scores = []
        
        # For IDF1 (Global ID Mapping)
        # matches[gt_id][pred_id] = count of frames they matched
        self.matches = defaultdict(lambda: defaultdict(int))
        self.gt_id_frames = defaultdict(int)   # Total frames each GT_ID exists
        self.pred_id_frames = defaultdict(int) # Total frames each Pred_ID exists
        
    def update(self, seq_name, tp_count, fp_count, fn_count, idsw_count, gt_count, match_pairs, unmapped_gt_ids, unmapped_pred_ids):
        """
        seq_name: str, used to make IDs unique across videos to fix Global Identity Collision.
        """
        self.stats['TP'] += tp_count
        self.stats['FP'] += fp_count
        self.stats['FN'] += fn_count
        self.stats['IDSW'] += idsw_count
        self.stats['GT'] += gt_count
        
        # Update IDF1 tracking with Sequence-Aware IDs
        for gid, pid in match_pairs:
            unique_gid = f"{seq_name}_{gid}"
            unique_pid = f"{seq_name}_{pid}"
            self.matches[unique_gid][unique_pid] += 1
            self.gt_id_frames[unique_gid] += 1
            self.pred_id_frames[unique_pid] += 1
            
        for gid in unmapped_gt_ids:
            unique_gid = f"{seq_name}_{gid}"
            self.gt_id_frames[unique_gid] += 1
            
        for pid in unmapped_pred_ids:
            unique_pid = f"{seq_name}_{pid}"
            self.pred_id_frames[unique_pid] += 1

    def add_confidences(self, tps_scores, fps_scores):
        self.tp_scores.extend(tps_scores)
        self.fp_scores.extend(fps_scores)

    def compute_metrics(self):
        gt = max(1, self.stats['GT'])
        # Simplified MOTA calculation
        mota = 1.0 - (self.stats['FN'] + self.stats['FP'] + self.stats['IDSW']) / gt
        
        # Safety for zero predictions
        denom_prec = max(1, self.stats['TP'] + self.stats['FP'])
        precision = self.stats['TP'] / denom_prec
        recall = self.stats['TP'] / gt
        
        # --- IDF1 Computation (On-the-fly Bipartite Matching) ---
        # 1. Build Cost Matrix from self.matches
        gt_ids = list(self.gt_id_frames.keys())
        pred_ids = list(self.pred_id_frames.keys())
        
        if not gt_ids or not pred_ids:
            idf1 = 0.0
        else:
            # Map string IDs to integers 0..N
            gt_map = {g: i for i, g in enumerate(gt_ids)}
            pred_map = {p: i for i, p in enumerate(pred_ids)}
            
            # Matrix size [num_gt, num_pred] filled with negative match counts (for minimization)
            cost_matrix = np.zeros((len(gt_ids), len(pred_ids)))
            
            for gid_key, pid_dict in self.matches.items():
                i = gt_map[gid_key]
                for pid_key, count in pid_dict.items():
                    if pid_key in pred_map:
                        j = pred_map[pid_key]
                        cost_matrix[i, j] = -count  # Negative because linear_sum_assignment finds min cost

            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            # Sum up IDTP (Identity True Positives)
            idtp = 0
            for r, c in zip(row_ind, col_ind):
                # Flip back sign
                idtp += (-cost_matrix[r, c])
                
            # IDF1 Formula: 2 * IDTP / (Sum(GT_Frames) + Sum(Pred_Frames))
            total_gt_frames = sum(self.gt_id_frames.values())
            total_pred_frames = sum(self.pred_id_frames.values())
            
            denom = total_gt_frames + total_pred_frames
            idf1 = (2 * idtp) / denom if denom > 0 else 0.0

        return {
            'MOTA': mota * 100,
            'IDF1': idf1 * 100,
            'DetPr': precision * 100,
            'DetRe': recall * 100,
            'IDSW': self.stats['IDSW'],
            'GT': self.stats['GT'],
            'TP': self.stats['TP'],
            'FP': self.stats['FP'],
            'FN': self.stats['FN']
        }

    def plot_histogram(self, output_dir):
        if not self.tp_scores and not self.fp_scores:
            return
            
        plt.figure(figsize=(10, 6))
        # Plot histogram of confidences
        plt.hist([self.tp_scores, self.fp_scores], bins=20, range=(0, 1), 
                 color=['green', 'red'], label=['True Positives', 'False Positives (Ghosts)'],
                 stacked=True, alpha=0.7)
        plt.axvline(x=0.5, color='blue', linestyle='--', label='Suggested Cutoff (0.5)')
        plt.title('Confidence Distribution: Real Cars vs Ghosts')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(output_dir, 'confidence_hist.png')
        plt.savefig(save_path)
        plt.close()
        print(f"📊 Histogram saved to: {save_path}")

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
            # MOT Format: frame, id, left, top, width, height, conf, class, visibility
            frame = int(parts[0])
            obj_id = int(parts[1])
            x, y, w, h = map(float, parts[2:6])
            
            # --- FILTERING LOGIC ---
            # Check if class/conf exists (standard MOT format has 9+ columns)
            if len(parts) >= 8:
                conf = float(parts[6])
                cls = int(float(parts[7])) # Sometimes parsed as float
                
                # Filter 1: Ignore 'conf=0' (Standard 'Don't Care' flag in MOT)
                if conf == 0:
                    continue
                
                # Filter 2: Select only specific classes (Adjust based on your dataset!)
                # BDD/DanceTrack usually use 1 for the main class (Car/Pedestrian)
                # If you are tracking Cars, ensure you aren't loading Pedestrians(1) or Trucks(3)
                # valid_classes = [1] # Uncomment this if you know your class ID
                # if cls not in valid_classes:
                #    continue

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
    if len(boxes) == 0:
        return np.empty((0, 4))
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
# Helper: H.264 Conversion (Quiet)
# -------------------------------------------------------------------------
def convert_to_h264_quiet(input_path, output_path):
    """
    Converts video to H.264 using ffmpeg without breaking tqdm.
    """
    try:
        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '23',
            '-pix_fmt', 'yuv420p',
            '-loglevel', 'error', # Suppress ffmpeg output
            output_path
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        tqdm.write(f"✅ H.264: {os.path.basename(output_path)}")
    except subprocess.CalledProcessError:
        tqdm.write(f"❌ Error converting {os.path.basename(input_path)}")

# -------------------------------------------------------------------------
# Main Processor
# -------------------------------------------------------------------------
def process_sequence(seq_path, gt_path, output_path, model, device, args, metrics):
    cap = cv2.VideoCapture(os.path.join(seq_path, 'img1/%08d.jpg'))
    if not cap.isOpened(): return

    W, H = int(cap.get(3)), int(cap.get(4))
    fps = 10
    seq_name = os.path.basename(seq_path.rstrip('/'))
    
    tracker = RuntimeTracker(
        model=model, sequence_hw=(H, W),
        miss_tolerance=30, det_thresh=args.score_thresh,
        dtype=torch.float32
    )
    tracker.bbox_unnorm = torch.tensor([W, H, W, H], device=device).float()
    
    # 🧠 Initialize Memory for this specific sequence
    memory = None
    if LongTermMemory is not None:
        memory = LongTermMemory(patience=900, gallery_size=5, similarity_thresh=0.85)

    gt_data = load_mot_gt(gt_path)
    
    # NEW: Setup Result Export (Standard MOT Format)
    res_file = os.path.join(args.output_dir, 'results', f"{os.path.basename(seq_path.rstrip('/'))}.txt")
    os.makedirs(os.path.dirname(res_file), exist_ok=True)
    res_f = open(res_file, 'w')
    
    # Temp file for OpenCV writing
    temp_out = output_path.replace(".mp4", "_temp.mp4")
    out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    gt_id_to_pred_id = {} 
    frame_idx = 1 
    
    # NEW: Sequence Accumulator for Running Stats (Visual Only)
    seq_stats = MetricAccumulator()

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
        
        # --- Memory Correction Logic ---
        if memory is not None and "embeddings" in res:
            raw_ids = res["id"].tolist()
            if len(raw_ids) > 0:
                # Update memory & get re-mapping
                id_map = memory.update(frame_idx, raw_ids, res["embeddings"])
                
                # Remap IDs
                remapped_ids = [id_map.get(rid, rid) for rid in raw_ids]
                res["id"] = torch.tensor(remapped_ids, dtype=torch.int64, device=res["id"].device)
        # -------------------------------

        raw_pred_boxes = res['bbox'].cpu().numpy()
        pred_ids = res['id'].tolist()
        # NEW: Handle scores for histogram
        pred_scores = res.get('scores', torch.ones(len(pred_ids))).cpu().tolist()
        
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
        
        # NEW: Metric Tracking per frame
        frame_tp = 0
        frame_idsw = 0
        tps_scores = []
        fps_scores = []
        match_pairs = [] # Store pairs for IDF1
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= 0.5:
                matches.append((r, c))
                matched_gt_indices.add(c)
                matched_pred_indices.add(r)
                
                # IDF1 Match pair
                pid = pred_ids[r]
                gid = gt_ids_frame[c]
                match_pairs.append((gid, pid))

                frame_tp += 1
                tps_scores.append(pred_scores[r])

                previous_pid = gt_id_to_pred_id.get(gid)
                
                if previous_pid is not None and previous_pid != pid:
                    frame_idsw += 1
                
                gt_id_to_pred_id[gid] = pid

        # Metric: FP / FN / GT
        frame_fp = len(pred_boxes) - len(matched_pred_indices)
        frame_fn = len(gt_boxes) - len(matched_gt_indices)
        frame_gt = len(gt_boxes)
        
        # Collect FP scores for histogram
        for i, score in enumerate(pred_scores):
            if i not in matched_pred_indices:
                fps_scores.append(score)
        
        # Identify Unmapped IDs for IDF1 update logic
        unmapped_gts = [gt_ids_frame[i] for i in range(len(gt_ids_frame)) if i not in matched_gt_indices]
        unmapped_preds = [pred_ids[i] for i in range(len(pred_ids)) if i not in matched_pred_indices]

        # Update metrics (Pass seq_name for Global ID separation)
        metrics.update(seq_name, frame_tp, frame_fp, frame_fn, frame_idsw, frame_gt, match_pairs, unmapped_gts, unmapped_preds)
        metrics.add_confidences(tps_scores, fps_scores)
        
        # Update Sequence Stats (Pass dummy name since this is isolated per video)
        seq_stats.update("viz", frame_tp, frame_fp, frame_fn, frame_idsw, frame_gt, match_pairs, unmapped_gts, unmapped_preds)
        current_stats = seq_stats.compute_metrics()

        # Write txt results
        for i, bbox in enumerate(raw_pred_boxes):
            l, t, w, h = bbox
            score = pred_scores[i]
            pid = pred_ids[i]
            # Write line standard MOT: frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
            res_f.write(f"{frame_idx},{pid},{l:.2f},{t:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n")

        # --- DRAWING ---
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
                # Show score to help debug ghosts
                cv2.putText(frame, f"FP {pred_scores[i]:.2f}", (x1, y1-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Matches (Green/Orange)
        for r, c in matches:
            pbox = pred_boxes[r]
            pid = pred_ids[r]
            gid = gt_ids_frame[c]
            x1, y1, x2, y2 = map(int, pbox)
            
            # Re-check switch logic for drawing color (Redundant but keeps drawing code isolated)
            previous_pid = gt_id_to_pred_id.get(gid)
            if previous_pid is not None and previous_pid != pid: 
                color = (0, 165, 255) # Orange
                text = f"SWITCH! {previous_pid}->{pid}"
            else:
                color = (0, 255, 0)
                text = f"P:{pid} G:{gid}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Legend (Expanded for Running Stats)
        cv2.rectangle(frame, (5, 5), (300, 210), (0,0,0), -1)
        cv2.putText(frame, "Green: Stable Match", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "Orange: ID SWITCH", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, "Red Box: False Pos", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "Blue: Missed GT", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Real-time Metrics Overlay
        cv2.putText(frame, f"Thresh: {args.score_thresh}", (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Run MOTA: {current_stats['MOTA']:.1f}%", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Run IDF1: {current_stats['IDF1']:.1f}%", (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Run Prec: {current_stats['DetPr']:.1f}%", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        out.write(frame)
        frame_idx += 1

    out.release()
    cap.release()
    res_f.close()
    
    # H.264 Conversion (Safe)
    if os.path.exists(temp_out):
        convert_to_h264_quiet(temp_out, output_path)
        if os.path.exists(output_path):
            os.remove(temp_out)

def copy_training_metrics(checkpoint_path, output_dir):
    """
    Looks for 'train.png' in the same folder as the checkpoint and copies it to output_dir.
    Returns True if found and copied, False otherwise.
    """
    ckpt_dir = os.path.dirname(checkpoint_path)
    potential_names = ['train.png', 'loss.png', 'metrics.png', 'train_dashboard.png']
    
    found_path = None
    for name in potential_names:
        p = os.path.join(ckpt_dir, name)
        if os.path.exists(p):
            found_path = p
            break
            
    if found_path:
        print(f"📊 Found training metrics: {found_path}")
        dest = os.path.join(output_dir, "train.png")
        shutil.copy(found_path, dest)
        return True
    else:
        return False

def generate_html_viewer(output_dir, video_files, template_path="viz_viewer.html"):
    """Generates the viewer by creating video_data.js and copying the HTML template."""
    print(f"📝 Generating Portable HTML viewer for {len(video_files)} videos...")
    
    filenames = [os.path.basename(f) for f in video_files]
    
    js_path = os.path.join(output_dir, "video_data.js")
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(f"const videoList = {json.dumps(filenames, indent=2)};\n")

    dest_path = os.path.join(output_dir, "index.html")
    
    if os.path.exists(template_path):
        shutil.copy(template_path, dest_path)
        print(f"✅ Viewer generated: {dest_path}")
        print(f"   Data File: {js_path}")
        print(f"   To view: cd {output_dir} && python -m http.server")
    else:
        print(f"⚠️  Template file '{template_path}' not found! Generated only data JS.")
        print(f"   Please place 'viz_viewer.html' in the working directory.")

# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/pretrain_r50_deformable_detr_bdd_mini.yaml')
    # Checkpoint optional in argparse, manually checked later
    parser.add_argument('--checkpoint', type=str, default=None)
    
    # Input Modes
    parser.add_argument('--dataset_root', type=str, default='./datasets/DanceTrack/val',
                        help="Path to a folder containing multiple sequences")
    parser.add_argument('--input_video', type=str, default=None,
                        help="Path to a SINGLE specific sequence directory (overrides dataset_root)")
    
    parser.add_argument('--output_dir', type=str, default='./outputs/stage1_id_viz')
    parser.add_argument('--score_thresh', type=float, default=0.4)
    parser.add_argument('--html_template', type=str, default='viz_viewer.html', 
                       help="Path to the HTML viewer template file")
    
    # New Flag
    parser.add_argument('--html_only', action='store_true', 
                       help="Skip video generation and only regenerate the HTML viewer")
    
    args = parser.parse_args()

    # Manual Validation: Checkpoint is required ONLY if NOT html_only
    if not args.html_only and args.checkpoint is None:
        parser.error("the following arguments are required: --checkpoint")

    # ---------------------------------------------------------------------
    # 🛡️ VRAM SAFETY FUSE (Skip if HTML Only)
    # ---------------------------------------------------------------------
    SAFETY_LIMIT = 0.15 
    if not args.html_only and torch.cuda.is_available():
        print(f"🔒 ACTIVATING VRAM SAFETY LIMITER: Max {SAFETY_LIMIT*100:.0f}% of GPU")
        try:
            torch.cuda.set_per_process_memory_fraction(SAFETY_LIMIT, 0)
        except RuntimeError:
            print("⚠️ Warning: Could not set memory limit (CUDA already initialized?)")
            print("❌ CRITICAL SAFETY ERROR: Aborting to protect active training run.")
            sys.exit(1)

    try:
        os.makedirs(args.output_dir, exist_ok=True)
        # NEW: Accumulator
        metrics = MetricAccumulator()
        generated_videos = []

        # --- HTML ONLY MODE ---
        if args.html_only:
            print("⏩ HTML Only Mode: Skipping model loading and video generation.")
            # Scan output dir for existing mp4s
            generated_videos = sorted(glob.glob(os.path.join(args.output_dir, '*.mp4')))
            print(f"   Found {len(generated_videos)} existing videos.")
            
            # Optional copy of metrics if checkpoint path was provided (but not required)
            if args.checkpoint:
                copy_training_metrics(args.checkpoint, args.output_dir)
            
            generate_html_viewer(args.output_dir, generated_videos, args.html_template)
            
        else:
            # --- NORMAL PROCESSING MODE ---
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

            # Input Logic
            if args.input_video:
                if not os.path.exists(args.input_video):
                    print(f"❌ Error: Input path not found: {args.input_video}")
                    sys.exit(1)
                seqs = [args.input_video]
                print(f"🎯 Processing SINGLE video: {args.input_video}")
            else:
                seqs = sorted(glob.glob(os.path.join(args.dataset_root, '*')))
                print(f"📂 Processing DATASET: {len(seqs)} sequences")
            
            for seq in tqdm(seqs):
                if not os.path.isdir(seq): continue
                seq_name = os.path.basename(seq.rstrip('/')) 
                gt_path = os.path.join(seq, 'gt/gt.txt')
                out_path = os.path.join(args.output_dir, f"{seq_name}_id_viz.mp4")
                
                # Pass metrics accumulator to processor
                process_sequence(seq, gt_path, out_path, model, device, args, metrics)
                
                if os.path.exists(out_path):
                    generated_videos.append(out_path)

            # Copy Metrics
            copy_training_metrics(args.checkpoint, args.output_dir)

            # Generate Viewer
            if generated_videos:
                generate_html_viewer(args.output_dir, generated_videos, args.html_template)
            else:
                print("⚠️ No videos generated. Check dataset path.")
            
            # NEW: Compute and Print Final Stats
            final_stats = metrics.compute_metrics()
            print("\n" + "="*40)
            print(f"📊 FINAL STATS (Threshold: {args.score_thresh})")
            print("="*40)
            print(f" MOTA  : {final_stats['MOTA']:.2f} %")
            print(f" IDF1  : {final_stats['IDF1']:.2f} %") # Added IDF1
            print(f" DetPr : {final_stats['DetPr']:.2f} %")
            print(f" DetRe : {final_stats['DetRe']:.2f} %")
            print(f" TP    : {final_stats['TP']}")
            print(f" FP    : {final_stats['FP']}")
            print(f" FN    : {final_stats['FN']}")
            print(f" IDSW  : {final_stats['IDSW']}")
            print("="*40)
            
            # NEW: Plot Histogram
            metrics.plot_histogram(args.output_dir)

    except torch.cuda.OutOfMemoryError:
        print("\n" + "🟥"*32)
        print("VRAM SAFETY FUSE BLOWN")
        print("🟥"*32)
        print(f"The script probably hit the {SAFETY_LIMIT*100:.0f}% memory limit and stopped itself.")
        print("\n✅ GOOD NEWS: any training run or other GPU work are SAFE.")
        print("❌ BAD NEWS: Visualization could not finish.")
        print("👉 ACTION: Wait for GPU availability and run again.")
        print("="*64 + "\n")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        # Optional: Print traceback to debug
        import traceback; traceback.print_exc()
        sys.exit(1)