import argparse
import cv2
import torch
import numpy as np
import os
import sys
import yaml
import time
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

# Add root to path so we can import internal modules
sys.path.append(os.getcwd())

from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker
from utils.box_ops import box_cxcywh_to_xywh

# -------------------------------------------------------------------------
# Configuration & Setup
# -------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser("MOTIP Video Inference")
    
    # Paths
    parser.add_argument('--config_path', type=str, default='./configs/r50_deformable_detr_motip_dancetrack.yaml', help="Path to config .yaml")
    parser.add_argument('--checkpoint', type=str, default='./pretrains/motip_dancetrack.pth', help="Path to .pth checkpoint")
    parser.add_argument('--video_path', type=str, required=True, help="Input video path")
    parser.add_argument('--output_path', type=str, default="output_dance.mp4", help="Output video path")
    
    # Runtime Options
    parser.add_argument('--score_thresh', type=float, default=0.5, help="Confidence threshold")
    parser.add_argument('--device', type=str, default="cuda")
    
    return parser.parse_args()

def generate_colors(num_colors=1000):
    """ Generate random unique colors for IDs """
    np.random.seed(42)
    return np.random.randint(0, 255, size=(num_colors, 3), dtype="uint8")

# -------------------------------------------------------------------------
# Main Inference Loop
# -------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = get_args()
    
    # 1. Setup Device & Print Stats
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"✅ Running on GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("⚠️  Running on CPU (Slow!)")

    # 2. Load Configuration
    print(f"📖 Loading config: {args.config_path}")
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if 'DEVICE' not in cfg: cfg['DEVICE'] = args.device
    if 'DISTRIBUTED' not in cfg: cfg['DISTRIBUTED'] = False

    # 3. Build Model
    print("🏗️  Building model...")
    build_output = build_model(cfg)
    if isinstance(build_output, tuple):
        model = build_output[0]
    else:
        model = build_output
    
    model.to(device)
    
    # Check Precision
    param_dtype = next(model.parameters()).dtype
    print(f"📊 Model Precision: {param_dtype} (FP16 if half, FP32 if float)")

    # 4. Load Checkpoint (Fixed Warning)
    print(f"📥 Loading weights: {args.checkpoint}")
    # Added weights_only=False to silence the FutureWarning
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    # 5. Setup Video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {args.video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 Processing: {args.video_path} ({width}x{height} @ {fps:.2f}fps)")
    
    # Initialize Video Writer
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    colors = generate_colors()

    # 6. Initialize RuntimeTracker
    tracker = RuntimeTracker(
        model=model,
        sequence_hw=(height, width),
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
        det_thresh=args.score_thresh,
        newborn_thresh=cfg.get("NEWBORN_THRESH", 0.5),
        id_thresh=cfg.get("ID_THRESH", 0.1),
        area_thresh=cfg.get("AREA_THRESH", 0),
        dtype=torch.float32 
    )

    frame_idx = 0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    # FPS Calculation
    start_time = time.time()
    fps_avg = 0

    while cap.isOpened():
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).to(device).float().permute(2, 0, 1) / 255.0
        
        # Resize logic
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        scale = 800 / min(h, w)
        if max(h, w) * scale > 1333:
            scale = 1333 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        img_norm = (img_resized - mean) / std
        
        # Forward Pass
        tracker.update(img_norm)
        
        # Get Results
        results = tracker.get_track_results()
        valid_boxes = results['bbox']
        valid_ids = results['id']
        valid_scores = results['score']
        
        # Visualization
        for i in range(len(valid_scores)):
            score = valid_scores[i].item()
            obj_id = int(valid_ids[i].item())
            x, y, w_box, h_box = valid_boxes[i].cpu().numpy()
            
            x1, y1, x2, y2 = int(x), int(y), int(x + w_box), int(y + h_box)
            color = [int(c) for c in colors[obj_id % 1000]]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID {obj_id}"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw FPS
        loop_time = time.time() - loop_start
        if loop_time > 0:
            fps_inst = 1.0 / loop_time
            fps_avg = 0.9 * fps_avg + 0.1 * fps_inst if frame_idx > 0 else fps_inst
        
        cv2.putText(frame, f"FPS: {int(fps_avg)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"GPU: {torch.cuda.get_device_name(device)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        out.write(frame)
        
        if frame_idx % 20 == 0:
            print(f"   Frame {frame_idx}/{total_frames} | FPS: {fps_avg:.1f}", end='\r')
        
        frame_idx += 1

    total_time = time.time() - start_time
    print(f"\n✅ Done! Processed {frame_idx} frames in {total_time:.1f}s ({frame_idx/total_time:.1f} FPS avg)")
    print(f"   Saved to: {args.output_path}")
    cap.release()
    out.release()

if __name__ == '__main__':
    main()