import argparse
import cv2
import torch
import numpy as np
import os
import sys
import yaml
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
    device = torch.device(args.device)

    # 1. Load Configuration
    print(f"📖 Loading config: {args.config_path}")
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    if 'DEVICE' not in cfg: cfg['DEVICE'] = args.device
    if 'DISTRIBUTED' not in cfg: cfg['DISTRIBUTED'] = False

    # 2. Build Model
    print("🏗️  Building model...")
    build_output = build_model(cfg)
    # Handle return values (robust to 2 or 3 items)
    if isinstance(build_output, tuple):
        model = build_output[0]
    else:
        model = build_output
    
    model.to(device)
    
    # 3. Load Checkpoint
    print(f"📥 Loading weights: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    # 4. Setup Video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {args.video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 Processing: {args.video_path} ({width}x{height} @ {fps}fps)")
    
    # Initialize Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    colors = generate_colors()

    # 5. Initialize RuntimeTracker
    # We must resize images to the training size (usually short side 800)
    # BUT we need to pass the *original* sequence HW to the tracker for normalization logic
    # The tracker internally handles un-normalization based on this.
    
    # Important: RuntimeTracker expects (H, W) tuple
    tracker = RuntimeTracker(
        model=model,
        sequence_hw=(height, width),
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=cfg.get("MISS_TOLERANCE", 30),
        det_thresh=args.score_thresh, # Override config with arg
        newborn_thresh=cfg.get("NEWBORN_THRESH", 0.5),
        id_thresh=cfg.get("ID_THRESH", 0.1),
        area_thresh=cfg.get("AREA_THRESH", 0),
        dtype=torch.float32 # Use float32 for safety unless you compiled FP16 ops
    )

    frame_idx = 0
    
    # Image normalization mean/std
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess
        # 1. Convert BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Convert to Tensor (0-255 byte -> 0-1 float)
        img_tensor = torch.from_numpy(img_rgb).to(device).float().permute(2, 0, 1) / 255.0
        
        # 3. Resize (Short side 800)
        # We need to maintain aspect ratio.
        # target_h, target_w = 800, 1333 (max) logic usually handled by transforms
        # Here we do a simple resize to 800 short side for consistency with training
        h, w = img_tensor.shape[1], img_tensor.shape[2]
        scale = 800 / min(h, w)
        if max(h, w) * scale > 1333:
            scale = 1333 / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        
        # 4. Normalize
        img_norm = (img_resized - mean) / std
        
        # 5. Forward Pass (using Tracker)
        # RuntimeTracker.update() expects a NestedTensor or a simple Tensor list
        # We pass the tensor directly. The model forward handles the rest.
        tracker.update(img_norm)
        
        # 6. Get Results
        results = tracker.get_track_results()
        
        # Results structure:
        # "bbox": [N, 4] (x1, y1, w, h) absolute pixels
        # "id": [N] int64
        # "score": [N] float
        
        valid_boxes = results['bbox']
        valid_ids = results['id']
        valid_scores = results['score']
        
        # 7. Visualization
        for i in range(len(valid_scores)):
            score = valid_scores[i].item()
            obj_id = int(valid_ids[i].item())
            
            # The tracker returns XYWH (TopLeft X, TopLeft Y, W, H)
            x, y, w_box, h_box = valid_boxes[i].cpu().numpy()
            
            x1 = int(x)
            y1 = int(y)
            x2 = int(x + w_box)
            y2 = int(y + h_box)
            
            # Draw
            color = [int(c) for c in colors[obj_id % 1000]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"ID {obj_id} ({score:.2f})"
            (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + w_text, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        
        if frame_idx % 20 == 0:
            print(f"   Frame {frame_idx}/{total_frames}", end='\r')
        
        frame_idx += 1

    cap.release()
    out.release()
    print(f"\n✅ Done! Video saved to: {args.output_path}")

if __name__ == '__main__':
    main()