import argparse
import cv2
import torch
import numpy as np
import os
import sys
import yaml
from PIL import Image
import torchvision.transforms as T

# Add root to path so we can import internal modules
sys.path.append(os.getcwd())

# --- FIX: Correct Import based on your train.py ---
from models.motip import build as build_model
from util.tool import load_model

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

def get_transform():
    """ Standard ImageNet normalization required by Deformable DETR / MOTIP """
    return T.Compose([
        T.Resize((800, 1333)), 
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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
    
    # MOTIP's build function expects the 'cfg' dictionary, not args.
    # We also inject device/distributed settings if they are missing.
    if 'DEVICE' not in cfg: cfg['DEVICE'] = args.device
    if 'DISTRIBUTED' not in cfg: cfg['DISTRIBUTED'] = False

    # 2. Build Model
    print("🏗️  Building model...")
    # FIX: Pass the dictionary 'cfg', not 'args'
    model, _, _ = build_model(cfg)
    model.to(device)
    
    # 3. Load Checkpoint
    print(f"📥 Loading weights: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Handle state dict structure (some checkpoints nest it under 'model')
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

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
    
    transform = get_transform()
    colors = generate_colors()
    
    # 5. Inference
    track_instances = None
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Preprocess
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # Forward Pass
        # track_instances carries the "memory" from previous frame
        outputs = model(img_tensor, track_instances=track_instances)
        track_instances = outputs['track_instances']
        
        # Decode Output
        logits = outputs['pred_logits'][0]
        boxes = outputs['pred_boxes'][0]
        
        # Filter by confidence
        scores = logits.softmax(-1)[:, 0] # Class 0 is Person
        keep = scores > args.score_thresh
        
        valid_scores = scores[keep]
        valid_boxes = boxes[keep]
        
        # Get IDs (MOTIP stores them in .obj_idx)
        if hasattr(track_instances, 'obj_idx'):
            valid_ids = track_instances.obj_idx[keep]
        else:
            # Fallback for older model versions
            valid_ids = torch.arange(len(valid_scores))

        # Visualization
        for i in range(len(valid_scores)):
            score = valid_scores[i].item()
            obj_id = int(valid_ids[i].item())
            
            # Denormalize Box (cx,cy,w,h -> pixel coords)
            box = valid_boxes[i] * torch.tensor([width, height, width, height], device=device)
            cx, cy, w, h = box.cpu().numpy()
            
            x1 = int(cx - w/2)
            y1 = int(cy - h/2)
            x2 = int(cx + w/2)
            y2 = int(cy + h/2)
            
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