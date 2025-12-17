import argparse
import cv2
import torch
import numpy as np
import os
import sys
from pathlib import Path

# Add root to path so we can import internal modules
sys.path.append(os.getcwd())

from models import build_model
from util.tool import load_model
import torchvision.transforms as T
from PIL import Image

# -------------------------------------------------------------------------
# Configuration & Setup
# -------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser("MOTIP Video Inference")
    
    # Model Config
    parser.add_argument('--config_path', type=str, required=True, help="Path to model config .yaml")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument('--video_path', type=str, required=True, help="Input video path")
    parser.add_argument('--output_path', type=str, default="output_direct.mp4", help="Output video path")
    parser.add_argument('--score_thresh', type=float, default=0.5, help="Confidence threshold to draw")
    parser.add_argument('--device', type=str, default="cuda")
    
    # MOTIP specific args usually needed by build_model
    # We add dummy defaults for arguments the model build process might expect
    parser.add_argument('--aux_loss', default=False, action='store_true')
    parser.add_argument('--masks', default=False, action='store_true')
    parser.add_argument('--meta_arch', default='motip', type=str)
    
    # Load the YAML config to populate other necessary args (hidden step)
    # For simplicity, we assume the user provides the yaml and we parse it
    # But usually, standard args are enough if we load the state_dict correctly.
    
    return parser.parse_args()

def get_transform():
    """
    Standard ImageNet normalization required by Deformable DETR / MOTIP.
    """
    return T.Compose([
        T.Resize((800, 1333)), # Resize short side to 800, max long side 1333
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# -------------------------------------------------------------------------
# Visualization Utilities
# -------------------------------------------------------------------------
COLORS = np.random.randint(0, 255, size=(1000, 3), dtype="uint8")

def draw_tracks(img, tracks, thresh=0.5):
    """
    img: opencv image (BGR)
    tracks: list of dicts or objects containing box and scores
    """
    # MOTIP outputs usually come as 'pred_logits' and 'pred_boxes'
    # but inside the loop we will process them into cleaner objects
    
    for track in tracks:
        score = track['score']
        if score < thresh:
            continue
            
        # Unpack box (cx, cy, w, h) -> (x1, y1, x2, y2)
        # We assume the box is already denormalized to pixel coordinates
        x1, y1, x2, y2 = track['box'].astype(int)
        obj_id = track['id']
        
        color = [int(c) for c in COLORS[obj_id % 1000]]
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID:{obj_id} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img

# -------------------------------------------------------------------------
# Main Inference Loop
# -------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)

    # 1. Load Configuration
    # We rely on the internal config parser of MOTIP if possible, 
    # but here we'll try to load the model structure directly.
    # NOTE: You might need to install 'yacs' or similar if the repo uses it.
    import yaml
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Merge cfg into args for build_model compatibility
    for k, v in cfg.items():
        setattr(args, k, v)

    print("🏗️  Building Model...")
    model, _, _ = build_model(args)
    model.to(device)
    
    print(f"📥 Loading Checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 2. Setup Video
    cap = cv2.VideoCapture(args.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Output Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (orig_w, orig_h))
    
    transform = get_transform()
    
    # 3. Initialize Tracker State
    # MOTIP/MOTR requires passing 'track_instances' from frame t to t+1
    track_instances = None 

    print("🚀 Starting Inference...")
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR (OpenCV) -> RGB (PIL) -> Tensor
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        # Preprocess
        # 'samples' expects NestedTensor usually, but plain tensor works for eval in some DETR variants
        # If it fails, we wrap it manually.
        img_tensor = transform(pil_img).unsqueeze(0).to(device) 
        
        # Construct Sample (Pseudo NestedTensor if needed by codebase)
        # For simplicity, we pass the tensor directly. 
        # If MOTIP demands nested tensors: 
        # samples = NestedTensor(img_tensor, mask)
        
        # Forward Pass
        # outputs usually contains: 'pred_logits', 'pred_boxes', 'track_instances', etc.
        outputs = model(img_tensor, track_instances=track_instances)
        
        # Update State for next frame
        track_instances = outputs['track_instances']
        
        # 4. Decode Outputs
        # Boxes are typically [cx, cy, w, h] normalized (0-1)
        logits = outputs['pred_logits'][0] # [num_queries, num_classes]
        boxes = outputs['pred_boxes'][0]   # [num_queries, 4]
        
        # Convert scores
        scores = logits.softmax(-1)[:, 0] # Assuming class 0 is 'person' or 'object'
        # Or if sigmoid focal loss: scores = logits.sigmoid()
        
        keep = scores > args.score_thresh
        
        valid_scores = scores[keep]
        valid_boxes = boxes[keep]
        
        # Retrieve IDs
        # In MOTIP, IDs are often stored in track_instances.obj_idx or similar
        # We need to extract them from the state.
        # This part depends heavily on MOTIP's specific structure.
        # Usually: valid_ids = track_instances.obj_ids[keep]
        # Let's try to infer from typical structure:
        if hasattr(track_instances, 'obj_ids'):
            valid_ids = track_instances.obj_ids[keep]
        else:
            # Fallback if structure differs
            valid_ids = torch.arange(len(valid_scores)) 

        # Denormalize Boxes (cxcywh 0..1 -> xyxy pixels)
        h, w = orig_h, orig_w
        valid_boxes = valid_boxes * torch.tensor([w, h, w, h], device=device)
        
        # Convert cx, cy, w, h -> x1, y1, x2, y2
        b_x1 = valid_boxes[:, 0] - valid_boxes[:, 2] / 2
        b_y1 = valid_boxes[:, 1] - valid_boxes[:, 3] / 2
        b_x2 = valid_boxes[:, 0] + valid_boxes[:, 2] / 2
        b_y2 = valid_boxes[:, 1] + valid_boxes[:, 3] / 2
        
        processed_tracks = []
        for i in range(len(valid_scores)):
            processed_tracks.append({
                'score': valid_scores[i].item(),
                'box': np.array([b_x1[i].item(), b_y1[i].item(), b_x2[i].item(), b_y2[i].item()]),
                'id': int(valid_ids[i].item()) if isinstance(valid_ids, torch.Tensor) else i
            })
            
        # 5. Draw
        out_frame = draw_tracks(frame, processed_tracks, thresh=args.score_thresh)
        out.write(out_frame)
        
        print(f"Processing frame {frame_idx}...", end='\r')
        frame_idx += 1
        
    cap.release()
    out.release()
    print("\n✅ Done!")

if __name__ == '__main__':
    main()
