import argparse
import cv2
import torch
import os
import sys
import yaml
import time
import torch.nn.functional as F
import warnings

# Custom Modules
from memory_manager import LongTermMemory
from utils_inference import generate_colors, convert_to_h264, recover_embeddings
from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

def get_args():
    parser = argparse.ArgumentParser("MOTIP + Memory")
    parser.add_argument('--config_path', type=str, default='./configs/r50_motip_bdd100k.yaml')
    parser.add_argument('--checkpoint', type=str, default='./pretrains/motip_bdd100k.pth')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="output_final.mp4")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--score_thresh', type=float, default=0.5)
    
    # Memory Settings
    parser.add_argument('--miss_tolerance', type=int, default=30, help="Short term tracker memory")
    parser.add_argument('--longterm_patience', type=int, default=9000, help="Long term gallery memory")
    
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device)
    print(f"🚀 Starting MOTIP on {device}")

    # 1. Setup Model
    with open(args.config_path, 'r') as f: cfg = yaml.safe_load(f)
    if 'DEVICE' not in cfg: cfg['DEVICE'] = args.device
    
    model = build_model(cfg)
    model = model[0] if isinstance(model, tuple) else model
    model.to(device)
    
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)

    # 2. Setup Modules
    memory = LongTermMemory(patience=args.longterm_patience)
    tracker = RuntimeTracker(
        model=model,
        sequence_hw=(1080, 1920), # Will update in loop if needed, but safe default
        miss_tolerance=args.miss_tolerance,
        det_thresh=args.score_thresh,
        dtype=torch.float16 if args.fp16 else torch.float32
    )

    # 3. Video IO
    cap = cv2.VideoCapture(args.video_path)
    W, H = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    tracker.bbox_unnorm = torch.tensor([W, H, W, H], device=device, dtype=tracker.dtype) # Fix resolution

    temp_out = "temp_" + args.output_path
    out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    colors = generate_colors()

    # Image Normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.fp16: mean, std = mean.half(), std.half()

    frame_idx = 0
    print(f"🎬 Processing {args.video_path}...")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            # Preprocess
            t_img = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255.0
            if args.fp16: t_img = t_img.half()
            
            # Simple resize to ~800p for inference speed/accuracy balance
            h, w = t_img.shape[1:]
            scale = 800 / min(h, w)
            if max(h, w) * scale > 1333: scale = 1333 / max(h, w)
            img_resized = F.interpolate(t_img.unsqueeze(0), size=(int(h*scale), int(w*scale)), mode='bilinear')
            img_norm = (img_resized - mean) / std

            # --- A. INFERENCE ---
            tracker.update(img_norm)
            
            # --- B. DATA EXTRACTION ---
            res = tracker.get_track_results()
            valid_boxes = res['bbox'] # [N, 4] xywh
            valid_ids = res['id']     # [N]
            
            # Recover Embeddings using our Helper
            active_embeds = recover_embeddings(tracker, valid_boxes, W, H, device)

            # --- C. MEMORY UPDATE ---
            mapped_ids = []
            if active_embeds is not None:
                id_map = memory.update(frame_idx, valid_ids.tolist(), active_embeds)
                mapped_ids = [id_map.get(vid, vid) for vid in valid_ids.tolist()]
            else:
                mapped_ids = valid_ids.tolist()

            # --- D. DRAWING ---
            for i, obj_id in enumerate(mapped_ids):
                x, y, wb, hb = valid_boxes[i].cpu().float().numpy()
                color = [int(c) for c in colors[int(obj_id) % 1000]]
                cv2.rectangle(frame, (int(x), int(y)), (int(x+wb), int(y+hb)), color, 2)
                cv2.putText(frame, f"ID {obj_id}", (int(x), int(y)-5), 0, 0.6, color, 2)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 20 == 0: print(f"\rFrame {frame_idx}", end="")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        out.release()
        convert_to_h264(temp_out, args.output_path)
        if os.path.exists(temp_out): os.remove(temp_out)
        print("\nDone.")

if __name__ == '__main__':
    main()