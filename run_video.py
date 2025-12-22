import argparse
import cv2
import torch
import os
import sys
import yaml
import time
import torch.nn.functional as F
import warnings
import types

# Custom Modules
from memory_manager import LongTermMemory
from utils_inference import generate_colors, convert_to_h264, recover_embeddings
from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

# -------------------------------------------------------------------------
# MONKEY PATCH: Enable Memory without editing runtime_tracker.py
# -------------------------------------------------------------------------
# We hook into '_get_activate_detections' to intercept the model output
# before the tracker throws it away.
original_get_activate_detections = RuntimeTracker._get_activate_detections

def patched_get_activate_detections(self, detr_out):
    # 1. Save the raw output so our external script can see it
    self.output = detr_out
    # 2. Run the original logic
    return original_get_activate_detections(self, detr_out)

# Apply the patch
RuntimeTracker._get_activate_detections = patched_get_activate_detections
print("🔧 RuntimeTracker patched successfully for Long-Term Memory support.")
# -------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser("MOTIP + Memory")
    
    # Standard MOTIP Arguments
    parser.add_argument('--config_path', type=str, default='./configs/r50_motip_bdd100k.yaml')
    parser.add_argument('--checkpoint', type=str, default='./pretrains/motip_bdd100k.pth')
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="output_final.mp4")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--score_thresh', type=float, default=0.5)
    
    # RESTORED: Frame Range Arguments (Backward Compatibility)
    parser.add_argument('--start_frame', type=int, default=0, help="Frame to start processing")
    parser.add_argument('--end_frame', type=int, default=None, help="Frame to stop processing")
    
    # New Memory Settings
    parser.add_argument('--miss_tolerance', type=int, default=30, help="Short term tracker memory")
    parser.add_argument('--longterm_patience', type=int, default=9000, help="Long term gallery memory")
    
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting MOTIP on {device}")

    # 1. Setup Model
    with open(args.config_path, 'r') as f: cfg = yaml.safe_load(f)
    if 'DEVICE' not in cfg: cfg['DEVICE'] = args.device
    
    model = build_model(cfg)
    model = model[0] if isinstance(model, tuple) else model
    model.to(device)
    
    print(f"📥 Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)

    # 2. Setup Modules
    memory = LongTermMemory(patience=args.longterm_patience)
    
    # 3. Video IO Setup
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {args.video_path}")
        return

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle Frame Range
    start_frame = max(0, args.start_frame)
    end_frame = args.end_frame if args.end_frame is not None else total_frames
    end_frame = min(end_frame, total_frames)
    
    print(f"🎬 Processing {args.video_path}")
    print(f"⏱️  Range: Frame {start_frame} -> {end_frame} ({end_frame - start_frame} frames)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Initialize Tracker
    tracker = RuntimeTracker(
        model=model,
        sequence_hw=(H, W), # Note: RuntimeTracker expects (H, W) tuple
        miss_tolerance=args.miss_tolerance,
        det_thresh=args.score_thresh,
        dtype=torch.float16 if args.fp16 else torch.float32
    )
    # Fix bbox normalization tensor for the tracker
    tracker.bbox_unnorm = torch.tensor([W, H, W, H], device=device, dtype=tracker.dtype)

    temp_out = "temp_" + os.path.basename(args.output_path)
    out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    colors = generate_colors()

    # Image Normalization Stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.fp16: mean, std = mean.half(), std.half()

    frame_idx = start_frame
    
    try:
        while cap.isOpened() and frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret: break

            # Preprocess
            t_img = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255.0
            if args.fp16: t_img = t_img.half()
            
            # Smart Resize (maintain aspect ratio, max 1333 size)
            h, w = t_img.shape[1:]
            scale = 800 / min(h, w)
            if max(h, w) * scale > 1333: scale = 1333 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            img_resized = F.interpolate(t_img.unsqueeze(0), size=(new_h, new_w), mode='bilinear')
            img_norm = (img_resized - mean) / std

            # --- A. INFERENCE ---
            tracker.update(img_norm)
            
            # --- B. DATA EXTRACTION ---
            res = tracker.get_track_results()
            valid_boxes = res['bbox'] # [N, 4] xywh
            valid_ids = res['id']     # [N]
            
            # Recover Embeddings using our Helper (Works because of Monkey Patch above)
            active_embeds = recover_embeddings(tracker, valid_boxes, W, H, device)

            # --- C. MEMORY UPDATE ---
            mapped_ids = []
            if active_embeds is not None:
                id_map = memory.update(frame_idx, valid_ids.tolist(), active_embeds)
                # Map the IDs: If map exists use it, else use original ID
                mapped_ids = [id_map.get(vid, vid) for vid in valid_ids.tolist()]
            else:
                mapped_ids = valid_ids.tolist()

            # --- D. DRAWING ---
            for i, obj_id in enumerate(mapped_ids):
                x, y, wb, hb = valid_boxes[i].cpu().float().numpy()
                color = [int(c) for c in colors[int(obj_id) % 1000]]
                
                # Draw Box
                cv2.rectangle(frame, (int(x), int(y)), (int(x+wb), int(y+hb)), color, 2)
                
                # Draw Label
                label = f"ID {obj_id}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x), int(y)-20), (int(x)+tw, int(y)), color, -1)
                cv2.putText(frame, label, (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            # Draw Status
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(frame)
            frame_idx += 1
            if frame_idx % 20 == 0: 
                print(f"\rProgress: {frame_idx}/{end_frame} ({(frame_idx-start_frame)/(end_frame-start_frame)*100:.1f}%)", end="")

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        out.release()
        convert_to_h264(temp_out, args.output_path)
        if os.path.exists(temp_out): os.remove(temp_out)
        print(f"\nDone. Output saved to: {args.output_path}")

if __name__ == '__main__':
    main()