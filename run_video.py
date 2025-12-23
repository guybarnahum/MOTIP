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
from utils_inference import convert_to_h264, recover_embeddings
from utils_display import Annotator  # <--- NEW: Import Annotator
from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

# -------------------------------------------------------------------------
# MONKEY PATCH: Enable Memory without editing runtime_tracker.py
# -------------------------------------------------------------------------
original_get_activate_detections = RuntimeTracker._get_activate_detections

def patched_get_activate_detections(self, detr_out):
    self.output = detr_out # Save raw output
    return original_get_activate_detections(self, detr_out)

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
    
    # RESTORED: Frame Range Arguments
    parser.add_argument('--start_frame', type=int, default=0, help="Frame to start processing")
    parser.add_argument('--end_frame', type=int, default=None, help="Frame to stop processing")
    
    # Memory Settings
    parser.add_argument('--miss_tolerance', type=int, default=30, help="Short term tracker memory")
    parser.add_argument('--longterm_patience', type=int, default=9000, help="Long term gallery memory")
    
    return parser.parse_args()

@torch.no_grad()
def main():
    args = get_args()
    
    # 1. Device Setup
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"✅ Running on GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        gpu_name = "CPU"
        print("⚠️  Running on CPU (Slow!)")

    # 2. Setup Model
    with open(args.config_path, 'r') as f: cfg = yaml.safe_load(f)
    if 'DEVICE' not in cfg: cfg['DEVICE'] = args.device
    
    print("🏗️  Building model...")
    model = build_model(cfg)
    model = model[0] if isinstance(model, tuple) else model
    model.to(device)
    
    print(f"📥 Loading weights: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt.get('model', ckpt), strict=False)

    # 3. Setup Modules
    memory = LongTermMemory(patience=args.longterm_patience)
    annotator = Annotator()  # <--- NEW: Initialize Annotator
    
    # 4. Video IO Setup
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {args.video_path}")
        return

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Handle Frame Range & Duration
    start_frame = max(0, args.start_frame)
    if args.end_frame is None or args.end_frame > total_frames_in_video:
        end_frame = total_frames_in_video
    else:
        end_frame = args.end_frame
        
    process_duration = end_frame - start_frame
    if process_duration <= 0:
        print("❌ Error: end_frame must be greater than start_frame")
        sys.exit(1)

    print(f"🎬 Video: {args.video_path} ({W}x{H} @ {fps:.2f}fps)")
    print(f"⏱️  Processing Range: Frame {start_frame} to {end_frame} (Total: {process_duration} frames)")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    tracker = RuntimeTracker(
        model=model,
        sequence_hw=(H, W),
        miss_tolerance=args.miss_tolerance,
        det_thresh=args.score_thresh,
        dtype=torch.float16 if args.fp16 else torch.float32
    )
    tracker.bbox_unnorm = torch.tensor([W, H, W, H], device=device, dtype=tracker.dtype)

    temp_out = "temp_" + os.path.basename(args.output_path)
    out = cv2.VideoWriter(temp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))
    
    # Image Normalization Stats
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.fp16: mean, std = mean.half(), std.half()

    frame_idx = start_frame
    frames_processed = 0
    start_time = time.time()
    
    try:
        while cap.isOpened() and frame_idx < end_frame:
            loop_start = time.time() # Start timer
            
            ret, frame = cap.read()
            if not ret: break

            # Preprocess
            t_img = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255.0
            if args.fp16: t_img = t_img.half()
            
            # Smart Resize
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
            valid_boxes = res['bbox'].cpu().float().numpy() # [N, 4]
            valid_ids = res['id'].tolist()                  # [N]
            
            active_embeds = recover_embeddings(tracker, res['bbox'], W, H, device)

            # --- C. MEMORY UPDATE ---
            final_ids = []
            if active_embeds is not None:
                id_map = memory.update(frame_idx, valid_ids, active_embeds)
                # Map the IDs: If map exists use it, else use original ID
                final_ids = [id_map.get(vid, vid) for vid in valid_ids]
            else:
                final_ids = valid_ids

            # --- D. ANNOTATION ---
            # 1. Update FPS
            loop_time = time.time() - loop_start
            annotator.update_fps(loop_time)
            
            # 2. Draw Tracks (In-Place)
            annotator.draw_tracks(frame, valid_boxes, final_ids, valid_ids)
            
            # 3. Draw Dashboard 
            # Calculate ACTIVE revivals for this specific frame only
            # Count how many objects have a different final_id than their tracker id
            current_overrides_count = sum(1 for o, f in zip(valid_ids, final_ids) if o != f)

            mem_stats = {
                "gallery_size": len(memory.storage),
                "active_overrides": current_overrides_count 
            }

            frame = annotator.draw_dashboard(frame, frame_idx, gpu_name, mem_stats)

            out.write(frame)
            
            # Progress Bar
            if frames_processed % 20 == 0:
                progress_pct = (frames_processed / process_duration) * 100
                fps_val = annotator.fps_avg
                sys.stdout.write(f"\r   Frame {frame_idx} (Processed {frames_processed}/{process_duration}) | {progress_pct:.1f}% | FPS: {fps_val:.1f}   ")
                sys.stdout.flush()

            frame_idx += 1
            frames_processed += 1

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user! Saving video so far...")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        out.release()
        total_time = time.time() - start_time
        print(f"\n✅ Finished processing {frames_processed} frames in {total_time:.1f}s.")
        
        # Convert to H.264
        if os.path.exists(temp_out) and frames_processed > 0:
            convert_to_h264(temp_out, args.output_path)
            if os.path.exists(args.output_path):
                os.remove(temp_out)

if __name__ == '__main__':
    main()