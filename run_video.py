import argparse
import cv2
import torch
import numpy as np
import os
import sys
import yaml
import time
import subprocess
import shutil
import torchvision.transforms as T
import torch.nn.functional as F
import warnings

# --- FIX: Suppress the meshgrid warning ---
warnings.filterwarnings("ignore", message=".*torch.meshgrid: in an upcoming release.*")

# Add root to path so we can import internal modules
sys.path.append(os.getcwd())

from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker

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
    parser.add_argument('--miss_tolerance', type=int, default=30, help="Max frames to keep ID alive without detection (default: 30)")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--fp16', action='store_true', default=True, help="Use Float16 precision for speed")
    
    # Frame Range Options
    parser.add_argument('--start_frame', type=int, default=0, help="Frame index to start processing from (default: 0)")
    parser.add_argument('--end_frame', type=int, default=None, help="Frame index to stop at (default: end of video)")
    
    return parser.parse_args()

def generate_colors(num_colors=1000):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(num_colors, 3), dtype="uint8")

def convert_to_h264(input_path, output_path):
    """
    Converts to H.264 with Timecode Metadata.
    """
    if shutil.which('ffmpeg') is None:
        print("⚠️  FFmpeg not found. Skipping H.264 conversion.")
        if input_path != output_path:
            shutil.move(input_path, output_path)
        return

    print("🔄 Converting to H.264 (Seekable + Timecode)...")
    cmd = [
        'ffmpeg', '-y', 
        '-i', input_path, 
        '-c:v', 'libx264', 
        '-preset', 'fast', 
        '-crf', '23', 
        '-g', '10',
        '-pix_fmt', 'yuv420p',
        '-metadata', 'timecode=00:00:00:00',
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ H.264 conversion complete: {output_path}")

# -------------------------------------------------------------------------
# Main Inference Loop
# -------------------------------------------------------------------------

@torch.no_grad()
def main():
    args = get_args()
    
    # 1. Setup Device
    if 'cuda' in args.device and torch.cuda.is_available():
        device = torch.device(args.device)
        gpu_name = torch.cuda.get_device_name(device)
        print(f"✅ Running on GPU: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("⚠️  Running on CPU (Slow!)")

    # 2. Load Config
    print(f"📖 Loading config: {args.config_path}")
    with open(args.config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if 'DEVICE' not in cfg: cfg['DEVICE'] = args.device
    if 'DISTRIBUTED' not in cfg: cfg['DISTRIBUTED'] = False

    # 3. Build Model
    print("🏗️  Building model...")
    build_output = build_model(cfg)
    model = build_output[0] if isinstance(build_output, tuple) else build_output
    model.to(device)

    # 4. Load Checkpoint
    print(f"📥 Loading weights: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    
    # 5. Setup Video Input
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {args.video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate Start/End Frames
    start_frame = max(0, args.start_frame)
    if args.end_frame is None or args.end_frame > total_frames_in_video:
        end_frame = total_frames_in_video
    else:
        end_frame = args.end_frame
    
    process_duration = end_frame - start_frame
    if process_duration <= 0:
        print("❌ Error: end_frame must be greater than start_frame")
        sys.exit(1)

    print(f"🎬 Video: {args.video_path} ({width}x{height} @ {fps:.2f}fps)")
    print(f"⏱️  Processing Range: Frame {start_frame} to {end_frame} (Total: {process_duration} frames)")
    
    # Seek to start frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 6. Setup Video Output
    temp_output = "temp_" + os.path.basename(args.output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    colors = generate_colors()

    # 7. Initialize RuntimeTracker
    target_dtype = torch.float16 if args.fp16 else torch.float32
    print(f"📊 Precision: {'FP16 (Half)' if args.fp16 else 'FP32 (Full)'}")
    print(f"🕵️  Tracking: Miss Tolerance = {args.miss_tolerance} frames")

    tracker = RuntimeTracker(
        model=model,
        sequence_hw=(height, width),
        use_sigmoid=cfg.get("USE_FOCAL_LOSS", False),
        assignment_protocol=cfg.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=args.miss_tolerance, # Override with CLI arg
        det_thresh=args.score_thresh,
        newborn_thresh=cfg.get("NEWBORN_THRESH", 0.5),
        id_thresh=cfg.get("ID_THRESH", 0.1),
        area_thresh=cfg.get("AREA_THRESH", 0),
        dtype=target_dtype 
    )

    # Image Normalization Constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.fp16:
        mean = mean.half()
        std = std.half()

    frame_idx = start_frame
    frames_processed = 0
    fps_avg = 0
    start_time = time.time()

    # 8. Loop
    try:
        while cap.isOpened() and frame_idx < end_frame:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
                
            # Preprocess
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).to(device)
            if args.fp16:
                img_tensor = img_tensor.half()
            else:
                img_tensor = img_tensor.float()
            
            img_tensor = img_tensor.permute(2, 0, 1) / 255.0
            
            # Resize
            h, w = img_tensor.shape[1], img_tensor.shape[2]
            scale = 800 / min(h, w)
            if max(h, w) * scale > 1333:
                scale = 1333 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            
            img_resized = F.interpolate(img_tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            img_norm = (img_resized - mean) / std
            
            # Forward Pass
            tracker.update(img_norm)
            
            # Results
            results = tracker.get_track_results()
            valid_boxes = results['bbox']
            valid_ids = results['id']
            valid_scores = results['score']
            
            # Draw Objects
            for i in range(len(valid_scores)):
                score = valid_scores[i].item()
                obj_id = int(valid_ids[i].item())
                x, y, w_box, h_box = valid_boxes[i].cpu().float().numpy()
                
                x1, y1, x2, y2 = int(x), int(y), int(x + w_box), int(y + h_box)
                color = [int(c) for c in colors[obj_id % 1000]]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID {obj_id}"
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw Stats
            loop_time = time.time() - loop_start
            if loop_time > 0:
                fps_inst = 1.0 / loop_time
                fps_avg = 0.9 * fps_avg + 0.1 * fps_inst if frames_processed > 0 else fps_inst
            
            cv2.putText(frame, f"FPS: {int(fps_avg)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"GPU: {gpu_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            frame_label = f"Frame: {frame_idx}"
            (tw, th), _ = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(frame, frame_label, (width - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            out.write(frame)
            
            if frames_processed % 20 == 0:
                progress_pct = (frames_processed / process_duration) * 100
                sys.stdout.write(f"\r   Frame {frame_idx} (Processed {frames_processed}/{process_duration}) | {progress_pct:.1f}% | FPS: {fps_avg:.1f}   ")
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
        if os.path.exists(temp_output) and frames_processed > 0:
            try:
                convert_to_h264(temp_output, args.output_path)
                if os.path.exists(args.output_path):
                    os.remove(temp_output)
            except Exception as e:
                print(f"⚠️  Conversion failed: {e}. Output remains at {temp_output}")

if __name__ == '__main__':
    main()