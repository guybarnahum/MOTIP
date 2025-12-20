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
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--fp16', action='store_true', default=True, help="Use Float16 precision for speed")
    
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
    # -metadata timecode="00:00:00:00": Embeds a timecode track starting at 0
    # -g 10: Keyframe every 10 frames for smooth scrubbing
    cmd = [
        'ffmpeg', '-y', 
        '-i', input_path, 
        '-c:v', 'libx264', 
        '-preset', 'fast', 
        '-crf', '23', 
        '-g', '10',
        '-pix_fmt', 'yuv420p',
        '-metadata', 'timecode=00:00:00:00',  # <--- Embeds Timecode Track
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
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"🎬 Processing: {args.video_path} ({width}x{height} @ {fps:.2f}fps)")
    
    # 6. Setup Video Output (Temporary file)
    temp_output = "temp_" + os.path.basename(args.output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
    
    colors = generate_colors()

    # 7. Initialize RuntimeTracker
    target_dtype = torch.float16 if args.fp16 else torch.float32
    print(f"📊 Precision: {'FP16 (Half)' if args.fp16 else 'FP32 (Full)'}")

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
        dtype=target_dtype 
    )

    # Image Normalization Constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.fp16:
        mean = mean.half()
        std = std.half()

    frame_idx = 0
    fps_avg = 0
    start_time = time.time()

    # 8. Loop
    try:
        while cap.isOpened():
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

            # Draw Stats (FPS & Frame Number)
            loop_time = time.time() - loop_start
            if loop_time > 0:
                fps_inst = 1.0 / loop_time
                fps_avg = 0.9 * fps_avg + 0.1 * fps_inst if frame_idx > 0 else fps_inst
            
            # Top-Left: FPS
            cv2.putText(frame, f"FPS: {int(fps_avg)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"GPU: {gpu_name}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            # Top-Right: Frame Counter (Visual Burn-in)
            frame_label = f"Frame: {frame_idx}"
            (tw, th), _ = cv2.getTextSize(frame_label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            cv2.putText(frame, frame_label, (width - tw - 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            out.write(frame)
            
            if frame_idx % 20 == 0:
                print(f"   Frame {frame_idx}/{total_frames} | FPS: {fps_avg:.1f}", end='\r')
            
            frame_idx += 1

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
        print(f"\n✅ Finished processing {frame_idx} frames.")
        
        # Convert to H.264
        if os.path.exists(temp_output) and frame_idx > 0:
            try:
                convert_to_h264(temp_output, args.output_path)
                if os.path.exists(args.output_path):
                    os.remove(temp_output)
            except Exception as e:
                print(f"⚠️  Conversion failed: {e}. Output remains at {temp_output}")

if __name__ == '__main__':
    main()