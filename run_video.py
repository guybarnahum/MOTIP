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
from utils_inference import convert_to_h264
from utils_display import Annotator
from models.motip import build as build_model
from models.runtime_tracker import RuntimeTracker
from models.longterm_memory import LongTermMemory

warnings.filterwarnings("ignore")
sys.path.append(os.getcwd())

def get_args():
    parser = argparse.ArgumentParser("MOTIP + Memory")
    
    # Config is now OPTIONAL (default None)
    parser.add_argument('--config_path', type=str, default=None, help="Required only for Dev models")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to .pth model (Deploy or Dev)")
    parser.add_argument('--video_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, default="output_final.mp4")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--score_thresh', type=float, default=0.5)
    
    # Frame Range Arguments
    parser.add_argument('--start_frame', type=int, default=0, help="Frame to start processing")
    parser.add_argument('--end_frame', type=int, default=None, help="Frame to stop processing")
    
    # Memory Settings
    parser.add_argument('--miss_tolerance', type=int, default=30, help="Short term tracker memory")
    parser.add_argument('--longterm_patience', type=int, default=9000, help="Long term gallery memory")
    parser.add_argument('--no_memory', action='store_true', help="Disable LongTerm Memory ReID")
    
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

    # 2. LOAD CHECKPOINT FIRST (To determine mode)
    print(f"📥 Loading weights: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return
        
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    # --- HYBRID LOADING LOGIC ---
    if 'model_args' in ckpt:
        # ==========================================
        # PATH A: DEPLOYMENT MODEL (Config-Free)
        # ==========================================
        print("🚀 Detected Deployment Model (Config-Free Inference)")
        model_args = ckpt['model_args']
        state_dict = ckpt['model_state_dict']
        
        # Now we only pass what we actually know/need.
        # The updated __init__.py handles the defaults.
        deploy_cfg = {
            'NUM_CLASSES': model_args.get('num_classes', 2),
            'HIDDEN_DIM': model_args.get('hidden_dim', 256),
            'NHEADS': model_args.get('nheads', 8),
            'ENC_LAYERS': model_args.get('num_encoder_layers', 6),
            'DEC_LAYERS': model_args.get('num_decoder_layers', 6),
            'DIM_FEEDFORWARD': model_args.get('dim_feedforward', 1024),
            'NUM_QUERIES': model_args.get('num_queries', 300),
            'AUX_LOSS': False,
            'WITH_BOX_REFINE': model_args.get('with_box_refine', True),
            'TWO_STAGE': model_args.get('two_stage', False),
            'DEVICE': args.device,
            'DROPOUT': 0.0,
            
            # Explicitly map keys that might have different names in export vs init
            'DETR_NUM_FEATURE_LEVELS': model_args.get('num_feature_levels', 4),
            'DETR_DEC_N_POINTS': model_args.get('dec_n_points', 4),
            'DETR_ENC_N_POINTS': model_args.get('enc_n_points', 4),
        }
        
        model = build_model(deploy_cfg)
        model = model[0] if isinstance(model, tuple) else model
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model built successfully from embedded args.")

    else:
        # ==========================================
        # PATH B: DEVELOPMENT MODEL (Requires Yaml)
        # ==========================================
        print("🛠️  Detected Development Model (Requires Config)")
        
        if args.config_path is None:
            print("⚠️  No config provided. Trying default './configs/r50_motip_bdd100k.yaml'")
            args.config_path = './configs/r50_motip_bdd100k.yaml'
            
        if not os.path.exists(args.config_path):
            print(f"❌ Error: Config not found at {args.config_path}")
            return

        with open(args.config_path, 'r') as f:
            cfg = yaml.safe_load(f)

        if "SUPER_CONFIG_PATH" in cfg:
            print(f"🔗 Inheriting config from: {cfg['SUPER_CONFIG_PATH']}")
            super_path = cfg["SUPER_CONFIG_PATH"]
            if not os.path.exists(super_path):
                config_dir = os.path.dirname(args.config_path)
                super_path = os.path.join(config_dir, cfg["SUPER_CONFIG_PATH"])
            
            if os.path.exists(super_path):
                with open(super_path, 'r') as f_base:
                    base_cfg = yaml.safe_load(f_base)
                base_cfg.update(cfg) 
                cfg = base_cfg

        if 'DEVICE' not in cfg: cfg['DEVICE'] = args.device
        
        print("🏗️  Building model from YAML...")
        model = build_model(cfg)
        model = model[0] if isinstance(model, tuple) else model
        model.load_state_dict(ckpt.get('model', ckpt), strict=False)

    # --- END HYBRID LOGIC ---

    model.to(device)
    model.eval()
    
    # 3. Setup Modules
    if args.no_memory:
        memory = None
        print("🧠 LongTerm Memory: DISABLED")
    else:
        memory = LongTermMemory(patience=args.longterm_patience)
        print("🧠 LongTerm Memory: ENABLED")
        
    annotator = Annotator()
    
    # 4. Video IO Setup
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video: {args.video_path}")
        return

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    if args.fp16: mean, std = mean.half(), std.half()

    frame_idx = start_frame
    frames_processed = 0
    start_time = time.time()
    
    try:
        while cap.isOpened() and frame_idx < end_frame:
            loop_start = time.time() 
            ret, frame = cap.read()
            if not ret: break

            # Preprocess
            t_img = torch.from_numpy(frame).to(device).permute(2, 0, 1).float() / 255.0
            if args.fp16: t_img = t_img.half()
            
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
            valid_boxes = res['bbox'].cpu().float().numpy() 
            valid_ids = res['id'].tolist()                  
            
            active_embeds = res.get('embeddings', None)

            # --- C. MEMORY UPDATE ---
            final_ids = []
            if memory is not None and active_embeds is not None and len(valid_ids) > 0:
                id_map = memory.update(frame_idx, valid_ids, active_embeds)
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
            
            if memory is not None:
                mem_stats = {
                    "gallery_size": len(memory.storage),
                    "active_overrides": current_overrides_count 
                }
            else:
                mem_stats = {
                    "gallery_size": 0,
                    "active_overrides": 0 
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