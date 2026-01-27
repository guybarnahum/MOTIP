import argparse
import cv2
import sys
import os
import time
import numpy as np

def get_args():
    parser = argparse.ArgumentParser("OpenCV Video Comparison")
    
    parser.add_argument('video1', type=str, help="Path to the first video (Left)")
    parser.add_argument('video2', type=str, help="Path to the second video (Right)")
    parser.add_argument('--output_path', type=str, default="comparison_output.mp4", help="Output filename")
    
    # Scaling
    parser.add_argument('-s', '--scale', action='store_true', help="Scale videos down by 50%%")
    
    # Frame Range Arguments
    parser.add_argument('--start_frame', type=int, default=0, help="Frame to start processing")
    parser.add_argument('--end_frame', type=int, default=None, help="Frame to stop processing")
    
    return parser.parse_args()

def main():
    args = get_args()

    # 1. Validation
    if not os.path.exists(args.video1):
        print(f"❌ Error: File not found: {args.video1}")
        return
    if not os.path.exists(args.video2):
        print(f"❌ Error: File not found: {args.video2}")
        return

    # 2. Open Video Captures
    cap1 = cv2.VideoCapture(args.video1)
    cap2 = cv2.VideoCapture(args.video2)

    if not cap1.isOpened() or not cap2.isOpened():
        print("❌ Error opening one of the videos.")
        return

    # 3. Get Video Properties (Assume they match based on your prompt)
    W = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap1.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))

    # Handle Frame Range
    start_frame = max(0, args.start_frame)
    if args.end_frame is None or args.end_frame > total_frames:
        end_frame = total_frames
    else:
        end_frame = args.end_frame
        
    process_duration = end_frame - start_frame
    if process_duration <= 0:
        print("❌ Error: end_frame must be greater than start_frame")
        return

    print(f"🎬 Video 1: {args.video1}")
    print(f"🎬 Video 2: {args.video2}")
    print(f"📏 Input Dim: {W}x{H} @ {fps:.2f}fps")
    print(f"⏱️  Processing Range: Frame {start_frame} to {end_frame} (Total: {process_duration} frames)")

    # 4. Set Output Dimensions
    if args.scale:
        print("-> Scaling videos down by factor of 2.")
        # Resize each to W/2, H/2. 
        # Final width is (W/2 + W/2) = W. Final Height is H/2.
        target_w = W // 2
        target_h = H // 2
        out_size = (W, target_h)
    else:
        # Full resolution side-by-side
        target_w = W
        target_h = H
        out_size = (W * 2, H)

    # 5. Seek to Start Frame
    cap1.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # 6. Initialize Writer
    # 'mp4v' is the standard OpenCV MP4 codec. 
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, out_size)
    
    frame_idx = start_frame
    frames_processed = 0
    start_time = time.time()

    try:
        while frame_idx < end_frame:
            loop_start = time.time()
            
            # Read frames
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2:
                print("\n⚠️ End of stream reached early.")
                break

            # Resize if necessary
            if args.scale:
                frame1 = cv2.resize(frame1, (target_w, target_h))
                frame2 = cv2.resize(frame2, (target_w, target_h))

            # Stack Horizontally
            # numpy.hstack is fast and efficient for this
            combined = np.hstack((frame1, frame2))

            # Optional: Add Frame Counter Text
            # cv2.putText(combined, f"Frame: {frame_idx}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            out.write(combined)

            # Progress Bar (Matches your reference style)
            if frames_processed % 10 == 0:
                loop_dur = time.time() - loop_start
                current_fps = 1.0 / (loop_dur + 1e-6)
                progress_pct = (frames_processed / process_duration) * 100
                sys.stdout.write(f"\r   Frame {frame_idx} (Processed {frames_processed}/{process_duration}) | {progress_pct:.1f}% | Speed: {current_fps:.1f} fps   ")
                sys.stdout.flush()

            frame_idx += 1
            frames_processed += 1

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user! Saving video so far...")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
    finally:
        cap1.release()
        cap2.release()
        out.release()
        total_time = time.time() - start_time
        print(f"\n✅ Finished processing {frames_processed} frames in {total_time:.1f}s.")
        print(f"💾 Output saved to: {args.output_path}")

if __name__ == '__main__':
    main()
