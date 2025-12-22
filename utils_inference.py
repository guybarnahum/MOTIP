import cv2
import torch
import numpy as np
import subprocess
import shutil
import os

def generate_colors(num_colors=1000):
    np.random.seed(42)
    return np.random.randint(0, 255, size=(num_colors, 3), dtype="uint8")

def convert_to_h264(input_path, output_path):
    """
    Converts to H.264 with Timecode Metadata (Restored from original).
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
        '-metadata', 'timecode=00:00:00:00', # <--- RESTORED THIS
        output_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"✅ H.264 conversion complete: {output_path}")

def recover_embeddings(tracker, valid_boxes, img_w, img_h, device):
    """
    Matches 'valid' filtered boxes back to the 'raw' model queries 
    to recover the feature embeddings (hs[-1]).
    """
    if not hasattr(tracker, 'output') or 'outputs' not in tracker.output:
        return None

    raw_outputs = tracker.output['outputs']       # [1, 300, 256]
    raw_boxes_norm = tracker.output['pred_boxes'] # [1, 300, 4]
    
    if len(valid_boxes) == 0:
        return None

    # 1. Un-normalize raw boxes to pixels
    scale = torch.tensor([img_w, img_h, img_w, img_h], device=device)
    raw_boxes_px = box_cxcywh_to_xyxy(raw_boxes_norm[0] * scale)
    
    # 2. Convert valid boxes (x,y,w,h) -> (x1,y1,x2,y2)
    valid_boxes_xyxy = valid_boxes.clone()
    valid_boxes_xyxy[:, 2] += valid_boxes_xyxy[:, 0]
    valid_boxes_xyxy[:, 3] += valid_boxes_xyxy[:, 1]
    
    # 3. IoU Match
    iou_matrix = box_iou(valid_boxes_xyxy, raw_boxes_px)
    
    # 4. Select best match
    _, indices = iou_matrix.max(dim=1)
    
    return raw_outputs[0][indices]

# --- Math Helpers ---
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union