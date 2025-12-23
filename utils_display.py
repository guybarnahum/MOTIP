import cv2
import numpy as np

class Annotator:
    def __init__(self, fps_smoothing=0.9):
        self.fps_avg = 0
        self.fps_smoothing = fps_smoothing
        self.colors = self._generate_colors()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # UI Colors (BGR)
        self.c_white = (255, 255, 255)
        self.c_black = (0, 0, 0)
        self.c_cyan = (255, 255, 0)
        self.c_orange = (0, 140, 255) # Deep Orange
        self.c_red = (0, 0, 255)

    def _generate_colors(self, num=1000):
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(num, 3), dtype="uint8")
        return [tuple(int(c) for c in color) for color in colors]

    def update_fps(self, loop_time):
        if loop_time > 0:
            fps_inst = 1.0 / loop_time
            if self.fps_avg == 0: self.fps_avg = fps_inst
            else: self.fps_avg = (self.fps_smoothing * self.fps_avg) + ((1 - self.fps_smoothing) * fps_inst)

    def draw_dashboard(self, frame, frame_idx, gpu_name, memory_stats):
        H, W = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 40), (20, 20, 20), -1)
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        cv2.putText(frame, f"FPS: {int(self.fps_avg)}", (15, 28), self.font, 0.7, self.c_cyan, 2)
        cv2.putText(frame, f"| {gpu_name}", (140, 28), self.font, 0.6, self.c_white, 1)

        # Center Stats
        gal_size = memory_stats.get('gallery_size', 0)
        active_overrides = memory_stats.get('total_revivals', 0)
        
        mem_text = f"LTM: {gal_size} stored"
        rev_text = f"Overrides: {active_overrides}"
        
        cv2.putText(frame, mem_text, (W//2 - 180, 28), self.font, 0.6, self.c_white, 1)
        
        rev_color = self.c_orange if active_overrides > 0 else (150, 150, 150)
        cv2.putText(frame, rev_text, (W//2 + 20, 28), self.font, 0.6, rev_color, 2)
        
        # Frame Count
        fr_text = f"Frame: {frame_idx}"
        (fw, _), _ = cv2.getTextSize(fr_text, self.font, 0.7, 2)
        cv2.putText(frame, fr_text, (W - fw - 20, 28), self.font, 0.7, self.c_white, 2)
        
        return frame

    def draw_tracks(self, frame, boxes, final_ids, original_ids=None):
        H, W = frame.shape[:2]
        if original_ids is None: original_ids = final_ids

        for i, (box, obj_id) in enumerate(zip(boxes, final_ids)):
            x, y, w, h = [int(v) for v in box]
            orig_id = original_ids[i]
            color = self.colors[obj_id % 1000]

            # 1. Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 2. Annotation
            if obj_id != orig_id:
                # --- REVIVAL CASE ---
                
                # Setup Text
                main_txt = f"ID {obj_id}"
                sub_txt = f"Was {orig_id}"
                (mw, mh), _ = cv2.getTextSize(main_txt, self.font, 0.8, 2)
                (sw, sh), _ = cv2.getTextSize(sub_txt, self.font, 0.6, 1)
                
                # Calculate Bubble Position (Try Top, Flip to Bottom if out of bounds)
                bubble_h = mh + sh + 20
                bubble_y_start = y - bubble_h - 20
                
                # Arrow Logic
                arrow_start = (x + 20, y - 20)
                arrow_end = (x, y)
                
                # Flip logic if too close to top
                if bubble_y_start < 40: # 40 is dashboard height
                    bubble_y_start = y + h + 10
                    arrow_start = (x + 20, y + h + 10)
                    arrow_end = (x, y + h)
                
                # Draw Bubble
                cv2.rectangle(frame, (x, bubble_y_start), (x + max(mw, sw) + 20, bubble_y_start + bubble_h), self.c_orange, -1)
                
                # Draw Text
                cv2.putText(frame, main_txt, (x + 10, bubble_y_start + mh + 5), self.font, 0.8, self.c_white, 2)
                cv2.putText(frame, sub_txt, (x + 10, bubble_y_start + mh + sh + 15), self.font, 0.6, self.c_black, 1)
                
                # Draw Arrow
                cv2.arrowedLine(frame, arrow_start, arrow_end, self.c_red, 3, tipLength=0.3)
                
                # Highlight Box
                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), self.c_orange, 2)

            else:
                # --- NORMAL CASE ---
                label = f"ID {obj_id}"
                (tw, th), _ = cv2.getTextSize(label, self.font, 0.6, 2)
                
                # Ensure label doesn't go off top
                lbl_y = y - 10
                if lbl_y < 50: lbl_y = y + 20
                    
                cv2.rectangle(frame, (x, lbl_y - th - 5), (x + tw + 10, lbl_y + 5), color, -1)
                cv2.putText(frame, label, (x + 5, lbl_y), self.font, 0.6, self.c_white, 2)