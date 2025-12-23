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
        self.c_cyan = (255, 255, 0)      # High vis text
        self.c_orange = (0, 140, 255)    # Revival background (Deep Orange)
        self.c_red = (0, 0, 255)         # Arrow color

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
        """Draws status bar. Returns NEW frame."""
        H, W = frame.shape[:2]
        
        # 1. Dashboard Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 40), (20, 20, 20), -1)
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 2. Stats
        fps_text = f"FPS: {int(self.fps_avg)}"
        cv2.putText(frame, fps_text, (15, 28), self.font, 0.7, self.c_cyan, 2)
        cv2.putText(frame, f"| {gpu_name}", (140, 28), self.font, 0.6, self.c_white, 1)

        # Center: Memory Stats
        gal_size = memory_stats.get('gallery_size', 0)
        active_overrides = memory_stats.get('total_revivals', 0)
        
        mem_text = f"LTM Gallery: {gal_size} Objects"
        rev_text = f"Active Overrides: {active_overrides}"
        
        cv2.putText(frame, mem_text, (W//2 - 200, 28), self.font, 0.6, self.c_white, 1)
        
        # Highlight "Overrides" if active
        rev_color = self.c_orange if active_overrides > 0 else (150, 150, 150)
        cv2.putText(frame, rev_text, (W//2 + 50, 28), self.font, 0.6, rev_color, 2)

        # Right: Frame Count
        fr_text = f"Frame: {frame_idx}"
        (fw, _), _ = cv2.getTextSize(fr_text, self.font, 0.7, 2)
        cv2.putText(frame, fr_text, (W - fw - 20, 28), self.font, 0.7, self.c_white, 2)
        
        return frame

    def draw_tracks(self, frame, boxes, final_ids, original_ids=None):
        """
        Draws boxes with explicit arrows for Revivals.
        """
        if original_ids is None: original_ids = final_ids

        for i, (box, obj_id) in enumerate(zip(boxes, final_ids)):
            x, y, w, h = [int(v) for v in box]
            orig_id = original_ids[i]
            
            color = self.colors[obj_id % 1000]
            
            # --- 1. Draw The Bounding Box ---
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # --- 2. Check for REVIVAL (Long-Term Memory Intervention) ---
            if obj_id != orig_id:
                # Text for the annotation
                main_label = f"ID {obj_id}"
                sub_label = f"Was #{orig_id}"
                
                # --- A. Draw the "Revived" Indicator (Floating above) ---
                # Position bubble higher up to make room for arrow
                bubble_x = x
                bubble_y = y - 60 
                
                (mw, mh), _ = cv2.getTextSize(main_label, self.font, 0.8, 2)
                (sw, sh), _ = cv2.getTextSize(sub_label, self.font, 0.5, 1)
                box_w = max(mw, sw) + 20
                
                # Draw Orange Bubble Background
                cv2.rectangle(frame, (bubble_x, bubble_y - mh - 10), (bubble_x + box_w, bubble_y + sh + 10), self.c_orange, -1)
                
                # Draw Text
                cv2.putText(frame, main_label, (bubble_x + 10, bubble_y), self.font, 0.8, self.c_white, 2)
                cv2.putText(frame, sub_label, (bubble_x + 10, bubble_y + sh + 5), self.font, 0.5, (50, 50, 50), 1)
                
                # --- B. Draw ARROW pointing to the box ---
                # Start: Bottom of bubble
                # End: Top-Left corner of bounding box
                start_pt = (bubble_x + 20, bubble_y + sh + 10)
                end_pt = (x, y)
                
                # Draw arrow
                cv2.arrowedLine(frame, start_pt, end_pt, self.c_red, 3, tipLength=0.3)
                
                # Optional: Highlight the box edges again in Red to show it's being modified
                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), self.c_orange, 1)

            else:
                # --- 3. Normal Label ---
                label = f"ID {obj_id}"
                (tw, th), _ = cv2.getTextSize(label, self.font, 0.6, 2)
                
                # Draw Label Background (Track Color)
                cv2.rectangle(frame, (x, y - 25), (x + tw + 10, y), color, -1)
                # Draw Text
                cv2.putText(frame, label, (x + 5, y - 8), self.font, 0.6, self.c_white, 2)

        return frame