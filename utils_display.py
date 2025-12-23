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
        self.c_orange = (0, 140, 255) 

    def _generate_colors(self, num=1000):
        np.random.seed(42)
        # Generate bright, distinct colors
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
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 40), (20, 20, 20), -1)
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # FPS & GPU
        cv2.putText(frame, f"FPS: {int(self.fps_avg)}", (15, 28), self.font, 0.7, self.c_cyan, 2)
        cv2.putText(frame, f"| {gpu_name}", (140, 28), self.font, 0.6, self.c_white, 1)

        # Stats
        gal_size = memory_stats.get('gallery_size', 0)
        active_overrides = memory_stats.get('active_overrides', 0) # Expecting FRAME-LEVEL count now
        
        mem_text = f"LTM Gallery: {gal_size}"
        rev_text = f"Overrides: {active_overrides}"
        
        cv2.putText(frame, mem_text, (W//2 - 150, 28), self.font, 0.6, self.c_white, 1)
        
        # Highlight Overrides count only if non-zero
        rev_color = self.c_orange if active_overrides > 0 else (150, 150, 150)
        cv2.putText(frame, rev_text, (W//2 + 50, 28), self.font, 0.6, rev_color, 2)

        # Frame Count
        fr_text = f"Frame: {frame_idx}"
        (fw, _), _ = cv2.getTextSize(fr_text, self.font, 0.7, 2)
        cv2.putText(frame, fr_text, (W - fw - 20, 28), self.font, 0.7, self.c_white, 2)
        
        return frame

    def draw_tracks(self, frame, boxes, final_ids, original_ids=None):
        """
        Compact drawing: 'ID 5' or 'ID 5 > 50'.
        Uses standard track color for box and label background.
        """
        if original_ids is None: original_ids = final_ids

        for i, (box, obj_id) in enumerate(zip(boxes, final_ids)):
            x, y, w, h = [int(v) for v in box]
            orig_id = original_ids[i]
            
            # Use normal distinct color by FINAL ID
            color = self.colors[obj_id % 1000]

            # 1. Main Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 2. Text Logic
            if obj_id != orig_id:
                # Revival Case: Compact Label "ID 5 > 50"
                label = f"ID {obj_id} > {orig_id}"
            else:
                # Normal Case
                label = f"ID {obj_id}"
            
            # 3. Draw Label
            (tw, th), _ = cv2.getTextSize(label, self.font, 0.6, 2)
            
            # Clamp label position so it stays on screen
            lbl_y = y - 10
            if lbl_y < 45: # Avoid overlapping with dashboard
                lbl_y = y + 25

            # Background Rectangle (Same as box color)
            cv2.rectangle(frame, (x, lbl_y - th - 5), (x + tw + 10, lbl_y + 5), color, -1)
            
            # Text (White)
            cv2.putText(frame, label, (x + 5, lbl_y), self.font, 0.6, self.c_white, 2)

        return frame