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
        """Draws status bar at the BOTTOM. Returns NEW frame."""
        H, W = frame.shape[:2]
        bar_h = 40
        y_start = H - bar_h  # Start of the bar (e.g. 1040 for a 1080p video)
        text_y = H - 12      # Baseline for text
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (W, H), (20, 20, 20), -1)
        alpha = 0.8
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # FPS & GPU
        cv2.putText(frame, f"FPS: {int(self.fps_avg)}", (15, text_y), self.font, 0.7, self.c_cyan, 2)
        cv2.putText(frame, f"| {gpu_name}", (140, text_y), self.font, 0.6, self.c_white, 1)

        # Stats
        gal_size = memory_stats.get('gallery_size', 0)
        active_overrides = memory_stats.get('active_overrides', 0)
        
        mem_text = f"LTM Gallery: {gal_size}"
        rev_text = f"Overrides: {active_overrides}"
        
        cv2.putText(frame, mem_text, (W//2 - 150, text_y), self.font, 0.6, self.c_white, 1)
        
        # Highlight Overrides count only if non-zero
        rev_color = self.c_orange if active_overrides > 0 else (150, 150, 150)
        cv2.putText(frame, rev_text, (W//2 + 50, text_y), self.font, 0.6, rev_color, 2)

        # Frame Count
        fr_text = f"Frame: {frame_idx}"
        (fw, _), _ = cv2.getTextSize(fr_text, self.font, 0.7, 2)
        cv2.putText(frame, fr_text, (W - fw - 20, text_y), self.font, 0.7, self.c_white, 2)
        
        return frame

    def draw_tracks(self, frame, boxes, final_ids, categories, original_ids=None):
        """
        Compact drawing: 'ID 5' or 'ID 5 > 50'.
        Now supports category-based color palettes and labels.
        categories: list of int (1 for Person, 2 for Vehicle)
        """
        if original_ids is None: original_ids = final_ids

        # Mapping for display text
        cat_names = {1: "Person", 2: "Vehicle"}

        for i, (box, obj_id) in enumerate(zip(boxes, final_ids)):
            x, y, w, h = [int(v) for v in box]
            orig_id = original_ids[i]
            cat_id = categories[i]
            
            # --- CATEGORY-AWARE COLOR LOGIC ---
            # Strategy: Use different starting offsets in the color palette 
            # to ensure People and Vehicles look distinctly different.
            if cat_id == 1: # Person
                color = self.colors[(obj_id + 100) % 1000] 
            else: # Vehicle (Car, etc.)
                color = self.colors[(obj_id + 600) % 1000]

            # 1. Main Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 2. Text Logic
            cat_prefix = cat_names.get(cat_id, "OBJ")
            if obj_id != orig_id:
                # Revival Case: Compact Label "Person ID 5 > 50"
                label = f"{cat_prefix} {obj_id} > {orig_id}"
            else:
                # Normal Case
                label = f"{cat_prefix} {obj_id}"
            
            # 3. Draw Label
            (tw, th), _ = cv2.getTextSize(label, self.font, 0.5, 2) # Slightly smaller font for density
            
            # Clamp label position so it stays on screen
            lbl_y = y - 10
            if lbl_y < 20: 
                lbl_y = y + 25

            # Background Rectangle (Solid color based on class)
            cv2.rectangle(frame, (x, lbl_y - th - 5), (x + tw + 10, lbl_y + 5), color, -1)
            
            # Text (White for better contrast on solid background)
            cv2.putText(frame, label, (x + 5, lbl_y), self.font, 0.5, self.c_white, 2)

        return frame