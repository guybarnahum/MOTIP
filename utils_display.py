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
        self.c_cyan = (255, 255, 0)    # High vis text
        self.c_orange = (0, 165, 255)  # Revival background
        self.c_red = (0, 0, 255)       # Alert background

    def _generate_colors(self, num=1000):
        np.random.seed(42)
        # Generate bright, distinct colors
        colors = np.random.randint(0, 255, size=(num, 3), dtype="uint8")
        return [tuple(int(c) for c in color) for color in colors]

    def update_fps(self, loop_time):
        if loop_time > 0:
            fps_inst = 1.0 / loop_time
            if self.fps_avg == 0:
                self.fps_avg = fps_inst
            else:
                self.fps_avg = (self.fps_smoothing * self.fps_avg) + ((1 - self.fps_smoothing) * fps_inst)

    def draw_dashboard(self, frame, frame_idx, gpu_name, memory_stats):
        """
        Draws a status bar at the top of the frame.
        IMPORTANT: This returns a NEW frame, so you must capture it.
        """
        H, W = frame.shape[:2]
        
        # 1. Create Dashboard Overlay
        overlay = frame.copy()
        # Draw dark background bar (top 40 pixels)
        cv2.rectangle(overlay, (0, 0), (W, 40), (20, 20, 20), -1)
        
        # Blend overlay (0.7 opacity for background)
        alpha = 0.7
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 2. Draw Stats (White/Cyan text)
        # Left: FPS
        fps_text = f"FPS: {int(self.fps_avg)}"
        cv2.putText(frame, fps_text, (15, 28), self.font, 0.7, self.c_cyan, 2)
        
        # Left-Center: GPU
        cv2.putText(frame, f"| {gpu_name}", (140, 28), self.font, 0.6, self.c_white, 1)

        # Center: Memory Stats
        gal_size = memory_stats.get('gallery_size', 0)
        revivals = memory_stats.get('total_revivals', 0)
        
        mem_text = f"Memory Bank: {gal_size} IDs"
        rev_text = f"Revivals: {revivals}"
        
        # Draw Memory stats
        cv2.putText(frame, mem_text, (W//2 - 150, 28), self.font, 0.6, self.c_white, 1)
        # Highlight Revivals in Orange if > 0
        rev_color = self.c_orange if revivals > 0 else self.c_white
        cv2.putText(frame, rev_text, (W//2 + 80, 28), self.font, 0.6, rev_color, 2)

        # Right: Frame Count
        fr_text = f"Frame: {frame_idx}"
        (fw, _), _ = cv2.getTextSize(fr_text, self.font, 0.7, 2)
        cv2.putText(frame, fr_text, (W - fw - 20, 28), self.font, 0.7, self.c_white, 2)
        
        return frame

    def draw_tracks(self, frame, boxes, final_ids, original_ids=None):
        """
        Draws bounding boxes and ID labels.
        Modifies 'frame' in-place.
        """
        if original_ids is None:
            original_ids = final_ids

        for i, (box, obj_id) in enumerate(zip(boxes, final_ids)):
            x, y, w, h = [int(v) for v in box]
            orig_id = original_ids[i]
            
            color = self.colors[obj_id % 1000]
            
            # 1. Main Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 2. Prepare Label
            label = f"ID {obj_id}"
            (tw, th), _ = cv2.getTextSize(label, self.font, 0.6, 2)
            
            # --- REVIVAL LOGIC ---
            if obj_id != orig_id:
                # This is a Revival! Draw distinctive tag.
                rev_msg = "REVIVED"
                (rw, rh), _ = cv2.getTextSize(rev_msg, self.font, 0.5, 1)
                
                # Draw "Revived" tag on top
                # Background: Orange
                cv2.rectangle(frame, (x, y - 45), (x + max(tw, rw) + 10, y), self.c_orange, -1)
                
                # Text: White
                cv2.putText(frame, rev_msg, (x + 5, y - 30), self.font, 0.5, self.c_white, 1)
                cv2.putText(frame, f"ID {obj_id}", (x + 5, y - 8), self.font, 0.6, self.c_white, 2)
                
                # Draw connecting line to show it changed
                # (Optional visual cue: small circle at top-left corner)
                cv2.circle(frame, (x, y), 4, self.c_orange, -1)
                
            else:
                # Standard ID Tag
                # Background: Track Color
                cv2.rectangle(frame, (x, y - 25), (x + tw + 10, y), color, -1)
                # Text: White (better contrast on colored backgrounds)
                cv2.putText(frame, label, (x + 5, y - 8), self.font, 0.6, self.c_white, 2)

        return frame