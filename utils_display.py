import cv2
import numpy as np

class Annotator:
    def __init__(self, fps_smoothing=0.9):
        self.fps_avg = 0
        self.fps_smoothing = fps_smoothing
        self.colors = self._generate_colors()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Dashboard settings
        self.dash_h = 120
        self.dash_color = (40, 40, 40) # Dark gray background

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
        memory_stats: dict with keys like 'active_tracks', 'gallery_size', 'revivals'
        """
        H, W = frame.shape[:2]
        
        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (W, 40), (0, 0, 0), -1)
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # Left: FPS & GPU
        fps_text = f"FPS: {int(self.fps_avg)}"
        cv2.putText(frame, fps_text, (10, 28), self.font, 0.7, (0, 255, 0), 2)
        
        gpu_text = f"GPU: {gpu_name}"
        cv2.putText(frame, gpu_text, (130, 28), self.font, 0.6, (200, 200, 200), 1)

        # Center: Memory Stats
        mem_text = f"Mem Bank: {memory_stats.get('gallery_size', 0)} IDs | Revived: {memory_stats.get('total_revivals', 0)}"
        (mw, _), _ = cv2.getTextSize(mem_text, self.font, 0.6, 1)
        cv2.putText(frame, mem_text, (W//2 - mw//2, 28), self.font, 0.6, (100, 255, 255), 1)

        # Right: Frame Count
        fr_text = f"Frame: {frame_idx}"
        (fw, _), _ = cv2.getTextSize(fr_text, self.font, 0.7, 2)
        cv2.putText(frame, fr_text, (W - fw - 10, 28), self.font, 0.7, (0, 255, 255), 2)
        
        return frame

    def draw_tracks(self, frame, boxes, final_ids, original_ids=None):
        """
        boxes: [N, 4] (x, y, w, h)
        final_ids: [N] (The ID after memory override)
        original_ids: [N] (The ID the tracker originally assigned, optional)
        """
        if original_ids is None:
            original_ids = final_ids

        for i, (box, obj_id) in enumerate(zip(boxes, final_ids)):
            x, y, w, h = [int(v) for v in box]
            orig_id = original_ids[i]
            
            color = self.colors[obj_id % 1000]
            
            # 1. Main Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # 2. Label Background
            label = f"ID {obj_id}"
            (tw, th), _ = cv2.getTextSize(label, self.font, 0.6, 2)
            
            # If this was a MEMORY OVERRIDE (Revival)
            if obj_id != orig_id:
                # Draw a special "Revival" tag
                revival_msg = f"Was {orig_id}"
                # Yellow background for emphasis
                cv2.rectangle(frame, (x, y - 40), (x + tw + 10, y), (0, 255, 255), -1) 
                cv2.putText(frame, "REVIVED", (x, y - 25), self.font, 0.5, (0, 0, 0), 1)
                cv2.putText(frame, label, (x, y - 5), self.font, 0.6, (0, 0, 0), 2)
                
                # Draw a "Flash" or border effect (optional)
                cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), (0, 255, 255), 1)
            else:
                # Standard Label
                cv2.rectangle(frame, (x, y - 20), (x + tw, y), color, -1)
                cv2.putText(frame, label, (x, y - 5), self.font, 0.6, (255, 255, 255), 2)

        return frame
