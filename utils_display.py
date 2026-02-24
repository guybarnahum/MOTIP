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
        """Draws multi-class status bar at the BOTTOM. Returns NEW frame."""
        H, W = frame.shape[:2]
        bar_h = 45 # Slightly taller for better readability
        y_start = H - bar_h  
        text_y = H - 15      
        
        # --- DEBUG TIER (Internal State Telemetry) ---
        # Draw a smaller bar above the main one for pointer tracking
        debug_h = 30
        debug_y = y_start - debug_h
        cv2.rectangle(frame, (0, debug_y), (W, y_start), (10, 10, 10), -1)
        
        ptr_p = memory_stats.get('ptr_p', -1)
        ptr_v = memory_stats.get('ptr_v', -1)
        nb_count = memory_stats.get('newborns', -1)
        
        # Pointers show the raw "index" before modulo is applied
        debug_text = f"DEBUG >> P-PTR: {ptr_p} | V-PTR: {ptr_v} | NEWBORNS: {nb_count}"
        cv2.putText(frame, debug_text, (15, debug_y + 20), self.font, 0.5, (0, 200, 255), 1)
        
        # 1. Semi-transparent background bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, y_start), (W, H), (15, 15, 15), -1)
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 2. System Info (FPS & GPU)
        cv2.putText(frame, f"FPS: {int(self.fps_avg)}", (15, text_y), self.font, 0.6, self.c_cyan, 2)
        cv2.putText(frame, f"| {gpu_name}", (120, text_y), self.font, 0.5, self.c_white, 1)

        # 3. Multi-Class Object Counts
        # We assume memory_stats now contains 'person_count' and 'vehicle_count'
        p_count = memory_stats.get('person_count' , -1)
        v_count = memory_stats.get('vehicle_count', -1)
        
        # Draw People Count (Greenish)
        p_text = f"People: {p_count}"
        cv2.putText(frame, p_text, (W//4, text_y), self.font, 0.6, (0, 255, 150), 2)
        
        # Draw Vehicle Count (Blueish)
        v_text = f"Vehicles: {v_count}"
        cv2.putText(frame, v_text, (W//4 + 180, text_y), self.font, 0.6, (255, 150, 0), 2)

        # 4. LTM Stats
        gal_size = memory_stats.get('gallery_size', -1)
        active_overrides = memory_stats.get('active_overrides', -1)
        
        mem_text = f"LTM: {gal_size}"
        cv2.putText(frame, mem_text, (W//2 + 150, text_y), self.font, 0.5, self.c_white, 1)
        
        # Highlight Overrides
        rev_color = self.c_orange if active_overrides > 0 else (100, 100, 100)
        cv2.putText(frame, f"Ovr: {active_overrides}", (W//2 + 300, text_y), self.font, 0.5, rev_color, 2)

        # 5. Frame Progress
        fr_text = f"FR: {frame_idx}"
        (fw, _), _ = cv2.getTextSize(fr_text, self.font, 0.6, 2)
        cv2.putText(frame, fr_text, (W - fw - 20, text_y), self.font, 0.6, self.c_white, 2)
        
        return frame


    def draw_tracks(self, frame, boxes, final_ids, categories, original_ids=None):
        """
        Compact drawing: 'Person 5' or 'Vehicle 505'.
        Aligned with RuntimeTracker: 0 for Person, 1 for Vehicle.
        """
        if original_ids is None: original_ids = final_ids

        # --- FIX: Match RuntimeTracker categories ---
        # Tracker uses 0 for Person, 1 for Vehicle
        cat_names = {0: "Person", 1: "Vehicle"}

        for i, (box, obj_id) in enumerate(zip(boxes, final_ids)):
            x, y, w, h = [int(v) for v in box]
            orig_id = original_ids[i]
            cat_id = categories[i]
            
            # --- CATEGORY-AWARE COLOR LOGIC ---
            # Using specific zones in the color palette:
            # People: Lower half of palette (offset by tracker ID)
            # Vehicles: Upper half of palette (offset by tracker ID)
            if cat_id == 0: # Person
                color = self.colors[obj_id % 500] 
            else: # Vehicle
                color = self.colors[(obj_id % 500) + 500]

            # 1. Main Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 2. Text Logic
            cat_prefix = cat_names.get(cat_id, "OBJ")
            if obj_id != orig_id:
                # Revival Case: "Person 5 > 50" (LongTerm Memory active)
                label = f"{cat_prefix} {obj_id}>{orig_id}"
            else:
                # Normal Case: "Person 5"
                label = f"{cat_prefix} {obj_id}"
            
            # 3. Draw Label
            (tw, th), _ = cv2.getTextSize(label, self.font, 0.45, 1) # Leaner font for higher density
            
            # Clamp label position so it stays inside the frame
            lbl_y = y - 5
            if lbl_y < 20: 
                lbl_y = y + th + 10

            # Background Rectangle (Solid color based on class/ID)
            cv2.rectangle(frame, (x, lbl_y - th - 5), (x + tw + 4, lbl_y + 2), color, -1)
            
            # Text (White or Black depending on color brightness could be added, but c_white is fine)
            cv2.putText(frame, label, (x + 2, lbl_y - 2), self.font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame