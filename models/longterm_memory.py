import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

class LongTermMemory:
    def __init__(self, patience=9000, gallery_size=5, similarity_thresh=0.85):
        self.patience = patience
        self.gallery_size = gallery_size
        self.similarity_thresh = similarity_thresh # Raised threshold for safety
        
        # storage[global_uuid] = {'gallery': [tensor...], 'last_seen': frame_idx}
        self.storage = {} 
        
        # Maps Current Frame Model ID -> Global Consistent UUID
        self.id_map = {} 

    def update(self, current_frame_idx, detections, embeddings):
        """
        detections: List of track IDs from the model for this frame.
        embeddings: Tensor of shape (N, D)
        """
        if len(detections) == 0:
            return {}

        # 1. Normalize Embeddings
        query_feats = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Separate "Stable" Tracks from "New" Tracks
        # Stable = We already mapped this Model ID recently. Trust the tracker.
        # New    = Model says this is a new ID. Let's verify if it's actually an OLD one returning.
        
        stable_indices = []
        newborn_indices = []
        
        for i, track_id in enumerate(detections):
            if track_id in self.id_map:
                stable_indices.append(i)
            else:
                newborn_indices.append(i)

        # --- PHASE 1: PASSIVE UPDATE (Stable Tracks) ---
        # Don't search. Just update the gallery. Trust the motion.
        for i in stable_indices:
            track_id = detections[i]
            uuid = self.id_map[track_id]
            self._update_gallery(uuid, query_feats[i], current_frame_idx)

        # --- PHASE 2: ACTIVE SEARCH (Only for New Tracks) ---
        if len(newborn_indices) > 0:
            self._attempt_reid(current_frame_idx, detections, query_feats, newborn_indices)

        self._cleanup(current_frame_idx)
        return self.id_map

    def _attempt_reid(self, frame_idx, detections, all_feats, newborn_indices):
        # 1. Gather Candidate Gallery (Only LOST objects)
        # We cannot match against objects currently on screen (active_uuids).
        active_uuids = set(self.id_map.values())
        
        lost_uuids = []
        gallery_stack = []
        
        for uuid, data in self.storage.items():
            if uuid in active_uuids: continue
            
            # Don't revive if it died very recently (flicker protection)
            if (frame_idx - data['last_seen']) < 5: continue

            # Use the single most recent feature (or average) for speed
            # Or use stack if you want max-similarity
            # Here we stack all snapshots for robustness
            snaps = data['gallery']
            gallery_stack.extend(snaps)
            lost_uuids.extend([uuid] * len(snaps))

        if len(gallery_stack) == 0:
            # No history to match against. All newborns are truly new.
            for i in newborn_indices:
                tid = detections[i]
                self.id_map[tid] = tid
                self._update_gallery(tid, all_feats[i], frame_idx)
            return

        # 2. Build Cost Matrix
        # Rows = Newborn Candidates, Cols = Gallery Snapshots
        newborn_feats = all_feats[newborn_indices] # (Num_New, D)
        gallery_tensor = torch.stack(gallery_stack).to(newborn_feats.device) # (Num_Gal, D)
        
        # Similarity Matrix (Cosine)
        sim_matrix = torch.mm(newborn_feats, gallery_tensor.t()) # (Num_New, Num_Gal)
        
        # Convert to Cost Matrix for Hungarian (minimize cost)
        # Cost = 1 - Similarity
        cost_matrix = 1.0 - sim_matrix.cpu().numpy()
        
        # 3. Hungarian Algorithm (1-to-1 Matching)
        # row_idx refers to newborn_indices[row_idx]
        # col_idx refers to lost_uuids[col_idx]
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        assigned_newborn_local_indices = set()

        for r, c in zip(row_indices, col_indices):
            similarity = sim_matrix[r, c].item()
            
            if similarity > self.similarity_thresh:
                # MATCH FOUND!
                # Map the new model ID to the old UUID
                new_idx = newborn_indices[r]
                track_id = detections[new_idx]
                matched_uuid = lost_uuids[c]
                
                self.id_map[track_id] = matched_uuid
                self._update_gallery(matched_uuid, all_feats[new_idx], frame_idx)
                
                assigned_newborn_local_indices.add(r)

        # 4. Handle Unmatched Newborns (Truly New)
        for r in range(len(newborn_indices)):
            if r not in assigned_newborn_local_indices:
                new_idx = newborn_indices[r]
                track_id = detections[new_idx]
                
                # Assign new global ID (same as track ID usually)
                self.id_map[track_id] = track_id
                self._update_gallery(track_id, all_feats[new_idx], frame_idx)

    def _update_gallery(self, uuid, new_feat, frame_idx):
        if uuid not in self.storage:
            self.storage[uuid] = {'gallery': [], 'last_seen': frame_idx}
        
        self.storage[uuid]['last_seen'] = frame_idx
        gallery = self.storage[uuid]['gallery']
        
        if len(gallery) < self.gallery_size:
            gallery.append(new_feat.detach().cpu())
        else:
            gallery.pop(0)
            gallery.append(new_feat.detach().cpu())

    def _cleanup(self, current_frame):
        # Remove tracks older than patience
        self.storage = {u:d for u,d in self.storage.items() if (current_frame - d['last_seen']) < self.patience}
        # Sync id_map: Only keep map entries if the tracker is still outputting that ID
        # Wait! Actually id_map should grow as the tracker finds new things. 
        # But we need to remove entries from id_map if the tracker STOPS outputting them?
        # NO. We effectively rebuild the relevant parts of id_map every frame based on 'detections'.
        # However, for memory safety, we can prune `id_map` of keys not seen in a while if you want,
        # but technically id_map acts as the "Short Term Translation Layer".
        pass