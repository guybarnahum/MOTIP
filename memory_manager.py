import torch
import torch.nn.functional as F

class LongTermMemory:
    def __init__(self, patience=9000, gallery_size=5, similarity_thresh=0.75):
        self.patience = patience
        self.gallery_size = gallery_size
        self.similarity_thresh = similarity_thresh
        
        self.storage = {} 
        self.id_map = {} 

    def update(self, current_frame_idx, detections, embeddings):
        """
        Vectorized Update: Fast matching for hundreds of objects.
        """
        if len(detections) == 0:
            return {}

        # 1. Prepare Queries (Current Detections)
        # Shape: [N, 256]
        query_feats = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Prepare Gallery (Lost Tracks) - VECTORIZED
        # We only want to search tracks that are NOT currently active.
        active_uuids = set(self.id_map.values())
        lost_uuids = []
        gallery_stack = []
        
        # Flatten the storage into a single tensor list
        # This part is still a loop, but we only do it once per frame and 
        # only for valid candidates. (Further optimization: cache the stack)
        for uuid, data in self.storage.items():
            if uuid in active_uuids: continue
            
            # Add all K snapshots of this person
            snaps = data['gallery'] # list of tensors
            gallery_stack.extend(snaps)
            # Keep track of which UUID belongs to which snapshot
            lost_uuids.extend([uuid] * len(snaps))

        final_id_map = {}
        
        # 3. RUN MATRIX MULTIPLICATION (If we have lost tracks)
        matches = {} # {query_idx: matched_uuid}
        
        if len(gallery_stack) > 0:
            # Stack: [Total_Snapshots, 256]
            gallery_tensor = torch.stack(gallery_stack).to(query_feats.device)
            
            # Matrix Mult: [N, 256] @ [256, Total_Snapshots] -> [N, Total_Snapshots]
            # This calculates sim score for EVERY pair instantly.
            sim_matrix = torch.mm(query_feats, gallery_tensor.t())
            
            # Find best match for each Query
            # max_sims: [N], indices: [N]
            max_sims, indices = sim_matrix.max(dim=1)
            
            # Filter by Threshold
            for i, (sim, idx) in enumerate(zip(max_sims, indices)):
                if sim > self.similarity_thresh:
                    # Map the flat index back to the UUID
                    matched_uuid = lost_uuids[idx.item()]
                    matches[i] = matched_uuid

        # 4. Assign IDs based on matches
        for i, track_id in enumerate(detections):
            # CASE A: Known active track
            if track_id in self.id_map:
                true_uuid = self.id_map[track_id]
                self._update_gallery(true_uuid, query_feats[i], current_frame_idx)
                final_id_map[track_id] = true_uuid
                
            # CASE B: Revival (Found in Matrix Search)
            elif i in matches:
                matched_uuid = matches[i]
                # print(f"♻️ VECTOR REVIVAL: {track_id} -> {matched_uuid}")
                self.id_map[track_id] = matched_uuid 
                self._update_gallery(matched_uuid, query_feats[i], current_frame_idx)
                final_id_map[track_id] = matched_uuid
                
            # CASE C: New
            else:
                self.id_map[track_id] = track_id
                self._update_gallery(track_id, query_feats[i], current_frame_idx)
                final_id_map[track_id] = track_id

        self._cleanup(current_frame_idx)
        return final_id_map

    def _update_gallery(self, uuid, new_feat, frame_idx):
        if uuid not in self.storage:
            self.storage[uuid] = {'gallery': [], 'last_seen': frame_idx}
        
        self.storage[uuid]['last_seen'] = frame_idx
        gallery = self.storage[uuid]['gallery']
        
        # Simple FIFO for speed in this demo, 
        # or use the diversity logic if speed allows
        if len(gallery) < self.gallery_size:
            gallery.append(new_feat.detach().cpu())
        else:
            # Replace oldest (simplest scaling)
            # Or replace random/most similar
            gallery.pop(0) 
            gallery.append(new_feat.detach().cpu())

    def _cleanup(self, current_frame):
        # Python dict comprehension is fast enough for cleanup
        # as it doesn't involve heavy GPU compute
        self.storage = {
            u: d for u, d in self.storage.items() 
            if (current_frame - d['last_seen']) < self.patience
        }
        
        # Clean id_map based on what remains in storage
        valid_uuids = set(self.storage.keys())
        # Also keep current frame's active IDs even if not in storage yet
        self.id_map = {
            k: v for k, v in self.id_map.items() 
            if v in valid_uuids or k in valid_uuids # simplistic safety check
        }