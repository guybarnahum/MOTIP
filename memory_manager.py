import torch
import torch.nn.functional as F

class LongTermMemory:
    def __init__(self, patience=9000, gallery_size=5, similarity_thresh=0.75):
        self.patience = patience
        self.gallery_size = gallery_size
        self.similarity_thresh = similarity_thresh
        
        # storage[uuid] = {'gallery': [tensor...], 'last_seen': frame_idx}
        self.storage = {} 
        self.id_map = {} 

    def update(self, current_frame_idx, detections, embeddings):
        if len(detections) == 0:
            return {}

        # 1. Prepare Queries
        query_feats = F.normalize(embeddings, p=2, dim=1)
        
        # 2. Identify ACTIVE UUIDs (To exclude from search)
        # If tracker says ID 5, and ID 5 is mapped to UUID 100, then UUID 100 is ACTIVE.
        active_uuids = set()
        for track_id in detections:
            if track_id in self.id_map:
                active_uuids.add(self.id_map[track_id])

        # 3. Build Gallery Stack (Only from LOST tracks)
        lost_uuids = []
        gallery_stack = []
        
        for uuid, data in self.storage.items():
            # CRITICAL FIX: Do not search for objects that are already on screen!
            if uuid in active_uuids: 
                continue 
            
            # Additional Check: Don't revive if it was seen very recently (e.g. < 5 frames ago)
            # This prevents flickering when the tracker is actually stable.
            if (current_frame_idx - data['last_seen']) < 5:
                continue

            snaps = data['gallery']
            gallery_stack.extend(snaps)
            lost_uuids.extend([uuid] * len(snaps))

        # 4. Run Search
        matches = {}
        if len(gallery_stack) > 0:
            gallery_tensor = torch.stack(gallery_stack).to(query_feats.device)
            sim_matrix = torch.mm(query_feats, gallery_tensor.t())
            max_sims, indices = sim_matrix.max(dim=1)
            
            for i, (sim, idx) in enumerate(zip(max_sims, indices)):
                if sim > self.similarity_thresh:
                    matches[i] = lost_uuids[idx.item()]

        # 5. Assign IDs
        final_id_map = {}
        for i, track_id in enumerate(detections):
            feat = query_feats[i]

            # A. Already Known & Mapped (Consistency)
            if track_id in self.id_map:
                true_uuid = self.id_map[track_id]
                self._update_gallery(true_uuid, feat, current_frame_idx)
                final_id_map[track_id] = true_uuid
                
            # B. Revival (Found in Lost Gallery)
            elif i in matches:
                matched_uuid = matches[i]
                self.id_map[track_id] = matched_uuid
                self._update_gallery(matched_uuid, feat, current_frame_idx)
                final_id_map[track_id] = matched_uuid
                
            # C. Truly New
            else:
                self.id_map[track_id] = track_id
                self._update_gallery(track_id, feat, current_frame_idx)
                final_id_map[track_id] = track_id

        self._cleanup(current_frame_idx)
        return final_id_map

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
        # Sync id_map
        valid_uuids = set(self.storage.keys())
        self.id_map = {k:v for k,v in self.id_map.items() if v in valid_uuids}