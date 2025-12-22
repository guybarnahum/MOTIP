import torch
import torch.nn.functional as F

class LongTermMemory:
    def __init__(self, patience=9000, gallery_size=5, similarity_thresh=0.75):
        """
        patience: Frames to keep a lost ID in memory.
        gallery_size: Snapshots per ID.
        similarity_thresh: Cosine sim needed to revive a track.
        """
        self.patience = patience
        self.gallery_size = gallery_size
        self.similarity_thresh = similarity_thresh
        
        # Storage: { uuid: {'gallery': [tensor], 'last_seen': frame_idx} }
        self.storage = {} 
        
        # Map: { current_run_id: original_uuid }
        self.id_map = {} 

    def update(self, current_frame_idx, detections, embeddings):
        """
        detections: list of IDs from the tracker
        embeddings: tensor [N, 256] matching the detections
        Returns: Dict mapping {tracker_id: memory_uuid}
        """
        if len(detections) == 0:
            return {}

        embeddings = F.normalize(embeddings, p=2, dim=1)
        final_id_map = {} 
        
        for i, track_id in enumerate(detections):
            feat = embeddings[i]
            
            # CASE A: Known active track
            if track_id in self.id_map:
                true_uuid = self.id_map[track_id]
                self._update_gallery(true_uuid, feat, current_frame_idx)
                final_id_map[track_id] = true_uuid
                
            # CASE B: New track (check history)
            else:
                matched_uuid = self._query_memory(feat)
                if matched_uuid is not None:
                    # REVIVAL
                    self.id_map[track_id] = matched_uuid 
                    self._update_gallery(matched_uuid, feat, current_frame_idx)
                    final_id_map[track_id] = matched_uuid
                else:
                    # GENUINELY NEW
                    self.id_map[track_id] = track_id
                    self._update_gallery(track_id, feat, current_frame_idx)
                    final_id_map[track_id] = track_id

        self._cleanup(current_frame_idx)
        return final_id_map

    def _query_memory(self, new_feat):
        """Compare new_feat against all lost tracks."""
        best_sim = -1
        best_uuid = None
        
        # Optimization: Only check IDs not currently active
        active_uuids = set(self.id_map.values())

        for uuid, data in self.storage.items():
            if uuid in active_uuids: continue
                
            gallery = torch.stack(data['gallery']).to(new_feat.device)
            sims = F.cosine_similarity(new_feat.unsqueeze(0), gallery)
            max_sim = sims.max().item()
            
            if max_sim > self.similarity_thresh and max_sim > best_sim:
                best_sim = max_sim
                best_uuid = uuid
        return best_uuid

    def _update_gallery(self, uuid, new_feat, frame_idx):
        if uuid not in self.storage:
            self.storage[uuid] = {'gallery': [], 'last_seen': frame_idx}
        
        self.storage[uuid]['last_seen'] = frame_idx
        gallery = self.storage[uuid]['gallery']
        
        if len(gallery) < self.gallery_size:
            gallery.append(new_feat.detach().cpu())
        else:
            # Diversity check: replace most similar if new one is distinct
            gallery_tensor = torch.stack(gallery).to(new_feat.device)
            sims = F.cosine_similarity(new_feat.unsqueeze(0), gallery_tensor)
            max_sim, idx = sims.max(dim=0)
            if max_sim < 0.95:
                gallery[idx] = new_feat.detach().cpu()

    def _cleanup(self, current_frame):
        to_remove = [u for u, d in self.storage.items() 
                     if (current_frame - d['last_seen']) > self.patience]
        for uuid in to_remove:
            del self.storage[uuid]
            # Clean reverse map
            self.id_map = {k:v for k,v in self.id_map.items() if v != uuid}
