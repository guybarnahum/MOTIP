# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import os
import einops
from scipy.optimize import linear_sum_assignment

from utils.misc import distributed_device
from utils.box_ops import box_cxcywh_to_xywh
from models.misc import get_model

class RuntimeTracker:
    def __init__(
            self,
            model,
            # Sequence infos:
            sequence_hw: tuple,
            # Inference settings:
            use_sigmoid: bool = False,
            assignment_protocol: str = "hungarian",
            miss_tolerance: int = 30,
            det_thresh: float = 0.5,
            newborn_thresh: float = 0.5,
            id_thresh: float = 0.1,
            area_thresh: int = 0,
            only_detr: bool = False,
            dtype: torch.dtype = torch.float32,
    ):
        self.model = model
        self.model.eval()

        self.dtype = dtype

        # For FP16:
        if self.dtype != torch.float32:
            if self.dtype == torch.float16:
                self.model.half()
            else:
                raise NotImplementedError(f"Unsupported dtype {self.dtype}.")

        self.use_sigmoid = use_sigmoid
        self.assignment_protocol = assignment_protocol.lower()
        self.miss_tolerance = miss_tolerance
        self.det_thresh = det_thresh
        self.newborn_thresh = newborn_thresh
        self.id_thresh = id_thresh
        self.area_thresh = area_thresh
        self.only_detr = only_detr
        self.num_id_vocabulary = get_model(model).num_id_vocabulary

        # Check for the legality of settings:
        assert self.assignment_protocol in ["hungarian", "id-max", "object-max", "object-priority", "id-priority"], \
            f"Assignment protocol {self.assignment_protocol} is not supported."

        self.bbox_unnorm = torch.tensor(
            [sequence_hw[1], sequence_hw[0], sequence_hw[1], sequence_hw[0]],
            dtype=dtype,
            device=distributed_device(),
        )

        # Trajectory fields:
        self.next_id = 0
        self.id_label_to_id = {}
        
        # Calculate split point
        self.split_idx = self.num_id_vocabulary // 2
        
        # --- FACTORED LOGIC: Dynamic Pools ---
        self.person_id_pool  = list(range(0, self.split_idx))
        self.vehicle_id_pool = list(range(self.split_idx, self.num_id_vocabulary))
        
        self.person_ptr  = 0
        self.vehicle_ptr = 0

        # All fields are in shape (T, N, ...)
        self.trajectory_features = torch.zeros(
            (0, 0, 256), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_boxes = torch.zeros(
            (0, 0, 4), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_id_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        # Multi-class support:
        self.trajectory_category_labels = torch.zeros(
            (0, 0), dtype=torch.int64, device=distributed_device(),
        )
        self.trajectory_times = torch.zeros(
            (0, 0), dtype=dtype, device=distributed_device(),
        )
        self.trajectory_masks = torch.zeros(
            (0, 0), dtype=torch.bool, device=distributed_device(),
        )

        self.current_track_results = {}
        return


    @torch.no_grad()
    def update(self, image):
        detr_out = self.model(frames=image, part="detr")
        self.output = detr_out

        scores, categories, boxes, output_embeds = self._get_activate_detections(detr_out=detr_out)
        
        if self.only_detr:
            id_pred_labels = self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            id_pred_labels = self._get_id_pred_labels(boxes=boxes, output_embeds=output_embeds, categories=categories)
        
        # Newborn Filtering
        keep_idxs = (id_pred_labels != self.num_id_vocabulary) | (scores > self.newborn_thresh)
        scores, categories, boxes = scores[keep_idxs], categories[keep_idxs], boxes[keep_idxs]
        output_embeds, id_pred_labels = output_embeds[keep_idxs], id_pred_labels[keep_idxs]

        # --- DEBUG TELEMETRY START ---
        # Capture how many were predicted as newborns before assignment
        newborn_mask = (id_pred_labels == self.num_id_vocabulary)
        self.debug_info = {
            "ptr_p": self.person_ptr,
            "ptr_v": self.vehicle_ptr,
            "newborns": newborn_mask.sum().item(),
            "matches": (~newborn_mask).sum().item()
        }
        # --- DEBUG TELEMETRY END ---

        # Newborn Assignment (Now uses the pointer-based method)
        id_labels = self._assign_newborn_id_labels(pred_id_labels=id_pred_labels, categories=categories)

        # Update the results with a safer "Background" fallback
        ids_list = []
        for l in id_labels:
            label_val = l.item()
            # Fallback to num_id_vocabulary (background) 
            ids_list.append(self.id_label_to_id.get(label_val, self.num_id_vocabulary))

        self.current_track_results = {
            "score": scores,
            "category": categories,
            "bbox": box_cxcywh_to_xywh(boxes) * self.bbox_unnorm,
            "id": torch.tensor(ids_list, dtype=torch.int64, device=boxes.device),
            "embeddings": output_embeds,
            "id_labels" : id_labels,
        }

        # Optional debug logging (enable with env var RUNTIME_TRACKER_DEBUG=1)
        if os.environ.get("RUNTIME_TRACKER_DEBUG") == "1":
            try:
                tid_head = self.trajectory_id_labels[0].cpu().tolist() if self.trajectory_id_labels.shape[0] > 0 else []
            except Exception:
                tid_head = str(self.trajectory_id_labels.shape)
            # Basic snapshot
            print("[RUNTIME_TRACKER DEBUG] id_labels=", id_labels.cpu().tolist(), "ids=", ids_list, "id_label_to_id=", dict(self.id_label_to_id), "traj_head=", tid_head)

            # Extended snapshot: shapes and internal pointers
            try:
                shapes = {
                    'features': tuple(self.trajectory_features.shape),
                    'boxes': tuple(self.trajectory_boxes.shape),
                    'id_labels': tuple(self.trajectory_id_labels.shape),
                    'masks': tuple(self.trajectory_masks.shape),
                }
            except Exception:
                shapes = str((self.trajectory_features.shape, self.trajectory_boxes.shape, self.trajectory_id_labels.shape))
            print("[RUNTIME_TRACKER DEBUG] shapes=", shapes, "next_id=", self.next_id, "person_ptr=", self.person_ptr, "vehicle_ptr=", self.vehicle_ptr)

            # Column-level mapping (column idx -> id_label -> global id)
            try:
                if self.trajectory_id_labels.shape[0] > 0:
                    col_labels = self.trajectory_id_labels[0].cpu().tolist()
                    col_map = []
                    for ci, lab in enumerate(col_labels):
                        col_map.append((ci, lab, self.id_label_to_id.get(lab, None)))
                else:
                    col_map = []
            except Exception:
                col_map = str(self.trajectory_id_labels.shape)
            print("[RUNTIME_TRACKER DEBUG] col_map=", col_map)

        # Trajectory & Cleanup
        self._update_trajectory_infos(boxes=boxes, output_embeds=output_embeds, id_labels=id_labels, categories=categories)
        self._filter_out_inactive_tracks()
        return

    def get_track_results(self):
        return self.current_track_results

    def _get_activate_detections(self, detr_out: dict):
        logits = detr_out["pred_logits"][0]
        boxes = detr_out["pred_boxes"][0]
        output_embeds = detr_out["outputs"][0]
        scores = logits.sigmoid()
        scores, categories = torch.max(scores, dim=-1)
        area = boxes[:, 2] * self.bbox_unnorm[2] * boxes[:, 3] * self.bbox_unnorm[3]
        activate_indices = (scores > self.det_thresh) & (area > self.area_thresh)
        # Selecting:
        boxes = boxes[activate_indices]
        output_embeds = output_embeds[activate_indices]
        scores = scores[activate_indices]
        categories = categories[activate_indices]
        return scores, categories, boxes, output_embeds

    def _get_id_pred_labels(self, boxes: torch.Tensor, output_embeds: torch.Tensor, categories: torch.Tensor):
        if self.trajectory_features.shape[0] == 0:
            return self.num_id_vocabulary * torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
        else:
            # 1. prepare current infos:
            current_features = output_embeds[None, ...]     # (T, N, C)
            current_boxes = boxes[None, ...]                # (T, N, 4)
            current_masks = torch.zeros((1, output_embeds.shape[0]), dtype=torch.bool, device=distributed_device())
            current_times = self.trajectory_times.shape[0] * torch.ones(
                (1, output_embeds.shape[0]), dtype=torch.int64, device=distributed_device(),
            )
            
            # 2. prepare seq_info:
            seq_info = {
                "trajectory_features": self.trajectory_features[None, None, ...],    # (B, G, T, N, C)
                "trajectory_boxes": self.trajectory_boxes[None, None, ...],          # (B, G, T, N, 4)
                "trajectory_id_labels": self.trajectory_id_labels[None, None, ...],    # (B, G, T, N)
                "trajectory_class_labels": self.trajectory_category_labels[None, None, ...], # (B, G, T, N)
                "trajectory_times": self.trajectory_times[None, None, ...],          # (B, G, T, N)
                "trajectory_masks": self.trajectory_masks[None, None, ...],          # (B, G, T, N)
                
                "unknown_features": current_features[None, None, ...],               # (B, G, T, N, C)
                "unknown_boxes": current_boxes[None, None, ...],                     # (B, G, T, N, 4)
                "unknown_class_labels": categories[None, None, None, ...], 
                "unknown_masks": current_masks[None, None, ...],                     # (B, G, T, N)
                "unknown_times": current_times[None, None, ...],                     # (B, G, T, N)
            }
            
            # 3. forward:
            seq_info = self.model(seq_info=seq_info, part="trajectory_modeling")
            id_logits, _, _ = self.model(seq_info=seq_info, part="id_decoder")

            # 4. get scores (with masking):
            id_logits = id_logits[0, 0, 0] # (N, Vocab)

            # Extra forensic debug: print raw logits/top-k and argmaxs for first few objects
            if os.environ.get("RUNTIME_TRACKER_DEBUG") == "1":
                try:
                    raw = id_logits.detach().cpu()
                    n_show = min(3, raw.shape[0])
                    topk_vals, topk_idx = raw.topk(5, dim=-1)
                    sample_info = []
                    for i in range(n_show):
                        sample_info.append((i, topk_idx[i].tolist(), [float(v) for v in topk_vals[i].tolist()]))
                    print("[RUNTIME_TRACKER DEBUG] raw_id_logits_topk=", sample_info)
                    print("[RUNTIME_TRACKER DEBUG] id_logits_argmax=", raw.argmax(dim=-1).tolist(), "argmax_max=", [float(x) for x in raw.max(dim=-1).values.tolist()])
                except Exception:
                    pass
            
            # --- ENHANCEMENT: Logit Masking to enforce partitions ---
            
            mask = torch.zeros_like(id_logits)
            is_person  = (categories == 0)
            is_vehicle = (categories == 1)

            mask[is_person ,  self.split_idx:] = -10000.0  # Block vehicle range for persons
            mask[is_vehicle, :self.split_idx ] = -10000.0 # Block person range for vehicles
            id_logits = id_logits + mask
            # --------------------------------------------------------

            if not self.use_sigmoid:
                id_scores = id_logits.softmax(dim=-1)
            else:
                id_scores = id_logits.sigmoid()

            id_scores[is_person ,  self.split_idx:] = 0.0
            id_scores[is_vehicle, :self.split_idx ] = 0.0

            # IMPORTANT: reduce id_scores to the current tracker columns (stable
            # `trajectory_id_labels`) plus the newborn slot. This makes assignment
            # outputs refer to tracker column indices (0..num_cols) instead of
            # global vocabulary indices (0..V-1). We store the mapping in
            # `self._col_id_labels_for_assignment` for assignment helpers to use.
            self._col_id_labels_for_assignment = None
            if self.trajectory_id_labels.shape[0] > 0:
                try:
                    col_id_labels = list(self.trajectory_id_labels[0].tolist())
                    # gather columns for existing labels
                    cols = torch.tensor(col_id_labels, dtype=torch.long, device=id_scores.device)
                    if cols.numel() > 0:
                        id_scores_cols = id_scores.index_select(dim=1, index=cols)
                    else:
                        id_scores_cols = id_scores.new_zeros((id_scores.shape[0], 0))
                    # newborn column is the vocabulary index equal to `num_id_vocabulary`
                    newborn_idx = int(self.num_id_vocabulary)
                    newborn_col = id_scores[:, newborn_idx:newborn_idx+1]
                    id_scores = torch.cat((id_scores_cols, newborn_col), dim=1)
                    # mapping: column_index -> id_label (last entry is newborn sentinel)
                    self._col_id_labels_for_assignment = col_id_labels + [self.num_id_vocabulary]
                except Exception:
                    # If any error occurs, fall back to using full-vocab id_scores
                    self._col_id_labels_for_assignment = None

            # 5. assign id labels:
            match self.assignment_protocol:
                case "hungarian": id_labels = self._hungarian_assignment(id_scores=id_scores)
                case "object-max": id_labels = self._object_max_assignment(id_scores=id_scores)
                case "id-max": id_labels = self._id_max_assignment(id_scores=id_scores)
                case _: raise NotImplementedError

            id_pred_labels = torch.tensor(id_labels, dtype=torch.int64, device=distributed_device())
            return id_pred_labels

    def _assign_newborn_id_labels(self, pred_id_labels: torch.Tensor, categories: torch.Tensor):
        newborn_mask = (pred_id_labels == self.num_id_vocabulary)
        if not newborn_mask.any():
            return pred_id_labels

        newborn_indices = newborn_mask.nonzero(as_tuple=True)[0]
        
        # Get pool sizes dynamically
        p_size = len(self.person_id_pool)
        v_size = len(self.vehicle_id_pool)
        
        for idx in newborn_indices:
            cat = categories[idx].item()
            
            # Select ID using the pointer and the dynamic pool size
            if cat == 0:
                new_id_label = self.person_id_pool[self.person_ptr % p_size]
                self.person_ptr += 1
            else:
                new_id_label = self.vehicle_id_pool[self.vehicle_ptr % v_size]
                self.vehicle_ptr += 1
            
            # --- Reuse existing lane instead of deleting columns ---
            if self.trajectory_id_labels.shape[0] > 0:
                rem_idx = (self.trajectory_id_labels[0] == new_id_label).nonzero(as_tuple=True)[0]
                if rem_idx.numel() > 0:
                    # Reuse the first matching column index to avoid re-ordering columns
                    col = int(rem_idx[0].item())
                    if self.trajectory_features.shape[0] > 0:
                        self.trajectory_features[:, col, :] = 0
                        self.trajectory_boxes[:, col, :] = 0
                        self.trajectory_times[:, col] = 0
                        self.trajectory_masks[:, col] = True
                    # Keep `trajectory_id_labels` consistent and update category
                    self.trajectory_id_labels[:, col] = new_id_label
                    self.trajectory_category_labels[:, col] = cat

            # Assign numeric ID for this id_label only if not already mapped
            if new_id_label not in self.id_label_to_id:
                self.id_label_to_id[new_id_label] = self.next_id
                if os.environ.get("RUNTIME_TRACKER_DEBUG") == "1":
                    print(f"[RUNTIME_TRACKER DEBUG] new mapping id_label->{new_id_label} => global id {self.next_id}")
                self.next_id += 1
            pred_id_labels[idx] = new_id_label

        return pred_id_labels

    def _update_trajectory_infos(self, boxes: torch.Tensor, output_embeds: torch.Tensor, id_labels: torch.Tensor, categories: torch.Tensor):
        # 1. cut trajectory infos:
        self.trajectory_features = self.trajectory_features[-self.miss_tolerance + 2:, ...]
        self.trajectory_boxes = self.trajectory_boxes[-self.miss_tolerance + 2:, ...]
        self.trajectory_id_labels = self.trajectory_id_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_category_labels = self.trajectory_category_labels[-self.miss_tolerance + 2:, ...]
        self.trajectory_times = self.trajectory_times[-self.miss_tolerance + 2:, ...]
        self.trajectory_masks = self.trajectory_masks[-self.miss_tolerance + 2:, ...]
        # 2. find out all new instances:
        already_id_labels = set(self.trajectory_id_labels[0].tolist() if self.trajectory_id_labels.shape[0] > 0 else [])
        _id_labels = set(id_labels.tolist())
        newborn_id_labels = _id_labels - already_id_labels
        # 3. add newborn instances to trajectory infos:
        if len(newborn_id_labels) > 0:
            # Preserve detection order when adding newborns to avoid
            # non-deterministic column ordering caused by iterating over sets.
            newborn_id_labels_list = []
            for l in id_labels.tolist():
                if l in newborn_id_labels and l not in newborn_id_labels_list:
                    newborn_id_labels_list.append(l)
            newborn_id_labels_tensor = torch.tensor(newborn_id_labels_list, dtype=torch.int64, device=distributed_device())
            _T = self.trajectory_id_labels.shape[0]
            _N = len(newborn_id_labels_list)
            _id_labels = einops.repeat(newborn_id_labels_tensor, 'n -> t n', t=_T)
            
            newborn_cat_list = []
            for nid in newborn_id_labels_list:
                idx = (id_labels == nid).nonzero(as_tuple=True)[0][0]
                newborn_cat_list.append(categories[idx].item())
            _categories = einops.repeat(torch.tensor(newborn_cat_list, dtype=torch.int64, device=distributed_device()), 'n -> t n', t=_T)
            
            _boxes = torch.zeros((_T, _N, 4), dtype=self.dtype, device=distributed_device())
            _times = einops.repeat(
                torch.arange(_T, dtype=torch.int64, device=distributed_device()), 't -> t n', n=_N,
            )
            _features = torch.zeros(
                (_T, _N, 256), dtype=self.dtype, device=distributed_device(),
            )
            _masks = torch.ones((_T, _N), dtype=torch.bool, device=distributed_device())
            # 3.1. padding to trajectory infos:
            self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, _id_labels], dim=1)
            self.trajectory_category_labels = torch.cat([self.trajectory_category_labels, _categories], dim=1)
            self.trajectory_boxes = torch.cat([self.trajectory_boxes, _boxes], dim=1)
            self.trajectory_times = torch.cat([self.trajectory_times, _times], dim=1)
            self.trajectory_features = torch.cat([self.trajectory_features, _features], dim=1)
            self.trajectory_masks = torch.cat([self.trajectory_masks, _masks], dim=1)
        # 4. update trajectory infos:
        _N = self.trajectory_id_labels.shape[1]
        current_id_labels = self.trajectory_id_labels[0] if self.trajectory_id_labels.shape[0] > 0 else id_labels
        current_features = torch.zeros((_N, 256), dtype=self.dtype, device=distributed_device())
        current_boxes = torch.zeros((_N, 4), dtype=self.dtype, device=distributed_device())
        current_categories = self.trajectory_category_labels[0] if self.trajectory_category_labels.shape[0] > 0 else categories
        current_times = self.trajectory_id_labels.shape[0] * torch.ones((_N,), dtype=torch.int64, device=distributed_device())
        current_masks = torch.ones((_N,), dtype=torch.bool, device=distributed_device())
        # 4.1. find out the same id labels (matching):
        indices = torch.eq(current_id_labels[:, None], id_labels[None, :]).nonzero(as_tuple=False)
        current_idxs = indices[:, 0]
        idxs = indices[:, 1]
        # 4.2. fill in the infos:
        current_id_labels[current_idxs] = id_labels[idxs]
        current_features[current_idxs] = output_embeds[idxs]
        current_boxes[current_idxs] = boxes[idxs]
        current_categories[current_idxs] = categories[idxs]
        current_masks[current_idxs] = False
        # 4.3. cat to trajectory infos:
        self.trajectory_features = torch.cat([self.trajectory_features, current_features[None, ...]], dim=0).contiguous()
        self.trajectory_boxes = torch.cat([self.trajectory_boxes, current_boxes[None, ...]], dim=0).contiguous()
        self.trajectory_id_labels = torch.cat([self.trajectory_id_labels, current_id_labels[None, ...]], dim=0).contiguous()
        self.trajectory_category_labels = torch.cat([self.trajectory_category_labels, current_categories[None, ...]], dim=0).contiguous()
        self.trajectory_times = torch.cat([self.trajectory_times, current_times[None, ...]], dim=0).contiguous()
        self.trajectory_masks = torch.cat([self.trajectory_masks, current_masks[None, ...]], dim=0).contiguous()
        # 4.4. fix "times":
        self.trajectory_times = einops.repeat(
            torch.arange(self.trajectory_times.shape[0], dtype=torch.int64, device=distributed_device()),
            't -> t n', n=self.trajectory_times.shape[1],
        ).contiguous().clone()
        return

    def _filter_out_inactive_tracks(self):
        # Determine which columns have any active (unmasked) entries.
        is_active = torch.sum((~self.trajectory_masks).to(torch.int64), dim=0) > 0
        # Preserve column ordering and indices: do NOT compact/squeeze columns in-place.
        # Compaction previously caused column index re-ordering which led to
        # ID switches when consumers relied on stable column indices.
        # Instead, keep inactive columns but ensure their feature/box buffers
        # are cleared and masks set so they can be safely reused later.
        if self.trajectory_features.numel() == 0:
            return
        inactive = (~is_active).to(self.trajectory_masks.device)
        if inactive.any():
            # Zero out buffers for inactive columns but KEEP the id label stored
            # so column <-> id_label mapping remains stable across frames.
            self.trajectory_features[:, inactive, :] = 0
            self.trajectory_boxes[:, inactive, :] = 0
            # keep `trajectory_id_labels` as-is to preserve mapping
            # keep `trajectory_category_labels` as-is as well
            self.trajectory_times[:, inactive] = 0
            self.trajectory_masks[:, inactive] = True
        return

    def _hungarian_assignment(self, id_scores: torch.Tensor):
        # Ensure we produce a label per object index (row). Using append caused
        # ordering/misalignment bugs when interpreting Hungarian outputs.
        num_objs = id_scores.shape[0]
        if num_objs == 0:
            return []

        if num_objs > 1:
            id_scores_newborn_repeat = id_scores[:, -1:].repeat(1, num_objs - 1)
            id_scores = torch.cat((id_scores, id_scores_newborn_repeat), dim=-1)

        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist()) if self.trajectory_id_labels.shape[0] > 0 else set()
        # If id_scores were reduced to tracker columns, use the stored mapping
        col_map = getattr(self, "_col_id_labels_for_assignment", None)

        # Initialize all as newborn (background) then fill by assigned row index
        id_labels = [self.num_id_vocabulary] * num_objs
        match_rows, match_cols = linear_sum_assignment(1 - id_scores.cpu())
        # Debug: show top scores per row
        try:
            tops = [list(enumerate(row.tolist()))[:5] for row in id_scores.cpu()]
            if os.environ.get("RUNTIME_TRACKER_DEBUG") == "1":
                print("[RUNTIME_TRACKER DEBUG] hungarian id_scores_rows_sample=", tops)
        except Exception:
            pass
        for r, c in zip(match_rows.tolist(), match_cols.tolist()):
            if col_map is not None:
                # c is a column index into `col_map`
                if c < 0 or c >= len(col_map):
                    label = self.num_id_vocabulary
                else:
                    label = int(col_map[c])
                # apply confidence threshold on the reduced id_scores
                if id_scores[r, c] < self.id_thresh:
                    label = self.num_id_vocabulary
                # ensure label exists in trajectory set unless it's newborn
                if label != self.num_id_vocabulary and label not in trajectory_id_labels_set:
                    label = self.num_id_vocabulary
            else:
                _id = c
                if _id >= self.num_id_vocabulary:
                    label = self.num_id_vocabulary
                elif _id not in trajectory_id_labels_set:
                    label = self.num_id_vocabulary
                elif id_scores[r, _id] < self.id_thresh:
                    label = self.num_id_vocabulary
                else:
                    label = int(_id)
            id_labels[r] = label
        if os.environ.get("RUNTIME_TRACKER_DEBUG") == "1":
            print("[RUNTIME_TRACKER DEBUG] hungarian_assignments_rows=", match_rows.tolist(), "cols=", match_cols.tolist(), "result_labels=", id_labels)
        return id_labels

    def _object_max_assignment(self, id_scores: torch.Tensor):
        id_labels = list()
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist()) if self.trajectory_id_labels.shape[0] > 0 else set()
        col_map = getattr(self, "_col_id_labels_for_assignment", None)
        object_max_confs, object_max_id_labels = torch.max(id_scores, dim=-1)
        id_max_confs = dict()
        # Build per-id (actual id_label) max confidences
        for conf, id_label_idx in zip(object_max_confs.tolist(), object_max_id_labels.tolist()):
            if col_map is not None:
                mapped = int(col_map[int(id_label_idx)]) if 0 <= int(id_label_idx) < len(col_map) else self.num_id_vocabulary
            else:
                mapped = int(id_label_idx)
            if mapped not in id_max_confs:
                id_max_confs[mapped] = conf
            else:
                id_max_confs[mapped] = max(id_max_confs[mapped], conf)
        if self.num_id_vocabulary in id_max_confs:
            id_max_confs[self.num_id_vocabulary] = 0.0

        for obj_idx in range(len(object_max_id_labels)):
            id_label_idx = int(object_max_id_labels[obj_idx].item())
            mapped_label = int(col_map[id_label_idx]) if (col_map is not None and 0 <= id_label_idx < len(col_map)) else id_label_idx
            if mapped_label not in trajectory_id_labels_set:
                id_labels.append(self.num_id_vocabulary)
            else:
                _conf = float(object_max_confs[obj_idx].item())
                if _conf < self.id_thresh or _conf < id_max_confs.get(mapped_label, -1.0):
                    id_labels.append(self.num_id_vocabulary)
                elif mapped_label in id_labels:
                    id_labels.append(self.num_id_vocabulary)
                else:
                    id_labels.append(mapped_label)
        if os.environ.get("RUNTIME_TRACKER_DEBUG") == "1":
            print("[RUNTIME_TRACKER DEBUG] object_max_result=", id_labels)
        return id_labels

    def _id_max_assignment(self, id_scores: torch.Tensor):
        id_labels = [self.num_id_vocabulary] * len(id_scores)
        trajectory_id_labels_set = set(self.trajectory_id_labels[0].tolist()) if self.trajectory_id_labels.shape[0] > 0 else set()
        col_map = getattr(self, "_col_id_labels_for_assignment", None)
        # id_max_confs: per-column confidence; id_max_obj_idxs: object idx which gives that max for that column
        id_max_confs, id_max_obj_idxs = torch.max(id_scores, dim=0)
        object_max_confs = dict()
        for conf, object_idx in zip(id_max_confs.tolist(), id_max_obj_idxs.tolist()):
            if object_idx not in object_max_confs:
                object_max_confs[object_idx] = conf
            else:
                if conf == object_max_confs[object_idx]:
                    conf = conf - 0.0001
                object_max_confs[object_idx] = max(object_max_confs[object_idx], conf)
        # Iterate per column to find its best object and possibly assign
        for col_idx in range(len(id_max_obj_idxs)):
            _obj_idx = int(id_max_obj_idxs[col_idx].item())
            _conf = float(id_max_confs[col_idx].item())
            _id_label = int(col_map[col_idx]) if (col_map is not None and 0 <= col_idx < len(col_map)) else col_idx
            if _conf < self.id_thresh or _conf < object_max_confs.get(_obj_idx, -1.0):
                continue
            if _id_label not in trajectory_id_labels_set:
                continue
            # Assign this id_label to the corresponding object index
            if _obj_idx < len(id_labels):
                id_labels[_obj_idx] = _id_label
        if os.environ.get("RUNTIME_TRACKER_DEBUG") == "1":
            print("[RUNTIME_TRACKER DEBUG] id_max_result=", id_labels)
        return id_labels