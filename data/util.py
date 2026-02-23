# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops

from utils.nested_tensor import nested_tensor_from_tensor_list


def is_legal(annotation: dict):
    assert "id" in annotation, "Annotation must have 'id' field."
    assert "category" in annotation, "Annotation must have 'category' field."
    assert "bbox" in annotation, "Annotation must have 'bbox' field."
    assert "visibility" in annotation, "Annotation must have 'visibility' field."

    assert len(annotation["id"]) == len(annotation["category"]) \
           == len(annotation["bbox"]) == len(annotation["visibility"]), \
           "The length of 'id', 'category', 'bbox', 'visibility' must be the same."

    # assert torch.unique(annotation["id"]).size(0) == annotation["id"].size(0), f"IDs must be unique."
    _id_unique = torch.unique(annotation["id"]).size(0) == annotation["id"].size(0)     # for PersonPath22

    # A hack implementation for DETR (300 queries):
    # TODO: to make it more general, maybe pass the number of queries as an parameter.
    leq_300 = annotation["id"].shape[0] <= 300

    # return len(annotation["id"]) > 0
    return len(annotation["id"]) > 0 and _id_unique and leq_300


def append_annotation(
        annotation: dict,
        obj_id: int,
        category: int,
        bbox: list,
        visibility: float,
):
    annotation["id"] = torch.cat([
        annotation["id"],
        torch.tensor([obj_id], dtype=torch.int64)
    ])
    annotation["category"] = torch.cat([
        annotation["category"],
        torch.tensor([category], dtype=torch.int64)
    ])
    annotation["bbox"] = torch.cat([
        annotation["bbox"],
        torch.tensor([bbox], dtype=torch.float32)
    ])
    annotation["visibility"] = torch.cat([
        annotation["visibility"],
        torch.tensor([visibility], dtype=torch.float32)
    ])

    # --- ADDED FOR MULTI-CLASS MOTIP SUPPORT ---
    # Shape logic: (N, 1, 1) is expected by the prepare_for_motip method
    # We populate both trajectory and unknown labels here; 
    # MOTIP's internal logic will decide which one to use during the training pass.
    class_label = torch.tensor([[[category]]], dtype=torch.int64)
    
    for key in ["trajectory_class_labels", "unknown_class_labels"]:
        if key not in annotation:
            annotation[key] = class_label
        else:
            # IMPORTANT: Concatenate on dim=2 to increase the number of objects (N)
            annotation[key] = torch.cat([annotation[key], class_label], dim=2)
    # --------------------------------------------

    return annotation


def collate_fn(batch):
    images, annotations, metas = zip(*batch)    # (B, T, ...)
    _B = len(batch)
    _T = len(images[0])
    images_list = [clip[_] for clip in images for _ in range(len(clip))]
    size_divisibility = metas[0][0]["size_divisibility"]
    nested_tensor = nested_tensor_from_tensor_list(images_list, size_divisibility=size_divisibility)
    # Reshape the nested tensor:
    nested_tensor.tensors = einops.rearrange(
        nested_tensor.tensors, "(b t) c h w -> b t c h w", b=_B, t=_T
    )
    nested_tensor.mask = einops.rearrange(
        nested_tensor.mask, "(b t) h w -> b t h w", b=_B, t=_T
    )
    # Above is prepared for DETR.
    # Below is prepared for MOTIP, pre-padding the annotations:
    max_N = max(annotation[0]["trajectory_id_labels"].shape[-1] for annotation in annotations)
    # Padding the ID annotations:
    for b in range(len(annotations)):
        for t in range(len(annotations[b])):
            _G, _, _N = annotations[b][t]["trajectory_id_labels"].shape
            if _N < max_N:
                annotations[b][t]["trajectory_id_labels"] = torch.cat([
                    annotations[b][t]["trajectory_id_labels"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["trajectory_id_masks"] = torch.cat([
                    annotations[b][t]["trajectory_id_masks"],
                    torch.ones((_G, 1, max_N - _N), dtype=torch.bool)
                ], dim=-1)
                annotations[b][t]["trajectory_ann_idxs"] = torch.cat([
                    annotations[b][t]["trajectory_ann_idxs"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["trajectory_times"] = torch.cat([
                    annotations[b][t]["trajectory_times"],
                    t * torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_id_labels"] = torch.cat([
                    annotations[b][t]["unknown_id_labels"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_id_masks"] = torch.cat([
                    annotations[b][t]["unknown_id_masks"],
                    torch.ones((_G, 1, max_N - _N), dtype=torch.bool)
                ], dim=-1)
                annotations[b][t]["unknown_ann_idxs"] = torch.cat([
                    annotations[b][t]["unknown_ann_idxs"],
                    - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)
                annotations[b][t]["unknown_times"] = torch.cat([
                    annotations[b][t]["unknown_times"],
                    t * torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                ], dim=-1)

                # --- ADDED: PADDING FOR TRAJECTORY & UNKNOWN CLASS LABELS ---
                for key in ["trajectory_class_labels", "unknown_class_labels"]:
                    if key in annotations[b][t]:
                        annotations[b][t][key] = torch.cat([
                            annotations[b][t][key],
                            - torch.ones((_G, 1, max_N - _N), dtype=torch.int64)
                        ], dim=-1)
                # ------------------------------------------------------------
            pass
    return {
        "images": nested_tensor,
        "annotations": annotations,
        "metas": metas,
    }


def verify_batch_integrity(targets, num_classes=None, id_vocabulary=None, step=None):
    """
    Runtime check to ensure Multi-Class labels, IDs, and BBoxes are within expected ranges.
    Restored with original detailed error messages and diagnostic logic.
    """
    all_categories = []
    all_ids = []
    all_bboxes = []
    
    # targets is Batch -> Time -> Dict
    for b_idx, clip in enumerate(targets):
        for t_idx, ann in enumerate(clip):
            if "category" in ann and ann["category"].numel() > 0:
                # --- PADDING FILTER ---
                # We must ignore the -1 padding from collate_fn to get accurate min/max
                valid_mask = ann["id"] >= 0
                if valid_mask.any():
                    all_categories.append(ann["category"][valid_mask])
                    all_ids.append(ann["id"][valid_mask])
                    all_bboxes.append(ann["bbox"][valid_mask])

    if not all_categories:
        return  # Skip empty batches

    all_categories = torch.cat(all_categories)
    all_ids = torch.cat(all_ids)
    all_bboxes = torch.cat(all_bboxes)

    # --- NAN / INF SANITY CHECK ---
    # Catch non-finite values across labels, ids, and boxes
    if not torch.isfinite(all_categories.float()).all() or \
       not torch.isfinite(all_ids.float()).all() or \
       not torch.isfinite(all_bboxes).all():
        raise ValueError(f"❌ [DIAGNOSTIC] Step {step}: NaN or Inf detected in batch data!")

    # --- CATEGORY CHECK ---
    max_cat = all_categories.max().item()
    min_cat = all_categories.min().item()
    
    if num_classes is not None and num_classes > 0:
        if max_cat >= num_classes:
            raise ValueError(
                f"❌ [DIAGNOSTIC] Step {step}: Category Index Error: Found category {max_cat}, "
                f"but num_classes is {num_classes}. (Check 0-indexing!)"
            )
    
    if min_cat < 0:
        raise ValueError(f"❌ [DIAGNOSTIC] Step {step}: Negative Category found: {min_cat}")

    # --- ID RANGE CHECK ---
    max_id = all_ids.max().item()
    if id_vocabulary is not None and id_vocabulary > 0:
        # MOTIP uses 'id_vocabulary' as the specific index for Newborns.
        # The embedding matrix size is vocabulary + 1, so max_id can be EQUAL to vocabulary.
        if max_id > id_vocabulary:
            raise ValueError(
                f"❌ [DIAGNOSTIC] Step {step}: ID Overflow: Found ID {max_id}, but "
                f"vocabulary limit is {id_vocabulary}. (Only IDs > {id_vocabulary} are illegal)"
            )
    
    # --- BOUNDING BOX "GIANT BOX" CHECK ---
    # DETR models expect normalized coordinates in [0, 1].
    # Margin allowed for augmentation, but values > 2.0 indicate raw pixel coordinates.
    box_min = all_bboxes.min().item()
    box_max = all_bboxes.max().item()
    if box_max > 2.0 or box_min < -1.0:
        raise ValueError(
            f"❌ [DIAGNOSTIC] Step {step}: Giant Box Detected! BBox values range from {box_min:.2f} to {box_max:.2f}. "
            f"DETR requires normalized [0, 1] coordinates. Ensure you divide by width/height in dancetrack.py!"
        )

    if (all_bboxes[:, 2:] <= 0).any():
        raise ValueError(f"❌ [DIAGNOSTIC] Step {step}: Found BBox with zero or negative width/height!")

    # --- MULTI-CLASS RANGE CHECK ---
    person_ids = all_ids[all_categories == 0]
    vehicle_ids = all_ids[all_categories == 1]

    if person_ids.numel() > 0:
        p_max = person_ids.max().item()
        if p_max >= 500:
            raise ValueError(f"❌ [DIAGNOSTIC] Step {step}: Partition Violation! Person (0) has IDs in Vehicle range (>=500): {p_max}")
            
    if vehicle_ids.numel() > 0:
        v_min = vehicle_ids.min().item()
        if v_min < 500:
            raise ValueError(f"❌ [DIAGNOSTIC] Step {step}: Partition Violation! Vehicle (1) has IDs in Person range (<500): {v_min}")

    # --- BATCH COMPOSITION LOGGING ---
    should_log = (step is None) or (step % 50 == 0)
    if should_log:
        p_count = (all_categories == 0).sum().item()
        v_count = (all_categories == 1).sum().item()
        msg = f"🔍 [DEBUG Batch {step if step is not None else ''}] People={p_count} | Vehicles={v_count}"
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(msg)
        elif not torch.distributed.is_initialized():
            print(msg)


def check_categorical_balance(targets, step, log_interval=100):
    """
    Logs distribution of real objects (ignoring padding -1).
    """
    global _CUMULATIVE_COUNTS
    if step == 0:
        _CUMULATIVE_COUNTS = {0: 0, 1: 0}

    batch_categories = []
    for clip in targets:
        for ann in clip:
            # Only count categories where the ID is not -1 (not padding)
            valid_mask = ann["id"] >= 0
            if valid_mask.any():
                batch_categories.append(ann["category"][valid_mask])
    
    if not batch_categories:
        return

    batch_categories = torch.cat(batch_categories)
    
    p_batch = (batch_categories == 0).sum().item()
    v_batch = (batch_categories == 1).sum().item()
    
    _CUMULATIVE_COUNTS[0] += p_batch
    _CUMULATIVE_COUNTS[1] += v_batch

    if step % log_interval == 0:
        total = _CUMULATIVE_COUNTS[0] + _CUMULATIVE_COUNTS[1]
        p_ratio = (_CUMULATIVE_COUNTS[0] / total * 100) if total > 0 else 0
        v_ratio = (_CUMULATIVE_COUNTS[1] / total * 100) if total > 0 else 0
        
        msg = (f"📊 [CLASS BALANCE] Step {step} | Total: {total} | "
               f"People: {p_ratio:.1f}% | Vehicles: {v_ratio:.1f}%")
        
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0: print(msg)
        else:
            print(msg)