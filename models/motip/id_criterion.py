# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import is_distributed, distributed_world_size, labels_to_one_hot

class IDCriterion(nn.Module):
    def __init__(
            self,
            weight: float,
            use_focal_loss: bool,
            num_id_vocabulary: int,
            num_classes: int,
            temperature: float = 0.07
    ):
        super().__init__()
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        self.num_id_vocabulary = num_id_vocabulary
        self.num_classes = num_classes
        self.temperature = temperature
        self.partition_size = num_id_vocabulary // num_classes

        if not self.use_focal_loss:
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        
        # Diagnostics
        self.step_count = 0

    def forward(self, id_logits, id_labels, id_masks, id_categories=None):
        id_logits = id_logits[:, :, 1:, :, :]
        id_labels = id_labels[:, :, 1:, :]
        id_masks = id_masks[:, :, 1:, :]
        
        if id_categories is not None:
            id_categories = id_categories[:, :, 1:, :]

        id_logits_flatten = einops.rearrange(id_logits, "b g t n c -> (b g t n) c")
        id_labels_flatten = einops.rearrange(id_labels, "b g t n -> (b g t n)")
        id_masks_flatten = einops.rearrange(id_masks, "b g t n -> (b g t n)")
        
        id_cats_flatten = None
        if id_categories is not None:
            id_cats_flatten = einops.rearrange(id_categories, "b g t n -> (b g t n)")

        valid_indices = ~id_masks_flatten
        id_logits_flatten = id_logits_flatten[valid_indices]
        id_labels_flatten = id_labels_flatten[valid_indices]
        
        if id_logits_flatten.shape[0] == 0:
            return id_logits.sum() * 0.0

        curr_temp = self.temperature if not self.use_focal_loss else 1.0
        id_logits_flatten = id_logits_flatten / curr_temp

        if id_cats_flatten is not None:
            id_cats_flatten = id_cats_flatten[valid_indices]
            # Mapping 1/2 to 0/1
            norm_id_cats = id_cats_flatten - 1 

            # Create mask
            mask = torch.full_like(id_logits_flatten, -10000.0) 
            for cls_idx in range(self.num_classes):
                is_this_cls = (norm_id_cats == cls_idx)
                start = cls_idx * self.partition_size
                end = (cls_idx + 1) * self.partition_size
                mask[is_this_cls, start:end] = 0.0 
            
            # --- 🛡️ EMERGENCY PROBE: CHECK IF TARGET IS MASKED ---
            # Extract the mask value at the actual ground-truth label position
            target_mask_check = mask[torch.arange(len(id_labels_flatten)), id_labels_flatten]
            masked_targets = (target_mask_check < -1.0)
            
            if masked_targets.any():
                bad_idx = torch.where(masked_targets)[0][0]
                print(f"\n🔥 [CRITICAL] MASKING EXPLOSION AT STEP {self.step_count if hasattr(self, 'step_count') else '?'}")
                print(f"   ∟ Dataset Category: {id_cats_flatten[bad_idx].item()}")
                print(f"   ∟ Normalized Category: {norm_id_cats[bad_idx].item()}")
                print(f"   ∟ Ground Truth ID: {id_labels_flatten[bad_idx].item()}")
                print(f"   ∟ Partition Limit: {norm_id_cats[bad_idx].item() * self.partition_size} to {(norm_id_cats[bad_idx].item() + 1) * self.partition_size}")
                print(f"   ∟ Result: Target is currently MASKED OUT (-10000). CE Loss will explode.")
            # ----------------------------------------------------
            
            id_logits_flatten = id_logits_flatten + mask

        if self.use_focal_loss:
            id_labels_one_hot = labels_to_one_hot(id_labels_flatten, class_num=id_logits_flatten.shape[-1])
            if not isinstance(id_labels_one_hot, torch.Tensor):
                id_labels_one_hot = torch.from_numpy(id_labels_one_hot).to(id_logits.device)
            loss = sigmoid_focal_loss(inputs=id_logits_flatten, targets=id_labels_one_hot).sum()
        else:
            loss = self.ce_loss(id_logits_flatten, id_labels_flatten).sum()
        
        # --- ROBUST NORMALIZATION ---
        num_ids = torch.tensor(id_logits_flatten.shape[0], dtype=torch.float, device=id_logits.device)
        if is_distributed():
            torch.distributed.all_reduce(num_ids)
            num_ids = num_ids / distributed_world_size()

        num_ids = torch.clamp(num_ids, min=1.0)
        self.step_count += 1

        return (loss / num_ids) * self.weight

def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum(1) 

def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        use_focal_loss=config.get("USE_FOCAL_LOSS", False),
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        num_classes=config["NUM_CLASSES"],
        temperature=config.get("ID_TEMPERATURE", 0.07)
    )