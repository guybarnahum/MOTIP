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
        # 1. PREPARE TENSORS (Standard MOTIP logic)
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

        # 2. TEMPERATURE SCALING
        curr_temp = self.temperature if not self.use_focal_loss else 1.0
        id_logits_flatten = id_logits_flatten / curr_temp

        # 3. MULTI-CLASS MODULO MAPPING & MASKING
        if id_cats_flatten is not None:
            id_cats_flatten = id_cats_flatten[valid_indices]
            norm_id_cats = id_cats_flatten # Data is 0/1, no offset needed

            # --- FIX 1: ID MODULO MAPPING ---
            # This handles IDs > 500 by wrapping them into the class partition.
            # Newborn (ID 1000) is kept as is.
            newborn_mask = (id_labels_flatten == self.num_id_vocabulary)
            id_labels_flatten = torch.where(
                newborn_mask,
                id_labels_flatten,
                (id_labels_flatten % self.partition_size) + (norm_id_cats * self.partition_size)
            )

            # --- FIX 2: APPLY PARTITION MASK ---
            mask = torch.full_like(id_logits_flatten, -10000.0) 
            
            # Protect Newborn Slot: Index 1000 is always allowed for everyone
            if id_logits_flatten.shape[-1] > self.num_id_vocabulary:
                mask[:, self.num_id_vocabulary] = 0.0

            for cls_idx in range(self.num_classes):
                is_this_cls = (norm_id_cats == cls_idx)
                start = cls_idx * self.partition_size
                end = (cls_idx + 1) * self.partition_size
                mask[is_this_cls, start:end] = 0.0 
            
            # --- 🛡️ DIAGNOSTIC PROBE (Retention Recommended) ---
            # Check if the target label is safe after modulo mapping
            target_mask_check = mask[torch.arange(len(id_labels_flatten)), id_labels_flatten]
            if (target_mask_check < -1.0).any():
                bad_idx = torch.where(target_mask_check < -1.0)[0][0]
                print(f"\n🔥 [CRITICAL] MODULO FAILURE AT STEP {getattr(self, 'step_count', '?')}")
                print(f"   ∟ Class: {norm_id_cats[bad_idx].item()} | Mapped ID: {id_labels_flatten[bad_idx].item()}")
                print(f"   ∟ Result: Target is STILL MASKED. CE Loss will explode.")
            
            id_logits_flatten = id_logits_flatten + mask

        # 4. CALCULATE FINAL LOSS
        if self.use_focal_loss:
            id_labels_one_hot = labels_to_one_hot(id_labels_flatten, class_num=id_logits_flatten.shape[-1])
            if not isinstance(id_labels_one_hot, torch.Tensor):
                id_labels_one_hot = torch.from_numpy(id_labels_one_hot).to(id_logits.device)
            loss = sigmoid_focal_loss(inputs=id_logits_flatten, targets=id_labels_one_hot).sum()
        else:
            loss = self.ce_loss(id_logits_flatten, id_labels_flatten).sum()
        
        # 5. ROBUST NORMALIZATION
        num_ids = torch.tensor(id_logits_flatten.shape[0], dtype=torch.float, device=id_logits.device)
        if is_distributed():
            torch.distributed.all_reduce(num_ids)
            num_ids = num_ids / distributed_world_size()

        num_ids = torch.clamp(num_ids, min=1.0)
        
        if hasattr(self, 'step_count'):
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