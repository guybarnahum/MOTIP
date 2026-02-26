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
            contrastive_weight: float = 0.5,
            temperature: float = 0.07
    ):
        super().__init__()
        self.weight = weight
        self.use_focal_loss = use_focal_loss
        self.num_id_vocabulary = num_id_vocabulary
        self.num_classes = num_classes
        self.temperature = temperature
        
        # Determine how many slots each class gets
        self.partition_size = num_id_vocabulary // num_classes

        if not self.use_focal_loss:
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        return


    def forward(self, id_logits, id_labels, id_masks, id_categories=None):
        """
        Fixed: Class Index Mapping and Numerical Stability.
        """
        # Remove the first T for supervision
        id_logits = id_logits[:, :, 1:, :, :]
        id_labels = id_labels[:, :, 1:, :]
        id_masks = id_masks[:, :, 1:, :]
        
        if id_categories is not None:
            id_categories = id_categories[:, :, 1:, :]

        # Flatten for loss calculation
        id_logits_flatten = einops.rearrange(id_logits, "b g t n c -> (b g t n) c")
        id_labels_flatten = einops.rearrange(id_labels, "b g t n -> (b g t n)")
        id_masks_flatten = einops.rearrange(id_masks, "b g t n -> (b g t n)")
        
        id_cats_flatten = None
        if id_categories is not None:
            id_cats_flatten = einops.rearrange(id_categories, "b g t n -> (b g t n)")

        # Filter out invalid labels
        valid_indices = ~id_masks_flatten
        id_logits_flatten = id_logits_flatten[valid_indices]
        id_labels_flatten = id_labels_flatten[valid_indices]
        
        if id_logits_flatten.shape[0] == 0:
            return id_logits.sum() * 0.0

        # Temperature Scaling
        curr_temp = self.temperature if not self.use_focal_loss else 1.0
        id_logits_flatten = id_logits_flatten / curr_temp

        # ✅ FIXED: MULTI-CLASS PARTITION MASKING
        if id_cats_flatten is not None:
            id_cats_flatten = id_cats_flatten[valid_indices]
            
            # 1. Map dataset classes (1, 2) to loop indices (0, 1)
            # This ensures id_cats_flatten matches the cls_idx in the loop.
            id_cats_flatten = id_cats_flatten - 1 

            # 2. Use a large negative constant for stability
            mask = torch.full_like(id_logits_flatten, -10000.0) 
            
            for cls_idx in range(self.num_classes):
                # Now cls_idx (0, 1) correctly matches the mapped categories
                is_this_cls = (id_cats_flatten == cls_idx)
                start = cls_idx * self.partition_size
                end = (cls_idx + 1) * self.partition_size
                mask[is_this_cls, start:end] = 0.0 
            
            id_logits_flatten = id_logits_flatten + mask

        # 3. CALCULATE LOSS
        if self.use_focal_loss:
            id_labels_one_hot = labels_to_one_hot(id_labels_flatten, class_num=id_logits_flatten.shape[-1])
            if not isinstance(id_labels_one_hot, torch.Tensor):
                id_labels_one_hot = torch.from_numpy(id_labels_one_hot).to(id_logits.device)
            
            loss = sigmoid_focal_loss(inputs=id_logits_flatten, targets=id_labels_one_hot).sum()
        else:
            loss = self.ce_loss(id_logits_flatten, id_labels_flatten).sum()
        
        num_ids = torch.as_tensor([len(id_logits_flatten)], dtype=torch.float, device=id_logits.device)
        if is_distributed():
            torch.distributed.all_reduce(num_ids)
        num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()

        return (loss / num_ids) * self.weight


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # ✅ Fix 1 continued: Return sum over classes for each object
    return loss.sum(1) 


def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        use_focal_loss=config.get("USE_FOCAL_LOSS", False),
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        num_classes=config["NUM_CLASSES"],
        temperature=config.get("ID_TEMPERATURE", 0.07)
    )