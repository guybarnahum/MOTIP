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
    ):
        super().__init__()
        self.weight = weight
        self.use_focal_loss = use_focal_loss

        if not self.use_focal_loss:
            self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        return

    def forward(self, id_logits, id_labels, id_masks, id_categories=None):
        # id_logits: [B, G, T, N, C]  (C=1000)
        # id_labels: [B, G, T, N]
        # id_masks:  [B, G, T, N]
        # id_categories: [B, G, T, N] (0=Person, 1=Vehicle)

        # Remove the first T for supervision:
        id_logits = id_logits[:, :, 1:, :, :]
        id_labels = id_labels[:, :, 1:, :]
        id_masks = id_masks[:, :, 1:, :]
        
        if id_categories is not None:
            id_categories = id_categories[:, :, 1:, :]

        # Flatten:
        id_logits_flatten = einops.rearrange(id_logits, "b g t n c -> (b g t n) c")
        id_labels_flatten = einops.rearrange(id_labels, "b g t n -> (b g t n)")
        id_masks_flatten = einops.rearrange(id_masks, "b g t n -> (b g t n)")
        
        # --- NEW: Flatten Categories ---
        id_cats_flatten = None
        if id_categories is not None:
            id_cats_flatten = einops.rearrange(id_categories, "b g t n -> (b g t n)")

        # Filter out the invalid id labels:
        valid_indices = ~id_masks_flatten
        id_logits_flatten = id_logits_flatten[valid_indices]
        id_labels_flatten = id_labels_flatten[valid_indices]
        
        # ✅ FIX 1: Apply Logit Masking for Multi-Class Partitioning
        if id_cats_flatten is not None:
            id_cats_flatten = id_cats_flatten[valid_indices]
            
            # Create a large negative mask to kill forbidden logits
            # Use -10000.0 or -1e9. Softmax will turn these into 0.0 probability.
            mask = torch.zeros_like(id_logits_flatten)
            
            # Category 0 (Person): Mask out indices 500-999
            is_person = (id_cats_flatten == 0)
            mask[is_person, 500:] = -10000.0
            
            # Category 1 (Vehicle): Mask out indices 0-499
            is_vehicle = (id_cats_flatten == 1)
            mask[is_vehicle, :500] = -10000.0
            
            # Add mask to logits before loss
            id_logits_flatten = id_logits_flatten + mask

        # Calculate the loss:
        if self.use_focal_loss:
            id_labels_flatten_one_hot = labels_to_one_hot(id_labels_flatten, class_num=id_logits_flatten.shape[-1])
            # Ensure it is a tensor and on correct device
            if not isinstance(id_labels_flatten_one_hot, torch.Tensor):
                id_labels_flatten_one_hot = torch.from_numpy(id_labels_flatten_one_hot).to(id_logits.device)
            
            loss = sigmoid_focal_loss(inputs=id_logits_flatten, targets=id_labels_flatten_one_hot).sum()
        else:
            loss = self.ce_loss(id_logits_flatten, id_labels_flatten).sum()
        
        num_ids = torch.as_tensor([len(id_logits_flatten)], dtype=torch.float, device=id_logits.device)

        if is_distributed():
            torch.distributed.all_reduce(num_ids)
        num_ids = torch.clamp(num_ids / distributed_world_size(), min=1).item()

        return loss / num_ids


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum()


def build(config: dict):
    return IDCriterion(
        weight=config["ID_LOSS_WEIGHT"],
        use_focal_loss=config["USE_FOCAL_LOSS"],
    )