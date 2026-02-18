# Copyright (c) Ruopeng Gao. All Rights Reserved.
import sys
import gc
import psutil
import torch
import torchvision
import copy
import math
import torch.nn as nn

from utils.misc import is_main_process, is_distributed

# --- DIAGNOSTIC HELPERS ---
diag_logs = False

def diag_log(msg: str):
    """Bypasses standard logging to flush messages immediately to terminal."""
    if diag_logs:
        sys.stderr.write(f"🚩 [DIAG] {msg}\n")
        sys.stderr.flush()

# --------------------------

# Several calculation functions that are used in multiple model structures:

def pos_to_pos_embed(pos, num_pos_feats: int = 64, temperature: int = 10000, scale: float = 2 * math.pi):
    pos = pos * scale
    dim_i = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_i = temperature ** (2 * (torch.div(dim_i, 2, rounding_mode="trunc")) / num_pos_feats)
    pos_embed = pos[..., None] / dim_i      # (N, M, n_feats) or (B, N, M, n_feats)
    pos_embed = torch.stack((pos_embed[..., 0::2].sin(), pos_embed[..., 1::2].cos()), dim=-1)
    pos_embed = torch.flatten(pos_embed, start_dim=-3)
    return pos_embed


def label_to_one_hot(labels: torch.Tensor, n_classes: int, dtype=torch.float32):
    one_hot = torch.eye(n=n_classes, device=labels.device, dtype=dtype)[labels]
    return one_hot


def inverse_sigmoid(x, eps=1e-5):
    """
    if      x = 1/(1+exp(-y))
    then    y = ln(x/(1-x))
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def load_detr_pretrain(model: nn.Module, pretrain_path: str, num_classes: int | None, default_class_idx: int | None = None):
    print(f"loading detr pretrain from {pretrain_path}")
    pretrain_model = torch.load(pretrain_path, map_location=lambda storage, loc: storage, weights_only=False)
    pretrain_state_dict = pretrain_model["model"]
    detr_state_dict = dict()
    model_state_dict = model.state_dict()
    
    # --- 1. Robust Prefix Handling ---
    has_detr_prefix = any(k.startswith("detr.") for k in model_state_dict.keys())

    for k, v in pretrain_state_dict.items():
        if has_detr_prefix and not k.startswith("detr."):
            new_key = "detr." + k
        else:
            new_key = k
        detr_state_dict[new_key] = v

    # --- 2. Shape Mismatch Handling ---
    for k, v in detr_state_dict.items():
        if "class_embed" in k:
            if num_classes is None:
                num_classes = len(detr_state_dict[k])
            
            if len(detr_state_dict[k]) == 91:   
                if num_classes == 1:
                    if default_class_idx is None:   
                        detr_state_dict[k] = detr_state_dict[k][1:2]
                    else:
                        detr_state_dict[k] = detr_state_dict[k][default_class_idx:default_class_idx+1]
                else:
                    print(f"⚠️  WARNING: COCO classes (91) != Model classes ({num_classes}). Resetting {k}.")
                    if k in model_state_dict:
                        detr_state_dict[k] = model_state_dict[k]
                    
            elif num_classes == len(detr_state_dict[k]):    
                pass
            
            else:
                print(f"⚠️  WARNING: Pretrain classes ({len(detr_state_dict[k])}) != Model classes ({num_classes}). Resetting {k}.")
                if k in model_state_dict:
                    detr_state_dict[k] = model_state_dict[k]
                else:
                    print(f"   ❌ Could not find {k} in model to reset it. Skipping.")

        if "label_enc" in k:
            if k in model_state_dict and len(detr_state_dict[k]) != len(model_state_dict[k]):
                if len(model_state_dict[k]) == 2:
                    try:
                        detr_state_dict[k] = torch.cat((detr_state_dict[k][1:2], detr_state_dict[k][91:92]), dim=0)
                    except:
                         detr_state_dict[k] = model_state_dict[k]
                else:
                    detr_state_dict[k] = model_state_dict[k]

    # --- 3. Final Load ---
    final_state_dict = {}
    for k, v in detr_state_dict.items():
        if k in model_state_dict:
            if v.shape != model_state_dict[k].shape:
                print(f"⚠️  Final shape mismatch for {k}: {v.shape} vs {model_state_dict[k].shape}. Resetting.")
                final_state_dict[k] = model_state_dict[k]
            else:
                final_state_dict[k] = v
        
    model.load_state_dict(state_dict=final_state_dict, strict=False)
    # Clear large pretrain dict from RAM
    del pretrain_model, pretrain_state_dict, detr_state_dict
    gc.collect()
    return


def save_checkpoint(model, path, states: dict, optimizer, scheduler, only_detr: bool = False):
    """Memory-safe checkpointing to avoid RAM/Swap explosion."""
    diag_log(f"Entering save_checkpoint. Target: {path}")

    if is_main_process():
        model_obj = get_model(model)
        if only_detr: model_obj = model_obj.detr

        diag_log("Extracting model.state_dict()...")
        m_state = model_obj.state_dict()

        # Optimizer state_dict is the main RAM spike point
        diag_log("Extracting optimizer.state_dict() [Potential Spike Point]...")
        o_state = optimizer.state_dict() if optimizer is not None else None
        
        diag_log("Assembling save_state dictionary...")
        save_state = {
            "model": m_state,
            "optimizer": o_state,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "states": states,
        }
        
        diag_log(f"Save dict assembled. RAM Usage: {psutil.virtual_memory().percent}%. Starting torch.save...")
        torch.save(save_state, path)
        
        diag_log("torch.save finished. PURGING intermediate CPU dictionaries...")
        # Clear specific references to the large CPU-side copies
        del save_state, m_state, o_state
        gc.collect()
        
        diag_log(f"Checkpoint complete. RAM recovered to: {psutil.virtual_memory().percent}%")

def load_checkpoint(model, path, states=None, optimizer=None, scheduler=None):
    load_state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=False)
    model_state = load_state["model"]

    if "bbox_embed.0.layers.0.weight" in model_state:
        load_detr_pretrain(model=model, pretrain_path=path, num_classes=None)
        return
    else:
        model.load_state_dict(model_state)

    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    if states is not None:
        states.update(load_state["states"])
    
    # Clean up load dictionary
    del load_state, model_state
    gc.collect()
    return


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_model(model):
    return model.module if is_distributed() else model

# For previous version of MOTIP models:
def load_previous_checkpoint(model, path, states=None, optimizer=None, scheduler=None):
    assert states is None and optimizer is None and scheduler is None, \
        "The states, optimizer, and scheduler should be None for previous versions."

    load_state = torch.load(path, map_location=lambda storage, loc: storage)
    model_state = load_state["model"]
    transfer_states = dict()

    if "bbox_embed.0.layers.0.weight" in model_state:
        load_detr_pretrain(model=model, pretrain_path=path, num_classes=None)
        return
    else:
        for k, v in model_state.items():
            if "detr" in k:
                transfer_states[k] = v
            elif "seq_decoder" in k:
                transfer_k = k
                transfer_k = transfer_k.replace("seq_decoder.", "")
                if "trajectory_feature_adapter" in transfer_k:
                    transfer_k = transfer_k.replace("trajectory_feature_adapter", "trajectory_modeling.adapter")
                    if "norm" in transfer_k:
                        transfer_k = transfer_k.replace("adapter.", "")
                elif "trajectory_augmentation" in transfer_k:
                    transfer_k = transfer_k.replace("trajectory_augmentation.trajectory_ffn", "trajectory_modeling.ffn")
                    if "ffn.norm" in transfer_k:
                        transfer_k = transfer_k.replace("ffn.norm", "ffn_norm")
                elif "related_temporal_embeds" in transfer_k:
                    transfer_k = transfer_k.replace("related_temporal_embeds", "rel_pos_embeds")
                elif "embed_to_word" in transfer_k:
                    for _ in range(0, 6):
                        _transfer_k = transfer_k
                        _transfer_k = _transfer_k.replace("embed_to_word", f"embed_to_word_layers.{_}")
                        transfer_states[_transfer_k] = v
                    continue
                elif "decoder_layers" in transfer_k:
                    transfer_k = transfer_k.replace("decoder_layers", "cross_attn_layers")
                elif ".norm_layers" in transfer_k:
                    transfer_k = transfer_k.replace("norm_layers", "cross_attn_norm_layers")
                elif "self_attn_layers" in transfer_k:
                    pass
                elif "self_norm_layers" in transfer_k:
                    transfer_k = transfer_k.replace("self_norm_layers", "self_attn_norm_layers")
                elif "ffn_layers" in transfer_k:
                    if "norm" in transfer_k:
                        transfer_k = transfer_k.replace("ffn_layers", "ffn_norm_layers")
                        transfer_k = transfer_k.replace("norm.", "")
                transfer_states[transfer_k] = v
        model.load_state_dict(transfer_states)

    if optimizer is not None:
        optimizer.load_state_dict(load_state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(load_state["scheduler"])
    if states is not None:
        states.update(load_state["states"])
    
    del load_state, model_state, transfer_states
    gc.collect()
    return