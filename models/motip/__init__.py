# Copyright (c) Ruopeng Gao. All Rights Reserved.

from .motip import MOTIP
from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder


def build(config: dict):
    # Generate DETR args with safe defaults for Inference
    detr_args = Args()
    
    # 1. Backbone & Optimization
    detr_args.backbone = config.get("BACKBONE", "resnet50")
    lr = config.get("LR", 0.0001)
    scale = config.get("LR_BACKBONE_SCALE", 0.1)
    detr_args.lr_backbone = lr * scale
    detr_args.dilation = config.get("DILATION", False)
    
    # 2. Transformer Architecture
    detr_args.num_classes = config.get("NUM_CLASSES", 1)
    detr_args.device = config.get("DEVICE", "cuda")
    
    # Robust Key Mapping
    detr_args.num_queries = config.get("DETR_NUM_QUERIES", config.get("NUM_QUERIES", 300))
    detr_args.num_feature_levels = config.get("DETR_NUM_FEATURE_LEVELS", config.get("NUM_FEATURE_LEVELS", 4))
    detr_args.aux_loss = config.get("DETR_AUX_LOSS", config.get("AUX_LOSS", True))
    detr_args.with_box_refine = config.get("DETR_WITH_BOX_REFINE", config.get("WITH_BOX_REFINE", True))
    detr_args.two_stage = config.get("DETR_TWO_STAGE", config.get("TWO_STAGE", False))
    detr_args.hidden_dim = config.get("DETR_HIDDEN_DIM", config.get("HIDDEN_DIM", 256))
    detr_args.masks = config.get("DETR_MASKS", config.get("MASKS", False))
    detr_args.position_embedding = config.get("DETR_POSITION_EMBEDDING", config.get("POSITION_EMBEDDING", "sine"))
    detr_args.nheads = config.get("DETR_NUM_HEADS", config.get("NHEADS", 8))
    detr_args.enc_layers = config.get("DETR_ENC_LAYERS", config.get("ENC_LAYERS", 6))
    detr_args.dec_layers = config.get("DETR_DEC_LAYERS", config.get("DEC_LAYERS", 6))
    detr_args.dim_feedforward = config.get("DETR_DIM_FEEDFORWARD", config.get("DIM_FEEDFORWARD", 1024))
    detr_args.dropout = config.get("DETR_DROPOUT", config.get("DROPOUT", 0.1))
    detr_args.dec_n_points = config.get("DETR_DEC_N_POINTS", config.get("DEC_N_POINTS", 4))
    detr_args.enc_n_points = config.get("DETR_ENC_N_POINTS", config.get("ENC_N_POINTS", 4))

    # 3. Loss Coefficients
    detr_args.cls_loss_coef = config.get("DETR_CLS_LOSS_COEF", 2.0)
    detr_args.bbox_loss_coef = config.get("DETR_BBOX_LOSS_COEF", 5.0)
    detr_args.giou_loss_coef = config.get("DETR_GIOU_LOSS_COEF", 2.0)
    detr_args.focal_alpha = config.get("DETR_FOCAL_ALPHA", 0.25)
    detr_args.set_cost_class = config.get("DETR_SET_COST_CLASS", 2.0)
    detr_args.set_cost_bbox = config.get("DETR_SET_COST_BBOX", 5.0)
    detr_args.set_cost_giou = config.get("DETR_SET_COST_GIOU", 2.0)

    # Framework Selection
    detr_framework = config.get("DETR_FRAMEWORK", "deformable_detr").lower()
    match detr_framework:
        case "deformable_detr":
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case _:
            raise NotImplementedError(f"DETR framework {detr_framework} is not supported.")

    # Build Components with Defaults
    only_detr = config.get("ONLY_DETR", False)
    
    # 1. Trajectory Modeling
    _trajectory_modeling = None
    if not only_detr:
        _trajectory_modeling = TrajectoryModeling(
            detr_dim=detr_args.hidden_dim,
            # FIX: Change 4.0 (float) to 4 (int)
            ffn_dim_ratio=config.get("FFN_DIM_RATIO", 4),
            feature_dim=int(config.get("FEATURE_DIM", 256)),
        )
        
    # 2. ID Decoder
    _id_decoder = None
    if not only_detr:
        _id_decoder = IDDecoder(
            feature_dim=int(config.get("FEATURE_DIM", 256)),
            id_dim=config.get("ID_DIM", 128),
            # FIX: Change 4.0 (float) to 4 (int)
            ffn_dim_ratio=config.get("FFN_DIM_RATIO", 4),
            num_layers=config.get("NUM_ID_DECODER_LAYERS", 6),
            head_dim=config.get("HEAD_DIM", 256),
            num_id_vocabulary=config.get("NUM_ID_VOCABULARY", 1000),
            rel_pe_length=config.get("REL_PE_LENGTH", 50),
            use_aux_loss=config.get("USE_AUX_LOSS", True),
            use_shared_aux_head=config.get("USE_SHARED_AUX_HEAD", True),
        )

    # Construct MOTIP model
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=only_detr,
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
    )

    return motip_model, detr_criterion