import torch
import yaml
import os
import argparse
import sys

# Ensure your project root is in path so 'models' can be imported
sys.path.append(os.getcwd())

# Import your model builder
try:
    from models.motip import build as build_model
except ImportError:
    print("❌ Error: Could not import 'models.motip'. Run this script from the project root.")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Export MOTIP model for config-free inference.")
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help="Path to the training checkpoint (e.g., outputs/stage2_v2/checkpoint.pth)")
    parser.add_argument('--config', type=str, required=True, 
                        help="Path to the training config (e.g., configs/stage2_v2.yaml)")
    parser.add_argument('--output', type=str, default='motip_deploy.pth', 
                        help="Path to save the deployment model (default: motip_deploy.pth)")
    return parser.parse_args()

def export_deployment_model(args):
    # 1. Load Configuration
    print(f"📖 Loading configuration from {args.config}...")
    if not os.path.exists(args.config):
        print(f"❌ Error: Config file not found at {args.config}")
        sys.exit(1)
        
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Handle parent configs (SUPER_CONFIG_PATH) logic
    if "SUPER_CONFIG_PATH" in cfg:
        config_dir = os.path.dirname(args.config)
        super_path = os.path.join(config_dir, cfg["SUPER_CONFIG_PATH"])
        if not os.path.exists(super_path):
             super_path = cfg["SUPER_CONFIG_PATH"]
             
        if os.path.exists(super_path):
            print(f"   ↳ Merging parent config: {super_path}")
            with open(super_path, 'r') as f_base:
                base_cfg = yaml.safe_load(f_base)
            base_cfg.update(cfg)
            cfg = base_cfg
        else:
            print(f"⚠️ Warning: SUPER_CONFIG_PATH defined but file not found: {super_path}")

    # 2. Build the model architecture (Empty shell)
    print("🏗️  Building model architecture...")
    
    # --- FIX: Robust Unpacking ---
    # Captures return values regardless of count (2 or 3)
    build_result = build_model(cfg)
    
    if isinstance(build_result, tuple):
        model = build_result[0]  # Model is always the first element
    else:
        model = build_result
    # -----------------------------
    
    # 3. Load the heavy training weights
    print(f"⚖️  Loading weights from {args.checkpoint}...")
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)
        
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Standardize state dict loading
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # Clean up keys if necessary (e.g. remove 'module.' prefix from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"⚠️  Missing keys: {len(missing)}")
    if unexpected:
        print(f"⚠️  Unexpected keys: {len(unexpected)}")

    model.eval()

    # 4. Create the "Deployment Payload"
    deploy_payload = {
        'model_state_dict': model.state_dict(),
        'model_args': {
            # --- DETR / Transformer Args ---
            'num_classes': cfg.get('NUM_CLASSES', 2),
            'hidden_dim': cfg.get('HIDDEN_DIM', 256),
            'nheads': cfg.get('NHEADS', 8),
            'num_encoder_layers': cfg.get('ENC_LAYERS', 6),
            'num_decoder_layers': cfg.get('DEC_LAYERS', 6),
            'dim_feedforward': cfg.get('DIM_FEEDFORWARD', 1024),
            'dropout': 0.0, 
            'activation': cfg.get('ACTIVATION', 'relu'),
            'num_feature_levels': cfg.get('NUM_FEATURE_LEVELS', 4),
            'dec_n_points': cfg.get('DEC_N_POINTS', 4),
            'enc_n_points': cfg.get('ENC_N_POINTS', 4),
            'two_stage': cfg.get('TWO_STAGE', False),
            'num_queries': cfg.get('NUM_QUERIES', 300),
            'aux_loss': False,
            'with_box_refine': cfg.get('WITH_BOX_REFINE', True),
            
            # --- MOTIP Specific Args (CRITICAL FIX) ---
            # These are required to match the checkpoint architecture exactly
            'feature_dim': int(cfg.get('FEATURE_DIM', 256)),
            'id_dim': int(cfg.get('ID_DIM', 128)),
            'ffn_dim_ratio': int(cfg.get('FFN_DIM_RATIO', 4)),
            'num_id_decoder_layers': int(cfg.get('NUM_ID_DECODER_LAYERS', 6)),
            'head_dim': int(cfg.get('HEAD_DIM', 256)),
            'num_id_vocabulary': int(cfg.get('NUM_ID_VOCABULARY', 1000)),
            'rel_pe_length': int(cfg.get('REL_PE_LENGTH', 50)),
            'use_aux_loss': cfg.get('USE_AUX_LOSS', True),
            'use_shared_aux_head': cfg.get('USE_SHARED_AUX_HEAD', True),
        }
    }

    # 5. Save
    print(f"💾 Saving lightweight model to {args.output}...")
    torch.save(deploy_payload, args.output)
    
    # Stats
    original_size = os.path.getsize(args.checkpoint) / (1024*1024)
    new_size = os.path.getsize(args.output) / (1024*1024)
    print(f"📉 Size reduced from {original_size:.2f} MB to {new_size:.2f} MB")
    print("✅ Ready for config-free inference!")

if __name__ == "__main__":
    args = parse_args()
    export_deployment_model(args)