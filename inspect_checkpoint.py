import torch
import os
import argparse

def inspect(path):
    print(f"🔍 Inspecting: {path}")
    if not os.path.exists(path):
        print("❌ File not found.")
        return

    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"❌ Corrupt file or load error: {e}")
        return

    if 'model_args' in ckpt:
        print("\n✅ Valid Deployment Model found!")
        args = ckpt['model_args']
        
        print("\n--- CRITICAL HYPERPARAMETERS ---")
        critical_keys = [
            'ffn_dim_ratio', 
            'id_dim', 
            'feature_dim', 
            'num_id_vocabulary', 
            'head_dim',
            'hidden_dim'
        ]
        
        for k in critical_keys:
            val = args.get(k, "⚠️ MISSING")
            print(f"{k:<25}: {val}")

        print("\n--- ALL SAVED KEYS ---")
        print(list(args.keys()))
        
    else:
        print("\n⚠️  'model_args' key NOT found.")
        print("   This looks like a raw training checkpoint, not an exported deployment model.")
        if 'model' in ckpt:
            print("   (Found 'model' state_dict keys)")
        elif isinstance(ckpt, dict):
            print(f"   (Found keys: {list(ckpt.keys())})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, default="./pretrains/motip_v2.pth", 
                        help="Path to the .pth file to inspect")
    args = parser.parse_args()
    
    inspect(args.checkpoint_path)
