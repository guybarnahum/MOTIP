#!/usr/bin/env python3
import toml
import subprocess
import sys
import os

def run_command(cmd):
    """Prints and executes a shell command."""
    print(f"\n🚀 Running: {' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Execution failed with error code {e.returncode}")
        sys.exit(e.returncode)

def main():
    if not os.path.exists("config.toml"):
        print("❌ config.toml not found!")
        sys.exit(1)

    cfg = toml.load("config.toml")
    
    # Extract config sections
    sys_cfg = cfg.get("system", {})
    data_cfg = cfg.get("data", {})
    exp_cfg = cfg.get("experiment", {})
    ckpt_cfg = cfg.get("checkpoint", {})

    mode = sys_cfg.get("mode", "train")
    num_proc = str(sys_cfg.get("num_processes", 1))
    
    # Base command: accelerate launch
    cmd = ["accelerate", "launch", "--num_processes", num_proc]

    # Select script based on mode
    if mode in ["train", "pretrain"]:
        script = "train.py"
    elif mode in ["submit", "evaluate"]:
        script = "submit_and_evaluate.py"
    else:
        print(f"❌ Unknown mode: {mode}")
        sys.exit(1)
        
    cmd.append(script)

    # Common arguments
    cmd.extend(["--data-root", data_cfg.get("root", "./datasets")])
    cmd.extend(["--config-path", exp_cfg.get("config_path", "")])
    
    # Mode specific arguments
    if mode == "train":
        cmd.extend(["--exp-name", exp_cfg.get("name", "default_exp")])
        # Add resume args if needed manually here, or handled by configs
        
    elif mode == "pretrain":
        cmd.extend(["--exp-name", exp_cfg.get("name", "default_pretrain")])
        
    elif mode in ["submit", "evaluate"]:
        cmd.extend(["--inference-mode", mode])
        cmd.extend(["--inference-dataset", data_cfg.get("dataset", "DanceTrack")])
        cmd.extend(["--inference-split", data_cfg.get("split", "test")])
        
        # Output directory usually matches the experiment name in outputs
        output_dir = os.path.join("./outputs", exp_cfg.get("name", "output"))
        cmd.extend(["--outputs-dir", output_dir])
        
        # Checkpoint is required for inference
        ckpt_path = ckpt_cfg.get("path", "")
        if not ckpt_path:
            print("❌ Error: 'checkpoint.path' in config.toml is required for inference.")
            sys.exit(1)
        cmd.extend(["--inference-model", ckpt_path])
        
        # FP16 Optimization
        if sys_cfg.get("dtype") == "fp16":
            cmd.extend(["--inference-dtype", "FP16"])

    # Run it
    run_command(cmd)

if __name__ == "__main__":
    main()
