# track_eval.py
import os
import yaml
import subprocess
import argparse

def run_standalone_audit(config_path, epoch_arg=None):
    # 1. Load the YAML config
    if not os.path.exists(config_path):
        print(f"❌ Error: Config not found at {config_path}")
        return

    print(f"📖 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Path Inference Logic
    # We need to find where the .txt files are. We check two main locations:
    
    # Base directory logic from train.py / submit_and_evaluate.py
    exp_name = config.get("EXP_NAME", "stability_test")
    outputs_dir = config.get("OUTPUTS_DIR")
    if outputs_dir is None:
        outputs_dir = os.path.join("./outputs/", exp_name)
    
    # Path A: Training Loop Evaluation (eval_during_train)
    # If epoch is provided via CLI, use it; otherwise, default to the last epoch in config
    target_epoch = epoch_arg if epoch_arg is not None else (config.get("EPOCHS", 1) - 1)
    training_eval_path = os.path.join(
        outputs_dir, "train", "eval_during_train", f"epoch_{target_epoch}", "tracker"
    )

    # Path B: Standalone Inference Path (evaluate/group/dataset/...)
    # This is constructed if INFERENCE_MODEL is present in config
    inference_eval_path = None
    inf_model = config.get("INFERENCE_MODEL")
    if inf_model:
        model_name = os.path.split(inf_model)[-1].rsplit('.', 1)[0]
        inference_eval_path = os.path.join(
            outputs_dir, "evaluate", 
            config.get("INFERENCE_GROUP", "default_group"), 
            config.get("INFERENCE_DATASET", "DanceTrack"),
            config.get("INFERENCE_SPLIT", "val"),
            model_name, "tracker"
        )

    # 3. The Search Cascade
    if os.path.exists(training_eval_path):
        tracker_dir = training_eval_path
        print(f"✨ Detected Training Results (Epoch {target_epoch})")
    elif inference_eval_path and os.path.exists(inference_eval_path):
        tracker_dir = inference_eval_path
        print(f"📦 Detected Standalone Inference Results")
    else:
        # Final fallback: look for a generic tracker folder in the root outputs
        fallback_path = os.path.join(outputs_dir, "tracker")
        if os.path.exists(fallback_path):
            tracker_dir = fallback_path
        else:
            print(f"❌ Error: No tracker results (.txt) found!")
            print(f"   Looked in: {training_eval_path}")
            if inference_eval_path: print(f"   And: {inference_eval_path}")
            return

    # 4. Pull Ground Truth and Class info from the YAML val_config
    val_cfg = config.get('val_config', {})
    gt_folder = val_cfg.get('GT_FOLDER', "./datasets/DanceTrack/val")
    seqmap = val_cfg.get('SEQMAP_FILE', "./datasets/DanceTrack/val_seqmap.txt")
    classes = val_cfg.get('CLASSES_TO_EVAL', ['pedestrian'])

    print(f"---------------------------------------------------")
    print(f"🚀 Launching Standalone TrackEval Audit...")
    print(f"📍 Tracker Dir:  {tracker_dir}")
    print(f"📍 GT Folder:    {gt_folder}")
    print(f"📍 Classes:      {', '.join(classes)}")
    print(f"---------------------------------------------------")

    # 5. Construct and Run the TrackEval Command
    # This runs with full system RAM now that train.py has exited.
    cmd = [
        "python", "TrackEval/scripts/run_mot_challenge.py",
        "--GT_FOLDER", gt_folder,
        "--TRACKERS_FOLDER", tracker_dir,
        "--BENCHMARK", "MOT17",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--CLASSES_TO_EVAL", *classes,
        "--USE_PARALLEL", "False",
        "--SEQMAP_FILE", seqmap,
        "--SKIP_SPLIT_FOL", "True"
    ]

    subprocess.run(cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone TrackEval Auditor")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--epoch", type=int, help="Optional: specific epoch to audit (defaults to last)")
    args = parser.parse_args()
    
    run_standalone_audit(args.config, args.epoch)