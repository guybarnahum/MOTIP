# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: Submit or evaluate the model.

import os
import json
import time
import torch
import subprocess
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader

from runtime_option import runtime_option
from utils.misc import yaml_to_dict
from configs.util import load_super_config, update_config
from log.logger import Logger
from data.joint_dataset import dataset_classes
from data.seq_dataset import SeqDataset
from models.runtime_tracker import RuntimeTracker
from log.log import Metrics
from models.motip import build as build_motip
from models.misc import load_checkpoint

# --- Import Memory Manager ---
try:
    from models.longterm_memory import LongTermMemory
except ImportError:
    print("⚠️ Warning: models/longterm_memory.py not found. LongTermMemory disabled.")
    LongTermMemory = None


def submit_and_evaluate(config: dict):
    # Init Accelerator at beginning:
    accelerator = Accelerator()
    state = PartialState()

    mode = config["INFERENCE_MODE"]
    assert mode in ["submit", "evaluate"], f"Mode {mode} is not supported."
    # Generate the output dir:
    assert "OUTPUTS_DIR" in config and config["OUTPUTS_DIR"] is not None, "OUTPUTS_DIR is not set."
    outputs_dir = config["OUTPUTS_DIR"]
    inference_group = config["INFERENCE_GROUP"]
    inference_dataset = config["INFERENCE_DATASET"]
    inference_split = config["INFERENCE_SPLIT"]
    inference_model = config["INFERENCE_MODEL"]
    _inference_model_name = os.path.split(inference_model)[-1][:-4]
    outputs_dir = os.path.join(
        outputs_dir, mode, inference_group, inference_dataset, inference_split, _inference_model_name
    )
    _is_outputs_dir_exist = os.path.exists(outputs_dir)
    accelerator.wait_for_everyone()
    os.makedirs(outputs_dir, exist_ok=True)

    # Init Logger, do not use wandb:
    logger = Logger(
        logdir=str(outputs_dir),
        use_wandb=False,
        config=config,
        # exp_owner=config["EXP_OWNER"],
        # exp_project=config["EXP_PROJECT"],
        # exp_group=config["EXP_GROUP"],
        # exp_name=config["EXP_NAME"],
    )
    # Log runtime config:
    logger.config(config=config)
    # Log other infos:
    logger.info(
        f"{mode.capitalize()} model: {inference_model}, inference dataset: {inference_dataset}, "
        f"inference split: {inference_split}, inference group: {inference_group}."
    )
    if _is_outputs_dir_exist:
        logger.warning(f"Outputs dir '{outputs_dir}' already exists, may overwrite the existing files.")
        time.sleep(5)   # wait for 5 seconds, give the user a chance to cancel.
    else:
        logger.info(f"Outputs dir '{outputs_dir}' created.")

    model, _ = build_motip(config=config)

    use_previous_checkpoint = config.get("USE_PREVIOUS_CHECKPOINT", False)
    if not use_previous_checkpoint:
        load_checkpoint(model, path=config["INFERENCE_MODEL"])
    else:
        from models.misc import load_previous_checkpoint
        load_previous_checkpoint(model, path=config["INFERENCE_MODEL"])

    model = accelerator.prepare(model)

    metrics = submit_and_evaluate_one_model(
        is_evaluate=config["INFERENCE_MODE"] == "evaluate",
        accelerator=accelerator,
        state=state,
        logger=logger,
        model=model,
        data_root=config["DATA_ROOT"],
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=outputs_dir,
        image_max_longer=config["INFERENCE_MAX_LONGER"],    # the max shorter side of the image is set to 800 by default
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
        use_sigmoid=config.get("USE_FOCAL_LOSS", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=config["MISS_TOLERANCE"],
        det_thresh=config["DET_THRESH"],
        newborn_thresh=config["NEWBORN_THRESH"],
        id_thresh=config["ID_THRESH"],
        area_thresh=config.get("AREA_THRESH", 0),
        inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
        else config["ONLY_DETR"],
        dtype=config.get("INFERENCE_DTYPE", "FP32"),
    )

    if metrics is not None:
        metrics.sync()
        logger.metrics(
            log=f"Finish evaluation for model '{inference_model}', dataset '{inference_dataset}', "
                f"split '{inference_split}', group '{inference_group}': ",
            metrics=metrics,
            fmt="{global_average:.4f}",
        )
    return

def submit_and_evaluate_one_model(
        is_evaluate: bool,
        accelerator: Accelerator,
        state: PartialState,
        logger: Logger,
        model,
        data_root: str,
        dataset: str,
        data_split: str,
        # Outputs:
        outputs_dir: str,
        # Parameters with defaults:
        val_config: dict = None,
        image_max_shorter: int = 800,
        image_max_longer: int = 1536,
        size_divisibility: int = 0,
        use_sigmoid: bool = False,
        assignment_protocol: str = "hungarian",
        miss_tolerance: int = 30,
        det_thresh: float = 0.5,
        newborn_thresh: float = 0.5,
        id_thresh: float = 0.1,
        area_thresh: int = 0,
        inference_only_detr: bool = False,
        dtype: str = "FP32",
):
    # Build the datasets:
    inference_dataset = dataset_classes[dataset](
        data_root=data_root,
        split=data_split,
        load_annotation=False,
    )
    # Set the dtype during inference:
    match dtype:
        case "FP32": dtype=torch.float32
        case "FP16": dtype=torch.float16
        case _: raise ValueError(f"Unknown dtype '{dtype}'.")
    # Filter out the sequences that will not be processed in this GPU (if we have multiple GPUs):
    _inference_sequence_names = list(inference_dataset.sequence_infos.keys())
    _inference_sequence_names.sort()
    # If we have multiple GPUs, we need to filter out the sequences that will not be processed in this GPU:
    if len(_inference_sequence_names) <= state.process_index:
        logger.info(
            log=f"Number of sequences is smaller than the number of processes, "
                f"a fake sequence will be processed on process {state.process_index}.",
            only_main=False,
        )
        inference_dataset.sequence_infos = {
            _inference_sequence_names[0]: inference_dataset.sequence_infos[_inference_sequence_names[0]]
        }
        inference_dataset.image_paths = {
            _inference_sequence_names[0]: inference_dataset.image_paths[_inference_sequence_names[0]]
        }
        is_fake = True
    else:
        for _ in range(len(_inference_sequence_names)):
            if _ % state.num_processes != state.process_index:
                inference_dataset.sequence_infos.pop(_inference_sequence_names[_])
                inference_dataset.image_paths.pop(_inference_sequence_names[_])
        is_fake = False

    # Process each sequence:
    for sequence_name in inference_dataset.sequence_infos.keys():
        sequence_dataset = SeqDataset(
            seq_info=inference_dataset.sequence_infos[sequence_name],
            image_paths=inference_dataset.image_paths[sequence_name],
            max_shorter=image_max_shorter,
            max_longer=image_max_longer,
            size_divisibility=size_divisibility,
            dtype=dtype,
        )
        sequence_loader = DataLoader(
            dataset=sequence_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: x[0],
        )
        sequence_wh = sequence_dataset.seq_hw()
        runtime_tracker = RuntimeTracker(
            model=model,
            sequence_hw=sequence_wh,
            use_sigmoid=use_sigmoid,
            assignment_protocol=assignment_protocol,
            miss_tolerance=miss_tolerance,
            det_thresh=det_thresh,
            newborn_thresh=newborn_thresh,
            id_thresh=id_thresh,
            area_thresh=area_thresh,
            only_detr=inference_only_detr,
            dtype=dtype,
        )

        # --- Initialize LongTermMemory for this sequence ---
        memory = None
        if LongTermMemory is not None:
            # You can adjust patience/thresholds here if needed
            memory = LongTermMemory(patience=900, gallery_size=5, similarity_thresh=0.85)

        if is_fake:
            logger.info(
                f"Fake submitting sequence {sequence_name} with {len(sequence_loader)} frames.",
                only_main=False
            )
        else:
            logger.info(f"Submitting sequence {sequence_name} with {len(sequence_loader)} frames.", only_main=False)
        
        # Pass memory to the processor
        sequence_results, sequence_fps = get_results_of_one_sequence(
            runtime_tracker=runtime_tracker,
            sequence_loader=sequence_loader,
            logger=logger,
            memory=memory 
        )
        # Write the results to the submit file:
        if dataset in ["DanceTrack", "SportsMOT", "MOT17", "PersonPath22_Inference", "BFT"]:
            sequence_tracker_results = []
            for t in range(len(sequence_results)):
                for obj_id, score, category, bbox in zip(
                        sequence_results[t]["id"],
                        sequence_results[t]["score"],
                        sequence_results[t]["category"],
                        sequence_results[t]["bbox"],    # [x, y, w, h]
                ):
                    sequence_tracker_results.append(
                        f"{t + 1},{obj_id.item()},"
                        f"{bbox[0].item()},{bbox[1].item()},{bbox[2].item()},{bbox[3].item()},"
                        f"1,-1,-1,-1\n"
                    )
            if not is_fake:
                os.makedirs(os.path.join(outputs_dir, "tracker"), exist_ok=True)
                with open(os.path.join(outputs_dir, "tracker", f"{sequence_name}.txt"), "w") as submit_file:
                    submit_file.writelines(sequence_tracker_results)
                logger.success(f"Submit sequence {sequence_name} done, FPS: {sequence_fps:.2f}. "
                               f"Saved to {os.path.join(outputs_dir, 'tracker', f'{sequence_name}.txt')}.",
                               only_main=False)
            else:
                logger.success(f"Fake submit sequence {sequence_name} done, FPS: {sequence_fps:.2f}.", only_main=False)
        else:
            raise NotImplementedError(f"Do not support to submit the results for dataset '{dataset}'.")

    # Post-process for submitting and evaluation:
    accelerator.wait_for_everyone()
    if not is_evaluate:
        logger.success(
            log=f"Submit done. Saved to {os.path.join(outputs_dir, 'tracker')}",
            only_main=True,
        )
        return None
    else:
        if accelerator.is_main_process:
            logger.info(log=f"Start evaluation...", only_main=True)
            
# --- EVALUATION CONFIGURATION LOGIC ---
            tracker_dir = os.path.join(outputs_dir, "tracker")
            
            # Default Settings
            gt_dir = os.path.join(data_root, dataset, data_split)
            seqmap_file = os.path.join(data_root, dataset, f"{data_split}_seqmap.txt")
            benchmark = "MOT17"
            classes_to_eval = None # Default: Let TrackEval decide (usually pedestrian)

            # Override with val_config if present
            if val_config is not None:
                if "GT_FOLDER" in val_config: gt_dir = val_config["GT_FOLDER"]
                if "SEQMAP_FILE" in val_config: seqmap_file = val_config["SEQMAP_FILE"]
                if "BENCHMARK" in val_config: benchmark = val_config["BENCHMARK"]
                if "CLASSES_TO_EVAL" in val_config: classes_to_eval = val_config["CLASSES_TO_EVAL"]

            # Special case for PersonPath22 defaults
            if dataset == "PersonPath22_Inference":
                gt_dir = os.path.join(data_root, dataset, "gts", "person_path_22-test")
                benchmark = "person_path_22"
                seqmap_file = os.path.join(data_root, dataset, "gts", "seqmaps", "person_path_22-test.txt")

            # Construct Arguments
            args = {
                "--SPLIT_TO_EVAL": data_split,
                "--METRICS": ["HOTA", "CLEAR", "Identity"],
                "--GT_FOLDER": gt_dir,
                "--SEQMAP_FILE": seqmap_file,
                "--SKIP_SPLIT_FOL": "True",
                "--TRACKERS_TO_EVAL": "",
                "--TRACKER_SUB_FOLDER": "",
                "--USE_PARALLEL": "True",
                "--NUM_PARALLEL_CORES": "8",
                "--PLOT_CURVES": "False",
                "--TRACKERS_FOLDER": tracker_dir,
                "--BENCHMARK": benchmark,
            }

            if classes_to_eval is not None:
                args["--CLASSES_TO_EVAL"] = classes_to_eval

            # Decide which script to run
            if dataset == "PersonPath22_Inference":
                cmd = ["python", "TrackEval/scripts/run_person_path_22.py"]
            else:
                cmd = ["python", "TrackEval/scripts/run_mot_challenge.py"]

            # Append args to command
            for k, v in args.items():
                cmd.append(k)
                if isinstance(v, list):
                    cmd += v
                else:
                    cmd.append(v)
            
            # Clone current environment
            eval_env = os.environ.copy()
            
            # Check config for custom mapping and inject as JSON string
            if val_config and "CLASS_NAME_TO_ID" in val_config:
                if accelerator.is_main_process:
                    logger.info(f"Passing custom class mapping to TrackEval: {val_config['CLASS_NAME_TO_ID']}", only_main=True)
                eval_env["TRACKEVAL_CLASS_MAP"] = json.dumps(val_config["CLASS_NAME_TO_ID"])

            # Execute TrackEval with the new environment
            _ = subprocess.run(cmd, env=eval_env)
            
            if _.returncode == 0:
                logger.success("Evaluation script is done.", only_main=True)
            else:
                raise RuntimeError("Evaluation script failed.")

        # Wait for all processes:
        accelerator.wait_for_everyone()

        # Get the metrics:
        metrics = Metrics()
        
        # Determine which summary file to read
        # If user specified classes, we usually read the first class's summary or the 'pedestrian' one.
        # TrackEval outputs: {class_name}_summary.txt
        primary_class = "pedestrian"
        if val_config and "CLASSES_TO_EVAL" in val_config and len(val_config["CLASSES_TO_EVAL"]) > 0:
             # Just read the first class to get *some* numbers into the logs.
             # Ideally, we should average them or log all, but for now we pick the first.
             primary_class = val_config["CLASSES_TO_EVAL"][0]
        
        eval_metrics_path = os.path.join(outputs_dir, "tracker", f"{primary_class}_summary.txt")
        
        # Fallback check: if primary class summary doesn't exist, try pedestrian
        if not os.path.exists(eval_metrics_path):
             alt_path = os.path.join(outputs_dir, "tracker", "pedestrian_summary.txt")
             if os.path.exists(alt_path):
                 eval_metrics_path = alt_path

        if os.path.exists(eval_metrics_path):
            eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metrics_path)
            metrics["HOTA"].update(eval_metrics_dict.get("HOTA", 0.0))
            metrics["DetA"].update(eval_metrics_dict.get("DetA", 0.0))
            metrics["AssA"].update(eval_metrics_dict.get("AssA", 0.0))
            metrics["DetPr"].update(eval_metrics_dict.get("DetPr", 0.0))
            metrics["DetRe"].update(eval_metrics_dict.get("DetRe", 0.0))
            metrics["AssPr"].update(eval_metrics_dict.get("AssPr", 0.0))
            metrics["AssRe"].update(eval_metrics_dict.get("AssRe", 0.0))
            metrics["MOTA"].update(eval_metrics_dict.get("MOTA", 0.0))
            metrics["IDF1"].update(eval_metrics_dict.get("IDF1", 0.0))
            logger.success(
                log=f"Get evaluation metrics from {eval_metrics_path}.",
                only_main=True,
            )
        else:
            logger.warning(f"Could not find metrics summary file at {eval_metrics_path}. HOTA will be 0.", only_main=True)

        return metrics

@torch.no_grad()
def get_results_of_one_sequence(
        logger: Logger,
        runtime_tracker: RuntimeTracker,
        sequence_loader: DataLoader,
        memory=None # Add memory argument
):
    tracker_results = []
    assert len(sequence_loader) > 10, "The sequence loader is too short."
    for t, (image, image_path) in enumerate(sequence_loader):
        if t == 10:
            begin_time = time.time()
        image.tensors = image.tensors.cuda()
        image.mask = image.mask.cuda()
        # image = nested_tensor_from_tensor_list(tensor_list=[image[0]])
        runtime_tracker.update(image=image)
        _results = runtime_tracker.get_track_results()
        
        # --- Memory Update Logic ---
        # Only proceed if we have a memory module AND embeddings are present
        if memory is not None and "embeddings" in _results:
            raw_ids = _results["id"].tolist()
            if len(raw_ids) > 0:
                # 1. Update Memory with current frame info
                # memory.update returns the mapping: {raw_id -> consistent_global_id}
                id_map = memory.update(t, raw_ids, _results["embeddings"])
                
                # 2. Remap IDs in the results
                # If valid map exists use it, otherwise keep raw ID
                new_ids = [id_map.get(rid, rid) for rid in raw_ids]
                
                # 3. Overwrite ID tensor
                _results["id"] = torch.tensor(new_ids, dtype=torch.int64, device=_results["id"].device)
        # ---------------------------

        tracker_results.append(_results)
    fps = (len(sequence_loader) - 10) / (time.time() - begin_time)
    return tracker_results, fps


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:  # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:  # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    submit_and_evaluate(config=cfg)
