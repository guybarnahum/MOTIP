# Copyright (c) Ruopeng Gao. All Rights Reserved.

import os
import torch
from collections import defaultdict
from configparser import ConfigParser

from .one_dataset import OneDataset
from .util import is_legal, append_annotation


class DanceTrack(OneDataset):
    def __init__(
            self,
            data_root: str = "./datasets/",
            sub_dir: str = "DanceTrack",
            split: str = "train",
            load_annotation: bool = True,
    ):
        super(DanceTrack, self).__init__(
            data_root=data_root,
            sub_dir=sub_dir,
            split=split,
            load_annotation=load_annotation,
        )

        # Prepare the data:
        self.sequence_infos = self._get_sequence_infos()
        self.image_paths = self._get_image_paths()
        if self.load_annotation:
            self.annotations = self._get_annotations()
        return

    def _get_sequence_names(self):
        return os.listdir(os.path.join(self.data_dir, self.split))

    def _get_sequence_infos(self):
        sequence_names = self._get_sequence_names()
        sequence_infos = dict()
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            ini = ConfigParser()
            ini.read(os.path.join(sequence_dir, "seqinfo.ini"))
            sequence_infos[sequence_name] = {
                "width": int(ini["Sequence"]["imWidth"]),
                "height": int(ini["Sequence"]["imHeight"]),
                "length": int(ini["Sequence"]["seqLength"]),
                "is_static": False,
            }
        return sequence_infos

    def _get_image_paths(self):
        sequence_names = self._get_sequence_names()
        image_paths = defaultdict(list)
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            for i in range(self.sequence_infos[sequence_name]["length"]):
                image_paths[sequence_name].append(self._get_image_path(sequence_dir, i))
        return image_paths

    @staticmethod
    def _get_sequence_dir(data_dir, split, sequence_name):
        return str(os.path.join(data_dir, split, sequence_name))

    @staticmethod
    def _get_image_path(sequence_dir, frame_idx):
        return str(os.path.join(sequence_dir, "img1", f"{frame_idx+1:08d}.jpg"))    # the image name is 1-indexed

    def _get_annotations(self):
        sequence_names = self._get_sequence_names()
        # Init the annotations:
        annotations = self._init_annotations(sequence_names)
        # Load the annotations:
        for sequence_name in sequence_names:
            sequence_dir = self._get_sequence_dir(self.data_dir, self.split, sequence_name)
            gt_file_path = os.path.join(sequence_dir, "gt", "gt.txt")
            
            # --- MULTI-CLASS ID RE-INDEXING ---
            # Map raw IDs to vocabulary ranges: Person [0, 499], Vehicle [500, 999]
            id_map = {} 
            next_person_idx = 0
            next_vehicle_idx = 500
            # ----------------------------------

            with open(gt_file_path, "r") as gt_file:
                for line in gt_file:
                    line = line.strip().split(",")
                    # frame, id, x, y, w, h, conf, class, visibility
                    frame_id, raw_obj_id, x, y, w, h, _, class_id, _ = line
                    
                    frame_id, raw_obj_id = map(int, [frame_id, raw_obj_id])
                    x, y, w, h = map(float, [x, y, w, h])
                    
                    category = int(class_id) 

                    # --- MAP RAW ID TO VOCABULARY RANGE ---
                    if (raw_obj_id, category) not in id_map:
                        if category == 1:  # Person
                            id_map[(raw_obj_id, category)] = next_person_idx
                            next_person_idx = (next_person_idx + 1) % 500
                        else:  # Vehicle
                            id_map[(raw_obj_id, category)] = next_vehicle_idx
                            # Ensure vehicle index stays within 500-999
                            next_vehicle_idx = 500 + ((next_vehicle_idx - 500 + 1) % 500)
                    
                    obj_id = id_map[(raw_obj_id, category)]
                    # ---------------------------------------
                    
                    bbox = [x, y, w, h]
                    visibility = 1.0
                    ann_index = frame_id - 1
                    
                    annotations[sequence_name][ann_index] = append_annotation(
                        annotation=annotations[sequence_name][ann_index],
                        obj_id=obj_id,
                        category=category,
                        bbox=bbox,
                        visibility=visibility,
                    )
        
        # Determine whether each annotation is legal:
        for sequence_name in sequence_names:
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name][i]["is_legal"] = is_legal(annotations[sequence_name][i])
        return annotations 
    
    def _init_annotations(self, sequence_names):
        annotations = dict()
        for sequence_name in sequence_names:
            annotations[sequence_name] = []
            for i in range(self.sequence_infos[sequence_name]["length"]):
                annotations[sequence_name].append({
                    "id": torch.zeros((0, ), dtype=torch.int64),
                    "category": torch.zeros((0, ), dtype=torch.int64),
                    "bbox": torch.zeros((0, 4), dtype=torch.float32),
                    "visibility": torch.zeros((0, ), dtype=torch.float32),
                    "trajectory_class_labels": torch.zeros((0, 1, 1), dtype=torch.int64), 
                    "trajectory_is_legal": torch.zeros((0, ), dtype=torch.bool),
                })
        return annotations
