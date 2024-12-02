import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import numpy as np
from PIL import Image
import torch
import detector

# MegaPose
from megapose.inference.pose_estimator import PoseEstimator, ObservationTensor
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData, transform_to_list
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.utils.tensor_collection import PandasTensorCollection
from megapose.lib3d.transform import Transform
from megapose.inference.utils import load_pose_models
from megapose.inference.icp_refiner import ICPRefiner
from megapose.utils.logging import get_logger, set_logging_level

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/system/miniconda3/envs/megapose/lib/

logger = get_logger(__name__)

_CLASSES =("valve","valve2")

LOCAL_DIR =Path(".")

NAMED_MODELS = {
    "megapose-1.0-RGB-multi-hypothesis": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": False,
        "inference_parameters": {
            "n_refiner_iterations": 7,
            "n_pose_hypotheses": 5,
        },
    },
    "megapose-1.0-RGB-multi-hypothesis-icp": {
        "coarse_run_id": "coarse-rgb-906902141",
        "refiner_run_id": "refiner-rgb-653307694",
        "requires_depth": True,
        "depth_refiner": "ICP",
        "inference_parameters": {
            "n_refiner_iterations": 5,
            "n_pose_hypotheses": 5,
            "run_depth_refiner": True,
        },
    },
}


class MegaposeEstimater:

    def __init__(self,data_dir: Path, use_depth:bool =False):
        print(os.cpu_count()) 

        if use_depth:
            model_name="megapose-1.0-RGB-multi-hypothesis-icp"
        else:
            model_name="megapose-1.0-RGB-multi-hypothesis"
        self.model_name = model_name
        
        logger.info(f"Loading model {model_name}.")
        n_workers = 6
        bsz_objects = 2
        bsz_images = 256
        SO3_grid_size = 72
        self.detector = detector.yolox("yolox_s.onnx")
        self.data_dir = data_dir
        self.mesh_data = self.make_object_dataset(self.data_dir)
        self.camera_data = self.load_camera_data(self.data_dir)
        self.pose_estimator = self.load_named_model(model_name, self.mesh_data,n_workers, bsz_objects, bsz_images, SO3_grid_size).cuda()
        logger.info(f"Loaded Model.")
        
    def load_named_model(
        self,
        model_name: str,
        object_dataset: RigidObjectDataset,
        n_workers = 6,
        bsz_objects: int = 8,
        bsz_images: int = 128,
        SO3_grid_size: int = 72,
    ) -> PoseEstimator:

        model = NAMED_MODELS[model_name]
        
        renderer_kwargs = {
            "preload_cache": False,
            "split_objects": False,
            "n_workers": n_workers,
        }

        coarse_model, refiner_model, mesh_db = load_pose_models(
            coarse_run_id=model["coarse_run_id"],
            refiner_run_id=model["refiner_run_id"],
            object_dataset=self.mesh_data,
            force_panda3d_renderer=True,
            renderer_kwargs=renderer_kwargs,
            models_root=LOCAL_DIR / "megapose-models",
        )

        depth_refiner = None
        if model.get("depth_refiner", None) == "ICP":
            depth_refiner = ICPRefiner(
                mesh_db,
                refiner_model.renderer,
            )

        pose_estimator = PoseEstimator(
            refiner_model=refiner_model,
            coarse_model=coarse_model,
            detector_model=None,
            depth_refiner=depth_refiner,
            bsz_objects=bsz_objects,
            bsz_images=bsz_images,
            SO3_grid_size=SO3_grid_size,
        )
        return pose_estimator

    def make_object_dataset(self, data_dir: Path) -> RigidObjectDataset:
        rigid_objects = []
        mesh_units = "mm"
        print(data_dir / "meshes")
        object_dirs = (data_dir / "meshes").iterdir()

        for object_dir in object_dirs:
            label = object_dir.name
            mesh_path = None
            for fn in object_dir.glob("*"):
                if fn.suffix in {".obj", ".ply"}:
                    assert not mesh_path, f"there multiple meshes in the {label} directory"
                    mesh_path = fn
            assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
            rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
            # TODO: fix mesh units
        
        rigid_object_dataset = RigidObjectDataset(rigid_objects)
        print(len(rigid_object_dataset.objects))
        return rigid_object_dataset

    def load_camera_data(self, data_dir: Path,load_depth: bool = False,) -> CameraData:
        camera_data = CameraData.from_json((data_dir / "camera_data.json").read_text())
        return camera_data

    def load_observation(self, rgb, depth = None) -> ObservationTensor:
        print(depth)
        assert rgb.shape[:2] == self.camera_data.resolution
        if depth is not None:
            assert depth.shape[:2] == self.camera_data.resolution
        observation = ObservationTensor.from_numpy(rgb, depth, self.camera_data.K)

        return observation

    def make_detections_from_yolo(self, boxes, cls_inds) -> DetectionsType:
        infos = pd.DataFrame(
            dict(
                label=[_CLASSES[i] for i in cls_inds],
                batch_im_id=0,
                instance_id=np.arange(len(cls_inds)),
            )
        )
        bboxes = torch.as_tensor(
            np.stack([bbox+10 for bbox in boxes]),
        )
        return PandasTensorCollection(infos=infos, bboxes=bboxes)
        
    def detection(self,img):
        final_boxes, final_scores, final_cls_inds = self.detector.detection(img)
        return final_boxes, final_scores, final_cls_inds
        
    def run_inference(self, img, depth=None):
        model_info = NAMED_MODELS[self.model_name]
        
        observation = self.load_observation(img, depth)
        observation = observation.cuda()
        print('observation loaded')
        final_boxes, final_scores, final_cls_inds = self.detection(img)
        if final_boxes is None:
            print("Object not found.")
        
        final_cls_inds = np.array(final_cls_inds, dtype="uint8")
        detections = self.make_detections_from_yolo(boxes = final_boxes, cls_inds=final_cls_inds)
        detections = detections.cuda()
        
        logger.info(f"Running inference.")

        output, extra = self.pose_estimator.run_inference_pipeline(
            observation, detections=detections, **model_info["inference_parameters"])
        
        labels = output.infos["label"]
        poses = output.poses.cpu().numpy()

        object_data = [
            ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)]
        outputs = []
        boxes = []

        for x ,box in zip(object_data,final_boxes):
            print(transform_to_list(x.TWO))
            outputs.append(transform_to_list(x.TWO))
            boxes.append(box)

        logger.info(f"Inference done.")

        print(f"Inference times; {extra['timing_str']}")
        print(output)
        print(f"Coarse model: {extra['coarse']['data']['timing_str']}")

        #print(f"Coarse model: {extra['coarse']['data']['debug']['renders'].shape}")
        #print(f"Coarse model: {extra['coarse']['data']['debug']['renders'][0,0,:3].shape}")
        #output renders
        #for i in range(576):
            #render_crop = (extra['coarse']['data']['debug']['renders'][0,i,:3].permute(1, 2, 0) * 255).cpu().numpy().astype(np.uint8)
            #pil_image = Image.fromarray(render_crop)
            #pil_image.save(f"./renders/render_crop_{i}.png")
        
        return outputs ,boxes

if __name__ == "__main__":    

    img = np.array(Image.open(LOCAL_DIR / "data/test.jpg"), dtype=np.uint8)
    
    set_logging_level("info")
    #print(torch.__config__.parallel_info()) 

    megapose = MegaposeEstimater(LOCAL_DIR / "data")

    out_puts = megapose.run_inference(img)

    try:
    	megapose.run_inference(observation, detections)
    except:
        print("error")
    megapose = None
