import cv2
import numpy as np
import os
from tqdm import tqdm
from os.path import join as ospj
from os.path import dirname as ospd
from evaluation import compute_bboxes_from_scoremaps
from evaluation import BoxEvaluator
from evaluation import MaskEvaluator
from evaluation import configure_metadata
from utils import t2n
from evaluation import calculate_multiple_iou
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
# _RESIZE_LENGTH = 224


def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= np.percentile(cam, 65)
    cam /= cam.max()
    cam = np.where(cam < 0, 0, cam)
    cam = np.where(cam > 1, 1, cam)

    return cam

class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, cam_curve_interval=.001, cam_path=None):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.cam_path = cam_path

        self.metadata = configure_metadata(metadata_root)
        self.cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))

        self.evaluator = {"OpenImages": MaskEvaluator,
                          "CUB": BoxEvaluator,
                          "CAR": BoxEvaluator,
                          "DOG": BoxEvaluator,
                          "AIR": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=self.metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=self.cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)
        self.total_iou_list = []

    def compute_and_evaluate_cams(self):
        # print("Computing and evaluating cams.")
        for images, targets, image_ids in tqdm(self.loader, desc="Evaluation", mininterval=0.1):
            image_size = images.shape[2:]
            images = images.cuda()
            cams = t2n(self.model(images, targets, return_cam=True))

            for cam, image_id in zip(cams, image_ids):
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                if self.split in ('test'):
                    cam_path = ospj(self.cam_path, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                else:
                    cam_path = ospj(self.cam_path, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                self.evaluator.accumulate(cam_normalized, image_id)
                self.total_iou_list.append(self.evaluator.image_iou_list)

        return self.evaluator.compute(), np.array(self.total_iou_list)