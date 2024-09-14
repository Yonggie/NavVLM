# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import pathlib
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


from core.abstract_perception import PerceptionModule
from core.interfaces import Observations
from perception.detection.utils import filter_depth, overlay_masks

sys.path.insert(
    0, str(Path(__file__).resolve().parent / "Detic/third_party/CenterNet2/")
)
from detectron2.config import CfgNode as CN

def add_centernet_config(cfg):
    _C = cfg

    _C.MODEL.CENTERNET = CN()
    _C.MODEL.CENTERNET.NUM_CLASSES = 80
    _C.MODEL.CENTERNET.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
    _C.MODEL.CENTERNET.FPN_STRIDES = [8, 16, 32, 64, 128]
    _C.MODEL.CENTERNET.PRIOR_PROB = 0.01
    _C.MODEL.CENTERNET.INFERENCE_TH = 0.05
    _C.MODEL.CENTERNET.CENTER_NMS = False
    _C.MODEL.CENTERNET.NMS_TH_TRAIN = 0.6
    _C.MODEL.CENTERNET.NMS_TH_TEST = 0.6
    _C.MODEL.CENTERNET.PRE_NMS_TOPK_TRAIN = 1000
    _C.MODEL.CENTERNET.POST_NMS_TOPK_TRAIN = 100
    _C.MODEL.CENTERNET.PRE_NMS_TOPK_TEST = 1000
    _C.MODEL.CENTERNET.POST_NMS_TOPK_TEST = 100
    _C.MODEL.CENTERNET.NORM = "GN"
    _C.MODEL.CENTERNET.USE_DEFORMABLE = False
    _C.MODEL.CENTERNET.NUM_CLS_CONVS = 4
    _C.MODEL.CENTERNET.NUM_BOX_CONVS = 4
    _C.MODEL.CENTERNET.NUM_SHARE_CONVS = 0
    _C.MODEL.CENTERNET.LOC_LOSS_TYPE = 'giou'
    _C.MODEL.CENTERNET.SIGMOID_CLAMP = 1e-4
    _C.MODEL.CENTERNET.HM_MIN_OVERLAP = 0.8
    _C.MODEL.CENTERNET.MIN_RADIUS = 4
    _C.MODEL.CENTERNET.SOI = [[0, 80], [64, 160], [128, 320], [256, 640], [512, 10000000]]
    _C.MODEL.CENTERNET.POS_WEIGHT = 1.
    _C.MODEL.CENTERNET.NEG_WEIGHT = 1.
    _C.MODEL.CENTERNET.REG_WEIGHT = 2.
    _C.MODEL.CENTERNET.HM_FOCAL_BETA = 4
    _C.MODEL.CENTERNET.HM_FOCAL_ALPHA = 0.25
    _C.MODEL.CENTERNET.LOSS_GAMMA = 2.0
    _C.MODEL.CENTERNET.WITH_AGN_HM = False
    _C.MODEL.CENTERNET.ONLY_PROPOSAL = False
    _C.MODEL.CENTERNET.AS_PROPOSAL = False
    _C.MODEL.CENTERNET.IGNORE_HIGH_FP = -1.
    _C.MODEL.CENTERNET.MORE_POS = False
    _C.MODEL.CENTERNET.MORE_POS_THRESH = 0.2
    _C.MODEL.CENTERNET.MORE_POS_TOPK = 9
    _C.MODEL.CENTERNET.NOT_NORM_REG = True
    _C.MODEL.CENTERNET.NOT_NMS = False
    _C.MODEL.CENTERNET.NO_REDUCE = False

    _C.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE = False
    _C.MODEL.ROI_BOX_HEAD.PRIOR_PROB = 0.01
    _C.MODEL.ROI_BOX_HEAD.USE_EQL_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = \
        'datasets/lvis/lvis_v1_train_cat_info.json'
    _C.MODEL.ROI_BOX_HEAD.EQL_FREQ_CAT = 200
    _C.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT = 50
    _C.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT = 0.5
    _C.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE = False

    _C.MODEL.BIFPN = CN()
    _C.MODEL.BIFPN.NUM_LEVELS = 5
    _C.MODEL.BIFPN.NUM_BIFPN = 6
    _C.MODEL.BIFPN.NORM = 'GN'
    _C.MODEL.BIFPN.OUT_CHANNELS = 160
    _C.MODEL.BIFPN.SEPARABLE_CONV = False

    _C.MODEL.DLA = CN()
    _C.MODEL.DLA.OUT_FEATURES = ['dla2']
    _C.MODEL.DLA.USE_DLA_UP = True
    _C.MODEL.DLA.NUM_LAYERS = 34
    _C.MODEL.DLA.MS_OUTPUT = False
    _C.MODEL.DLA.NORM = 'BN'
    _C.MODEL.DLA.DLAUP_IN_FEATURES = ['dla3', 'dla4', 'dla5']
    _C.MODEL.DLA.DLAUP_NODE = 'conv'

    _C.SOLVER.RESET_ITER = False
    _C.SOLVER.TRAIN_ITER = -1

    _C.INPUT.CUSTOM_AUG = ''
    _C.INPUT.TRAIN_SIZE = 640
    _C.INPUT.TEST_SIZE = 640
    _C.INPUT.SCALE_RANGE = (0.1, 2.)
    # 'default' for fixed short/ long edge, 'square' for max size=INPUT.SIZE
    _C.INPUT.TEST_INPUT_TYPE = 'default' 
    _C.INPUT.NOT_CLAMP_BOX = False
    
    _C.DEBUG = False
    _C.SAVE_DEBUG = False
    _C.SAVE_PTH = False
    _C.VIS_THRESH = 0.3
    _C.DEBUG_SHOW_NAME = False


from perception.detection.detic.Detic.detic.config import (  # noqa: E402
    add_detic_config,
)
from perception.detection.detic.Detic.detic.modeling.text.text_encoder import (  # noqa: E402
    build_text_encoder,
)
from perception.detection.detic.Detic.detic.modeling.utils import (  # noqa: E402
    reset_cls_test,
)

BUILDIN_CLASSIFIER = {
    "lvis": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/lvis_v1_clip_a+cname.npy",
    "objects365": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/o365_clip_a+cnamefix.npy",
    "openimages": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/oid_clip_a+cname.npy",
    "coco": Path(__file__).resolve().parent
    / "Detic/datasets/metadata/coco_clip_a+cname.npy",
}

BUILDIN_METADATA_PATH = {
    "lvis": "lvis_v1_val",
    "objects365": "objects365_v2_val",
    "openimages": "oid_val_expanded",
    "coco": "coco_2017_val",
}


def get_clip_embeddings(vocabulary, prompt="a "):
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb


class DeticPerception(PerceptionModule):
    def __init__(
        self,
        config_file=None,
        vocabulary="coco",
        custom_vocabulary="",
        checkpoint_file=None,
        sem_gpu_id=0,
        verbose: bool = False,
        threshold=0.5
    ):
        """Load trained Detic model for inference.

        Arguments:
            config_file: path to model config
            vocabulary: currently one of "coco" for indoor coco categories or "custom"
             for a custom set of categories
            custom_vocabulary: if vocabulary="custom", this should be a comma-separated
             list of classes (as a single string)
            checkpoint_file: path to model checkpoint
            sem_gpu_id: GPU ID to load the model on, -1 for CPU
            verbose: whether to print out debug information
        """
        self.verbose = verbose
        if config_file is None:
            config_file = str(
                Path(__file__).resolve().parent
                / "Detic/configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
            )
        if checkpoint_file is None:
            checkpoint_file = str(
                Path(__file__).resolve().parent
                / "Detic/models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth"
            )
        if self.verbose:
            print(
                f"Loading Detic with config={config_file} and checkpoint={checkpoint_file}"
            )

        string_args = f"""
            --config-file {config_file} --vocabulary {vocabulary}
            """

        if vocabulary == "custom":
            assert custom_vocabulary != ""
            string_args += f""" --custom_vocabulary {custom_vocabulary}"""

        string_args += f""" --opts MODEL.WEIGHTS {checkpoint_file}"""

        if sem_gpu_id == -1:
            string_args += """ MODEL.DEVICE cpu"""
        else:
            string_args += f""" MODEL.DEVICE cuda:{sem_gpu_id}"""
        
        # string_args += f""" confidence_threshold {threshold}"""

        string_args = string_args.split()
        args = get_parser().parse_args(string_args)
        args.confidence_threshold = threshold

        cfg = setup_cfg(args, verbose=verbose)

        assert vocabulary in ["coco", "custom", "lvis"]
        if args.vocabulary == "custom":
            if "__unused" in MetadataCatalog.keys():
                MetadataCatalog.remove("__unused")
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = args.custom_vocabulary.split(",")
            classifier = get_clip_embeddings(self.metadata.thing_classes)
            self.categories_mapping = {
                i: i for i in range(len(self.metadata.thing_classes))
            }
        elif args.vocabulary == "coco":
            self.metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]
            # this is only for num2num!
            self.categories_mapping = {
                56: 0,  # chair
                57: 1,  # couch
                58: 2,  # plant
                59: 3,  # bed
                61: 4,  # toilet
                62: 5,  # tv
                60: 6,  # table
                69: 7,  # oven
                71: 8,  # sink
                72: 9,  # refrigerator
                73: 10,  # book
                74: 11,  # clock
                75: 12,  # vase
                41: 13,  # cup
                39: 14,  # bottle
                0: 15, # categories that don't fall into the above ones
                16: 16, # llm
            }

            self.category_id_to_name = {
                0: 'chair',
                1: 'couch',
                2: 'plant',
                3: 'bed',
                4: 'toilet',
                5: 'tv',
                6: 'table',
                7: 'oven',
                8: 'sink',
                9: 'refrigerator',
                10: 'book',
                11: 'clock',
                12: 'vase',
                13: 'cup',
                14: 'bottle',
                15: 'llm',
                16: 'other things',
            }

        elif args.vocabulary == "lvis":
            self.metadata = MetadataCatalog.get(
                BUILDIN_METADATA_PATH[args.vocabulary])
            classifier = BUILDIN_CLASSIFIER[args.vocabulary]
            self.categories_mapping = {
                i: i for i in range(len(self.metadata.thing_classes))
            }
            # print(self.categories_mapping)

        self.num_sem_categories = len(self.categories_mapping)
        # print(self.num_sem_categories)

        num_classes = len(self.metadata.thing_classes)
        self.cpu_device = torch.device("cpu")
        self.instance_mode = ColorMode.IMAGE
        self.predictor = DefaultPredictor(cfg)

        if type(classifier) == pathlib.PosixPath:
            classifier = str(classifier)
        reset_cls_test(self.predictor.model, classifier, num_classes)
        

    def reset_vocab(self, new_vocab: List[str], vocab_type="custom"):
        """Resets the vocabulary of Detic model allowing you to change detection on
        the fly. Note that previous vocabulary is not preserved.
        Args:
            new_vocab: list of strings representing the new vocabulary
            vocab_type: one of "custom" or "coco"; only "custom" supported right now
        """
        if self.verbose:
            print(f"Resetting vocabulary to {new_vocab}")
        MetadataCatalog.remove("__unused")
        if vocab_type == "custom":
            self.metadata = MetadataCatalog.get("__unused")
            self.metadata.thing_classes = new_vocab
            classifier = get_clip_embeddings(self.metadata.thing_classes)
            self.categories_mapping = {
                i: i for i in range(len(self.metadata.thing_classes))
            }
        else:
            raise NotImplementedError(
                "Detic does not have support for resetting from custom to coco vocab"
            )
        self.num_sem_categories = len(self.categories_mapping)

        num_classes = len(self.metadata.thing_classes)
        reset_cls_test(self.predictor.model, classifier, num_classes)

    def predict(
        self,
        obs: Observations,
        depth_threshold: Optional[float] = None,
        draw_instance_predictions: bool = True,
    ) -> Observations:
        """
        Arguments:
            obs.rgb: image of shape (H, W, 3) (in RGB order - Detic expects BGR)
            obs.depth: depth frame of shape (H, W), used for depth filtering
            depth_threshold: if specified, the depth threshold per instance

        Returns:
            obs.semantic: segmentation predictions of shape (H, W) with
             indices in [0, num_sem_categories - 1]
            obs.task_observations["semantic_frame"]: segmentation visualization
             image of shape (H, W, 3)
        """
        image = cv2.cvtColor(obs.rgb, cv2.COLOR_RGB2BGR)
        depth = obs.depth
        height, width, _ = image.shape

        pred = self.predictor(image)

        if obs.task_observations is None:
            obs.task_observations = {}

        if draw_instance_predictions:
            visualizer = Visualizer(
                image[:, :, ::-1], self.metadata, instance_mode=self.instance_mode
            )
            visualization = visualizer.draw_instance_predictions(
                predictions=pred["instances"].to(self.cpu_device)
            ).get_image()
            obs.task_observations["semantic_frame"] = visualization
        else:
            obs.task_observations["semantic_frame"] = None

        # Sort instances by mask size
        masks = pred["instances"].pred_masks.cpu().numpy()
        class_idcs = pred["instances"].pred_classes.cpu().numpy()
        scores = pred["instances"].scores.cpu().numpy()

        if depth_threshold is not None and depth is not None:
            masks = np.array(
                [filter_depth(mask, depth, depth_threshold) for mask in masks]
            )

        semantic_map, instance_map = overlay_masks(masks, class_idcs, (height, width))

        obs.semantic = semantic_map.astype(int)
        obs.task_observations["instance_map"] = instance_map
        obs.task_observations["instance_classes"] = class_idcs
        obs.task_observations["instance_scores"] = scores

        return obs


def setup_cfg(args, verbose: bool = True):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = (
        args.confidence_threshold
    )
    # args.confidence_threshold = 0.9
    print("[DETIC] Confidence threshold =", args.confidence_threshold)
    if verbose:
        print("[DETIC] Confidence threshold =", args.confidence_threshold)
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
    if not args.pred_all_class:
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    # Fix cfg paths given we're not running from the Detic folder
    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(
        Path(__file__).resolve().parent / "Detic" / cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
    )
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", help="Take inputs from webcam.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU only.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--vocabulary",
        default="lvis",
        choices=["lvis", "openimages", "objects365", "coco", "custom"],
        help="",
    )
    parser.add_argument(
        "--custom_vocabulary",
        default="",
        help="",
    )
    parser.add_argument("--pred_all_class", action="store_true")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.45,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser