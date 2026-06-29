matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pycocotools import mask as maskUtils
import os
import argparse
from tqdm import tqdm
from mmcv_box.visualization.image import imshow_det_bboxes
import multiprocessing
import os.path as osp
import sys
import pandas as pd
import gc
import cv2
from chainercv.evaluations import eval_instance_segmentation_voc


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = osp.abspath(osp.dirname(osp.dirname(__file__)))
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)
from datasets.json_inference import coco_inst_seg_eval

result_file = "results_coco/BAS_COCO.json"
label_file = "./datasets/CUB_200_2011/CUB_as_COCO/annotations/instances_val2017.json"
root = "./datasets/CUB_200_2011/CUB_200_2011/images_combined"

mAP, cls_ap, cls_names = coco_inst_seg_eval(label_file, result_file)