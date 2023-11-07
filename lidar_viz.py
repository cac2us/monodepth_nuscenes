from __future__ import absolute_import, division, print_function
import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import torch
import sys
from tqdm import tqdm
from torchvision import transforms, datasets
import open3d as o3d
import networks
from nuscenes.nuscenes import NuScenes
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes', verbose=True)
    
my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
sample_data_list = []
npy_list = []
lidar_data = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
