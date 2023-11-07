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
from torchvision import transforms, datasets
import open3d as o3d
import networks
from nuscenes.nuscenes import NuScenes
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist


npy input
np.concatenate
depth_map = disp_resized_np

            # Intrinsic Matrix와 Extrinsic Matrix (예시로 임의의 값으로 설정)
intrinsic_matrix = cam_int

# Pseudo point cloud를 생성합니다.
p_c = pseudo_point_cloud(depth_map, intrinsic_matrix, extrinsic_matrix)
# import pdb; pdb.set_trace()
# 결과 출력
print(p_c)
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(p_c)
# Visualize the PointCloud
o3d.visualization.draw_geometries([point_cloud])