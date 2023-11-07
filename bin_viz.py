from __future__ import absolute_import, division, print_function
import open3d as o3d
import numpy as np
import os
import sys
import glob
import argparse
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import shutil
import torch
import sys
from tqdm import tqdm
from torchvision import transforms, datasets
import networks
from nuscenes.nuscenes import NuScenes
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

def convert_pcd_bin_to_pcd(bin_file_path, pcd_file_path):
    # Step 1: Load the .pcd.bin file
    # import pdb; pdb.set_trace()
    pcd_bin_data = np.fromfile(bin_file_path, dtype=np.float32)

    # Step 2: Reshape the data into a 2D array of XYZ coordinates (3 columns)
    points = pcd_bin_data.reshape(-1, 5)[:, :3]

    # Step 3: Create an Open3D point cloud object from the XYZ coordinates
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Step 4: Save the point cloud to .pcd file in ASCII format
    o3d.io.write_point_cloud(pcd_file_path, pcd)

    print("Conversion successful.")

def visualize_pcd(pcd_file_path):
    # Step 1: Load the .pcd file
    pcd = o3d.io.read_point_cloud(pcd_file_path)

    if not pcd.has_points():
        print("The point cloud is empty.")
        return

    # Step 2: Visualize the point cloud
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3.0
    num_points = len(pcd.points)
    black_color = [0, 0, 0]  # RGB values for black (all 0)
    pcd.colors = o3d.utility.Vector3dVector([black_color] * num_points)
    o3d.visualization.draw_geometries([pcd])





nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes', verbose=True)
    
my_scene = nusc.scene[0]
first_sample_token = my_scene['first_sample_token']
my_sample = nusc.get('sample', first_sample_token)
sample_data_list = []
npy_list = []
input_pcd = nusc.get('sample_data', my_sample['data']['LIDAR_TOP'])
input_pcd_bin_file = os.path.join("../nuscenes/v1.0-trainval", input_pcd['filename'])
# Example usage:
# input_pcd_bin_file = "../nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"
# input_pcd_bin_file ="../nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-09-18-14-18-33-0400__LIDAR_TOP__1537294727198449.pcd.bin"


output_pcd_file = "./output.pcd"
convert_pcd_bin_to_pcd(input_pcd_bin_file, output_pcd_file)

input_pcd_file = "./output.pcd"
visualize_pcd(input_pcd_file)