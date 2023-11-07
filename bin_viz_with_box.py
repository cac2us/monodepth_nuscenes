import open3d as o3d
import numpy as np
# from __future__ import absolute_import, division, print_function
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
import cv2
from torchvision import transforms, datasets
import open3d as o3d
import networks
from nuscenes.nuscenes import NuScenes
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist

# Replace 'v1.0' with the appropriate version if needed
nusc = NuScenes(version='v1.0-trainval', dataroot='../nuscenes/v1.0-trainval', verbose=True)

## 8
# sample_data_token = 'samples/CAM_FRONT/n008-2018-09-18-14-54-39-0400__CAM_FRONT__1537297346862404.jpg'
# sample_data_token = 'samples/CAM_FRONT_LEFT/n008-2018-09-18-14-54-39-0400__CAM_FRONT_LEFT__1537297346862404.jpg'
# sample_data_token = 'samples/CAM_FRONT_RIGHT/n008-2018-09-18-14-54-39-0400__CAM_FRONT_RIGHT__1537297346862404.jpg'
# sample_data_token = 'samples/CAM_BACK/n008-2018-09-18-14-54-39-0400__CAM_BACK__1537297346862404.jpg'
# sample_data_token = 'samples/CAM_BACK_LEFT/n008-2018-09-18-14-54-39-0400__CAM_BACK_LEFT__1537297346862404.jpg'
# sample_data_token = 'samples/CAM_BACK_RIGHT/n008-2018-09-18-14-54-39-0400__CAM_BACK_RIGHT__1537297346862404.jpg'

## 15
# sample_data_token = 'samples/CAM_FRONT/n015-2018-11-21-19-58-31+0800__CAM_FRONT__1542801733412460.jpg'

def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    lwh = gt_boxes[3:6]
    axis_angles = np.array(gt_boxes[6])
    ego_angles = np.array(gt_boxes[7])
    # w = axis_angles[0]
    # x = axis_angles[1]
    # y = axis_angles[2]
    # z = axis_angles[3]
    # axis_angles[0] =axis_angles[1]
    # axis_angles[1] =axis_angles[2]
    # axis_angles[2] =axis_angles[3]
    # axis_angles[3] =axis_angles[0]
    # rot = o3d.geometry.get_rotation_matrix_from_quaternion(axis_angles)
    # ego_rott = o3d.geometry.get_rotation_matrix_from_quaternion(ego_angles)
    # rot = rot * ego_rott
    # theta = np.pi/2
    # c, s = np.cos(theta), np.sin(theta)
    # T = np.eye(3)
    # T[:3, :3] = np.array([[c,-s,0],[s,c,0],[0,0,1]])
    # # T[0,3] = 0
    # # T[1,3] = 0
    # # T[2,3] = -3
    # rot = rot @ T
    # axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set, box3d

def convert_pcd_bin_to_pcd(bin_file_path, pcd_file_path):
    # Step 1: Load the .pcd.bin file
    pcd_bin_data = np.fromfile(bin_file_path, dtype=np.float32)

    # Step 2: Reshape the data into a 2D array of XYZ coordinates (3 columns)
    points = pcd_bin_data.reshape(-1, 5)[:, :3]

    # Step 3: Create an Open3D point cloud object from the XYZ coordinates
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Step 4: Save the point cloud to .pcd file in ASCII format
    o3d.io.write_point_cloud(pcd_file_path, pcd)

    print("Conversion successful.")

def quaternion_to_rotation_matrix(quaternion):
    x, y, z, w = quaternion
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])
    return rotation_matrix

def visualize_pcd(pcd_file_path, anno_book):
    # Step 1: Load the .pcd file
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3.0
    black_color = [0, 0, 0]  # RGB values for black (all 0)
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    num_points = len(pcd.points)
    import pdb; pdb.set_trace()
    pcd.colors = o3d.utility.Vector3dVector([black_color] * num_points)
    if not pcd.has_points():
        print("The point cloud is empty.")
        return
    vis.add_geometry(pcd)
    # import pdb; pdb.set_trace()
    dimensions = []
    # Step 2: Visualize the point cloud
    # for box in gt_boxes:
    #     center = box[0:3]
    #     dimensions = box[3:6]        
    #     orientation = box[6:10]
    #     rotation_matrix = quaternion_to_rotation_matrix(orientation)
        
    #     obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, dimensions)
    #     obb.color = (0, 1, 0)  # Green color for GT boxes
    #     vis.add_geometry(obb)
    for i in range(44):
        
        w,l,h = anno_book[i][0]
        x,y,z = anno_book[i][1]
        # z=z-2
        # x = -x
        orientation = anno_book[i][2]
        ego_rot = anno_book[i][3]
        # orientation = anno_book[i][2][2]
        # orientation = anno_book[i][2][2]-np.pi/2
        # import pdb; pdb.set_trace()
        # orientation[0] -= np.pi/2
        # orientation[1] -= np.pi/2
        # orientation[2] -= np.pi/2
        # rotation_matrix = quaternion_to_rotation_matrix(orientation)
        # x,y,z = rotate_points(x,y,z)
        # import pdb; pdb.set_trace()
        gt_box = [x,y,z]+[h,w,l]+[orientation]+[ego_rot]
        line_set, _ = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color((0,3,0))
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = 10
        vis.add_geometry(line_set)
    # o3d.visualization.draw_geometries([pcd])
    vis.run()
    vis.destroy_window()

# Example usage:
input_pcd_bin_file = "../nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"
# input_pcd_bin_file = "../nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915690896796.pcd.bin"

# input_pcd_bin_file ="../nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-09-18-14-18-33-0400__LIDAR_TOP__1537294727198449.pcd.bin"
# 1526915690896796
# output_pcd_file = "./output.pcd"
# convert_pcd_bin_to_pcd(input_pcd_bin_file, output_pcd_file)

input_pcd_file = "./output.pcd"
anno_book = []
gt_boxes=[]
for i in range(2631083):
    # import pdb; pdb.set_trace()
    if nusc.__dict__['sample_data'][i]['filename'] == "/".join(input_pcd_bin_file.split('/')[3:]):
        sample = nusc.get('sample', nusc.__dict__['sample_data'][i]['sample_token'])
        ego = nusc.get('ego_pose', nusc.__dict__['sample_data'][i]['ego_pose_token'])
        
        for annotation_token in sample['anns']:
            annotation = nusc.get('sample_annotation', annotation_token)
            # import pdb; pdb.set_trace()
            nusc.render_annotation(annotation_token, './')
            # cv2.imwrite("output_image.jpg", what)
            # Extract relevant information
            # annotation['size']
            # box = annotation['translation'] + annotation['size'] + annotation['rotation']
            # annotation['rotation']
            w = annotation['size'][0]
            l = annotation['size'][1]  # Height of the bounding box
            h = annotation['size'][2]  # Length of the bounding box
            trans = annotation['translation']
            # trans = (a-b for a, b in zip (annotation['translation'], ego['translation'])) # Translation (x, y, z)
            # rot = (x-y for x, y in zip (annotation['rotation'], ego['rotation']))#annotation['rotation'] #- ego['rotation'] # Rotation quaternion (x, y, z, w)
            rot = annotation['rotation'] #- ego['rotation'] # Rotation quaternion (x, y, z, w)
            ego_rot = ego['rotation']
            # rot[0] = rot[0] / ego['rotation'][0]
            # rot[1] = rot[1] / ego['rotation'][1]
            # rot[2] = rot[2] / ego['rotation'][2]
            # rot[3] = rot[3] / ego['rotation'][3]
            # rot = quaternion_to_rotation_matrix(rot)
            # import pdb; pdb.set_trace()
            # gt_boxes.append(box)
            anno_book.append([[w,h,l], trans, rot, ego_rot])
visualize_pcd(input_pcd_file, anno_book)