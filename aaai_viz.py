import open3d as o3d
import numpy as np

import struct
import sys
from struct import unpack


file_path = "../nuscenes/v1.0-trainval/samples/LIDAR_TOP/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"

# with open(file_path, "rb") as f:
#     number = f.read(4)
#     while number != b"":
#         print(np.frombuffer(number, dtype=np.float32))
#         number = f.read(4)

# Step 2: Load the .pcd.bin file
# file_path = "../nuscenes/v1.0-trainval/v1.0-trainval/samples/LIDAR/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"
point_cloud = o3d.io.read_point_cloud(file_path, format='xyzn')
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_size = 3.0
vis.get_render_option().background_color = np.zeros(3)
# pcd = o3d.io.read_point_cloud(pcd_path)
num_points = len(point_cloud.points)
black_color = [0, 0, 0]  # RGB values for black (all 0)
point_cloud.colors = o3d.utility.Vector3dVector([black_color] * num_points)
point_cloud.points = o3d.utility.Vector3dVector(o3d.io.read_point_cloud(file_path).points)

# Step 4: Visualize the point cloud
o3d.visualization.draw_geometries([point_cloud])


# For XYZ coordinates + intens


# Step 2: Load the .pcd.bin file
# file_path = "../nuscenes/v1.0-trainval/v1.0-trainval/samples/LIDAR/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"

# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(o3d.io.read_point_cloud(file_path).points)

# # Step 4: Visualize the point cloud
# o3d.visualization.draw_geometries([point_cloud])
