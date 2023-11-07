import open3d as o3d
import numpy as np
import json
import math
import copy

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
    axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set, box3d

def generate_plane(vis):
    xy = np.linspace(-100,100,num=1000)
    aa,bb = np.meshgrid(xy,xy)
    aa = np.array(aa)
    bb = np.array(bb)
    xy = np.stack([aa,bb],axis=-1).reshape(-1,2)
    z = np.zeros([xy.shape[0],1])
    xyz = np.concatenate([xy,z],axis=-1)
    pcdv2 = o3d.geometry.PointCloud()
    pcdv2.points = o3d.utility.Vector3dVector(xyz)
    pcdv2.paint_uniform_color((0,0,1))
    vis.add_geometry(pcdv2)

def rotate_points(x,y,z):
    theta = np.pi / 13.5
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[:3, :3] = np.array([[1,0,0],[0,c,-s],[0,s,c]])
    T[0,3] = 0
    T[1,3] = 0
    T[2,3] = 5.7
    points = np.array([x,y,z,1])
    points = points @ T.T
    return points[:-1]

def show_open3d_pcd(pcd_path, anno_path, show_origin=True, origin_size=3, show_grid=True):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.zeros(3)
    pcd = o3d.io.read_point_cloud(pcd_path)
    T = np.eye(4)
    T[:3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi / 13.5, 0, 0))
    T[0, 3] = 0
    T[1, 3] = 0
    T[2, 3] = 5.7
    # pcd = copy.deepcopy(pcd).transform(T)
    pcd.paint_uniform_color((1, 1, 1))
    vis.add_geometry(pcd)
    generate_plane(vis)
    # with open(anno_path, "r") as f:
    #     anno = np.load(f)
    # num_obj = anno['annotation_metadata']['object_num']
    anno = np.load(anno_path)
    if (nsweeps == 4) & (test_env ==  "voxel"):
        num_obj = 500
    if (nsweeps == 4) & (test_env ==  "loading"):
        num_obj = anno.shape[0]
    if (nsweeps == 1):
        num_obj = anno.shape[0]
    # import pdb; pdb.set_trace()

    ### box test
    if nsweeps == 1:
        for i in range(num_obj):
            # import pdb; pdb.set_trace()
            x,y,z = anno[i][:3]
            l,h,w = anno[i][3:6]     
            # h,w,l = anno[i][3:6]     
            
            orientation = -anno[i][6]-np.pi/2
            gt_box = [x,y,z]+[h,w,l]+[orientation]
            line_set, _ = translate_boxes_to_open3d_instance(gt_box)
            line_set.paint_uniform_color((0,1,0))
            vis.add_geometry(line_set)

    ### 4sweeps in loading
    if (nsweeps == 4) & (test_env == "loading"):
        for i in range(num_obj):
            # import pdb; pdb.set_trace()
            x,y,z = anno[i][:3]
            # l,h,w = anno[0][i][3:6]     
            h,w,l = anno[i][3:6]     
            
            orientation = -anno[i][6]-np.pi/2
            gt_box = [x,y,z]+[h,w,l]+[orientation]
            line_set, _ = translate_boxes_to_open3d_instance(gt_box)
            line_set.paint_uniform_color((0,1,0))
            vis.add_geometry(line_set)

    ### 4sweeps in voxelnet
    if (nsweeps == 4) & (test_env ==  "voxel"):
        for i in range(num_obj):
            # import pdb; pdb.set_trace()
            x,y,z = anno[0][i][:3]
            l,h,w = anno[0][i][3:6]     
            # h,w,l = anno[0][i][3:6]     
            
            orientation = -anno[0][i][6]-np.pi/2
            gt_box = [x,y,z]+[h,w,l]+[orientation]
            line_set, _ = translate_boxes_to_open3d_instance(gt_box)
            line_set.paint_uniform_color((0,1,0))
            vis.add_geometry(line_set)


    vis.run()
    vis.destroy_window()


nsweeps = 1
test_env = "loading"

### 4sweep
if nsweeps == 4:
    pcd_path = "/home/CenterPoint/points_11.pcd"
    anno_path = "/home/CenterPoint/box_11.npy"

### 1sweep
if nsweeps == 1:
    pcd_path = "/home/CenterPoint/points_4.pcd"
    anno_path = "/home/CenterPoint/box_4.npy"
# pcd_path = "/home/CenterPoint/points.pcd"
# anno_path = "/home/CenterPoint/box.npy"
# anno_path = "/home/CenterPoint/box_org.npy"

## sweeps test
# pcd_path = "/home/CenterPoint/data/stitch/sweeps/LIDAR_TOP/000090_1637629085.005454.pcd"

## sweeps rotate test
# pcd_path = "/home/CenterPoint/data/stitch/sweeps/LIDAR_TOP/000539_1637802058.931937.pcd"

# anno_path = "/home/CenterPoint/box.npy"


# # Read PCD files
# pcd_1 = o3d.io.read_point_cloud("/home/stitch/stitch_transformed/annotated_data/scene_001/lidar/008630_1637629936.465564.pcd")
# pcd_2 = o3d.io.read_point_cloud("/home/stitch/stitch_transformed/annotated_data/scene_001/lidar/008631_1637629936.565542.pcd")
# pcd_3 = o3d.io.read_point_cloud("/home/stitch/stitch_transformed/annotated_data/scene_001/lidar/008632_1637629936.665545.pcd")
# pcd_4 = o3d.io.read_point_cloud("/home/stitch/stitch_transformed/annotated_data/scene_001/lidar/008633_1637629936.765551.pcd")

# # Concatenate point clouds
# merged_pcd = o3d.geometry.PointCloud()
# merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_1.points), np.asarray(pcd_2.points), np.asarray(pcd_3.points), np.asarray(pcd_4.points)]))

# # Visualize merged point cloud
# o3d.visualization.draw_geometries([merged_pcd])

# import numpy as np
# import open3d as o3d

mode = 'keyframe'
    # Define colors for each point cloud
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1,1,1]]
# 005329_1637629608.911972
# 005328_1637629608.811967
if mode == 'keyframe':#005327_1637629608.711982
    # pcd_1 = o3d.io.read_point_cloud("../CenterPoint/data/stitch/sweeps/LIDAR_TOP/005327_1637629608.711982.pcd")
    # pcd_2 = o3d.io.read_point_cloud("../CenterPoint/data/stitch/sweeps/LIDAR_TOP/005328_1637629608.811967.pcd")
    # pcd_3 = o3d.io.read_point_cloud("../CenterPoint/data/stitch/sweeps/LIDAR_TOP/005329_1637629608.911972.pcd")
    pcd_4 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/keyframe_lidar/2_tues/005330_1637629609.011980.pcd")
    pcd_1 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/keyframe_lidar/2_tues/005330_1637629609.011980.pcd")
    pcd_2 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/keyframe_lidar/2_tues/005330_1637629609.011980.pcd")
    pcd_3 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/keyframe_lidar/2_tues/005330_1637629609.011980.pcd")

    # pcd_1 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/keyframe_lidar/1_mon/000870_1637568835.592679.pcd")
    # pcd_2 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/keyframe_lidar/1_mon/000870_1637568835.592679.pcd")
    # pcd_3 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/keyframe_lidar/1_mon/000870_1637568835.592679.pcd")
    # pcd_1 = o3d.io.read_point_cloud("../CenterPoint/example_4sweeps/4sweep_000740_copy.pcd")
    # pcd_2 = o3d.io.read_point_cloud("../CenterPoint/example_4sweeps/4sweep_000740_copy.pcd")
    # pcd_3 = o3d.io.read_point_cloud("../CenterPoint/example_4sweeps/4sweep_000740_copy.pcd")
    # pcd_4 = o3d.io.read_point_cloud("../CenterPoint/example_4sweeps/4sweep_000740_copy.pcd")
    
    # pcd_1 = o3d.io.read_point_cloud("../CenterPoint/data/stitch/sweeps/LIDAR_TOP/083150_1637637388.510252.pcd")
    # pcd_2 = o3d.io.read_point_cloud("../CenterPoint/data/stitch/sweeps/LIDAR_TOP/083151_1637637388.610252.pcd")
    # pcd_3 = o3d.io.read_point_cloud("../CenterPoint/data/stitch/sweeps/LIDAR_TOP/083152_1637637388.710230.pcd")
    # pcd_4 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/scene_003/lidar/083153_1637637388.810234.pcd")
if mode == 'scene':
    # Read PCD files
    pcd_1 = o3d.io.read_point_cloud("/home/stitch/stitch_transformed/annotated_data/scene_001/lidar/008630_1637629936.465564.pcd")
    pcd_2 = o3d.io.read_point_cloud("/home/stitch/stitch_transformed/annotated_data/scene_001/lidar/008631_1637629936.565542.pcd")
    pcd_3 = o3d.io.read_point_cloud("/home/stitch/stitch_transformed/annotated_data/scene_001/lidar/008632_1637629936.665545.pcd")
    pcd_4 = o3d.io.read_point_cloud("/home/stitch/stitch_transformed/annotated_data/scene_001/lidar/008633_1637629936.765551.pcd")
# /home/stitch/ark_scene_new/scene/scene_041/320669_1637697561.083268.pcd
# 000869_1637568835.492674.pcd
# 000868_1637568835.392636.pcd
# 000867_1637568835.292607.pcd
# Assign color to each point cloud
# pcd_1.paint_uniform_color(colors[0])
# pcd_2.paint_uniform_color(colors[1])
# pcd_3.paint_uniform_color(colors[2])
# pcd_4.paint_uniform_color(colors[3])
pcd_1.paint_uniform_color(colors[2])
pcd_2.paint_uniform_color(colors[2])
pcd_3.paint_uniform_color(colors[2])
pcd_4.paint_uniform_color(colors[2])
# Concatenate point clouds
merged_pcd = o3d.geometry.PointCloud()

## 4sweep test용
merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_1.points), np.asarray(pcd_2.points), np.asarray(pcd_3.points), np.asarray(pcd_4.points)]))
merged_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_1.colors), np.asarray(pcd_2.colors), np.asarray(pcd_3.colors), np.asarray(pcd_4.colors)]))


## 1sweep test용
# merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_1.points)]))
# merged_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_1.colors)]))

# if mode == 'keyframe':
#     o3d.io.write_point_cloud("keyframe_sweeps.pcd", merged_pcd)
# if mode == 'scene':
#     o3d.io.write_point_cloud("scene_sweeps.pcd", merged_pcd)

# Visualize merged point cloud
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.get_render_option().point_size = 3.0
vis.get_render_option().background_color = np.zeros(3)
merged_pcd.paint_uniform_color((1, 1, 1))
vis.add_geometry(merged_pcd)
generate_plane(vis)
vis.run()
vis.destroy_window()


# Add the point cloud to the visualization window
# vis.add_geometry(merged_pcd)
# o3d.visualization.draw_geometries([merged_pcd])

