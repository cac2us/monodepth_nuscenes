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
    # pcd = o3d.io.read_point_cloud(pcd_path)
    T = np.eye(4)
    # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi / 13.5, 0, 0))
    # T[0, 3] = 0
    # T[1, 3] = 0
    xx = 10000000
    T[2, 3] = xx

    # adjust camera position for closer view
    cam_x = 0   # adjust this value to move camera in x-axis direction
    cam_y = 0   # adjust this value to move camera in y-axis direction
    cam_z = -xx  # adjust this value to move camera in z-axis direction
    T_cameraview = np.eye(4)
    T_cameraview[:3, 3] = [cam_x, cam_y, cam_z]
    T = np.dot(T_cameraview, T)
    # pcd = pcd.transform(T)
    # import pdb; pdb.set_trace()
    # pcd_1 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/scene_003/lidar/083150_1637637388.510252.pcd")
    # pcd_2 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/scene_003/lidar/083151_1637637388.610252.pcd")
    # pcd_3 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/scene_003/lidar/083152_1637637388.710230.pcd")
    # pcd_4 = o3d.io.read_point_cloud("./stitch_transformed/annotated_data/scene_003/lidar/083153_1637637388.810234.pcd")
    
    # pcd_1.paint_uniform_color((1, 1, 1))
    # pcd_2.paint_uniform_color((1, 1, 1))
    # pcd_3.paint_uniform_color((1, 1, 1))
    # pcd_4.paint_uniform_color((1, 1, 1))
    
    # # Concatenate point clouds
    # merged_pcd = o3d.geometry.PointCloud()
    # merged_pcd.points = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_1.points), np.asarray(pcd_2.points), np.asarray(pcd_3.points), np.asarray(pcd_4.points)]))
    # merged_pcd.colors = o3d.utility.Vector3dVector(np.concatenate([np.asarray(pcd_1.colors), np.asarray(pcd_2.colors), np.asarray(pcd_3.colors), np.asarray(pcd_4.colors)]))

    # pcd = merged_pcd
    pcd = copy.deepcopy(pcd).transform(T)
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
    # pcd_path = "/home/CenterPoint/points_11.pcd"
    # anno_path = "/home/CenterPoint/box_11.npy"
    #1sweep voxel
    # pcd_path = "/home/CenterPoint/example_4sweeps/points_1sweep.pcd"
    # anno_path = "/home/CenterPoint/example_4sweeps/box_1sweep.npy"
    #4sweep voxel
    # pcd_path = "/home/CenterPoint/example_4sweeps/points_4sweep.pcd"
    # anno_path = "/home/CenterPoint/example_4sweeps/box_4sweep.npy"
    #1sweep loading
    # pcd_path = "/home/CenterPoint/example_4sweeps/1sweep_loading.pcd"
    anno_path = "/home/CenterPoint/example_4sweeps/1sweep_loading.npy"
    #4sweep loading
    pcd_path = "/home/CenterPoint/example_4sweeps/4sweep_000740_copy.pcd"
    # pcd_path = "/home/stitch/ark_scene_new/scene/scene_041/320669_1637697561.083268.pcd"
    # anno_path = "/home/stitch/stitch_transformed/annotation/scene_021/lidar/000740_1637802079.031945.json"
### 1sweep
if nsweeps == 1:
    pcd_path = "/home/CenterPoint/scene.pcd"
    anno_path = "/home/CenterPoint/bbox.npy"

# pcd_path = "/home/CenterPoint/points.pcd"
# anno_path = "/home/CenterPoint/box.npy"
# anno_path = "/home/CenterPoint/box_org.npy"

## sweeps test
# pcd_path = "/home/stitch/stitch_transformed/annotated_data/scene_003/lidar/083152_1637637388.710230.pcd"

## sweeps rotate test
# pcd_path = "/home/CenterPoint/data/stitch/sweeps/LIDAR_TOP/000539_1637802058.931937.pcd"

# anno_path = "/home/CenterPoint/box.npy"
# pcd_1 = o3d.io.read_point_cloud("/home/CenterPoint/data/stitch/sweeps/LIDAR_TOP/083150_1637637388.510252.pcd")
# pcd_2 = o3d.io.read_point_cloud("/home/CenterPoint/data/stitch/sweeps/LIDAR_TOP/083151_1637637388.610252.pcd.pcd")
# pcd_3 = o3d.io.read_point_cloud("/home/CenterPoint/data/stitch/sweeps/LIDAR_TOP/083152_1637637388.710230.pcd.pcd")
# pcd_4 = o3d.io.read_point_cloud("/home/stitch_transformed/annotated_data/scene_003/lidar/083152_1637637388.710230.pcd")


### transform
show_open3d_pcd(pcd_path, anno_path)
