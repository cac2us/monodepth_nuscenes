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
   
    # pcd = copy.deepcopy(pcd).transform(T)
    pcd.paint_uniform_color((1, 1, 1))
    vis.add_geometry(pcd)
    generate_plane(vis)
    # with open(anno_path, "r") as f:
    #     anno = np.load(f)
    # num_obj = anno['annotation_metadata']['object_num']
    anno = np.load(anno_path)
    num_obj = 500

    ### pkl test
    # for i in range(num_obj):
    #     # obj_attr = anno['annotation_metadata']['object_list'][i]
    #     # obj_attr = anno[0][i][9]
    #     # h,w,l = obj_attr['bbox_size']
    #     # x,y,z = obj_attr['bbox_center']

    #     w,l,h = anno[i][3:6]
    #     x,y,z = anno[i][:3]
        
    #     orientation = -anno[i][6]



    #     # w,l,h = anno[0][i][3:6]
    #     # x,y,z = anno[0][i][:3]        
    #     # orientation = -anno[0][i][6]
    #     # x,y,z = rotate_points(x,y,z)
    #     gt_box = [x,y,z]+[h,w,l]+[orientation]
    #     line_set, _ = translate_boxes_to_open3d_instance(gt_box)
    #     line_set.paint_uniform_color((0,1,0))
    #     vis.add_geometry(line_set)

    for i in range(num_obj):
        obj_attr = anno['annotation_metadata']['object_list'][i]
        # obj_attr = anno[0][i][9]
        h,w,l = obj_attr['bbox_size']
        x,y,z = obj_attr['bbox_center']
        # h,w,l = anno[0][i][3:6]        
        # w,l,h = anno[0][i][3:6]
        # x,y,z = anno[0][i][:3]
        import pdb; pdb.set_trace()
        
        orientation = 0

        # w,l,h = anno[0][i][3:6]
        # x,y,z = anno[0][i][:3]        
        # orientation = -anno[0][i][6]
        # x,y,z = rotate_points(x,y,z)
        gt_box = [x,y,z]+[h,w,l]+[orientation]
        line_set, _ = translate_boxes_to_open3d_instance(gt_box)
        line_set.paint_uniform_color((0,1,0))
        vis.add_geometry(line_set)

    ### box test
    # for i in range(num_obj):
    #     # obj_attr = anno['annotation_metadata']['object_list'][i]
    #     # obj_attr = anno[0][i][9]
    #     # h,w,l = obj_attr['bbox_size']
    #     # x,y,z = obj_attr['bbox_center']
    #     h,w,l = anno[0][i][3:6]        
    #     # w,l,h = anno[0][i][3:6]
    #     x,y,z = anno[0][i][:3]
        
    #     orientation = -anno[0][i][6]-np.pi/2

    #     # w,l,h = anno[0][i][3:6]
    #     # x,y,z = anno[0][i][:3]        
    #     # orientation = -anno[0][i][6]
    #     # x,y,z = rotate_points(x,y,z)
    #     gt_box = [x,y,z]+[h,w,l]+[orientation]
    #     line_set, _ = translate_boxes_to_open3d_instance(gt_box)
    #     line_set.paint_uniform_color((0,1,0))
    #     vis.add_geometry(line_set)

    ### box_org test
    # for i in range(num_obj):
    #     # obj_attr = anno['annotation_metadata']['object_list'][i]
    #     # obj_attr = anno[0][i][9]
    #     # h,w,l = obj_attr['bbox_size']
    #     # x,y,z = obj_attr['bbox_center']

    #     # w,l,h = anno[i][3:6]
    #     # x,y,z = anno[i][:3]
    #     w,l,h = anno[2 * i + 1]
    #     x,y,z = anno[2 * i]
    #     # orientation = -anno[i][6]
    #     orientation = 0       



    #     # w,l,h = anno[0][i][3:6]
    #     # x,y,z = anno[0][i][:3]        
    #     # orientation = -anno[0][i][6]
    #     # x,y,z = rotate_points(x,y,z)
    #     gt_box = [x,y,z]+[h,w,l]+[orientation]
    #     line_set, _ = translate_boxes_to_open3d_instance(gt_box)
    #     line_set.paint_uniform_color((0,1,0))
    #     vis.add_geometry(line_set)

    vis.run()
    vis.destroy_window()
# T = np.eye(4)
# T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 12, 0))
# T[0, 3] = 0
# T[1, 3] = 0
# print(T)
### point too much
# pcd_path = "/home/stitch/stitch_transformed/annotated_data/scene_002/lidar/058880_1637634961.500939.pcd"
# anno_path = "/home/stitch/stitch_transformed/annotation/scene_002/lidar/058880_1637634961.500939.json"

### point too little
pcd_path = "/home/CenterPoint/points_11.pcd"
anno_path = "/home/CenterPoint/box_11.npy"
# anno_path = "/home/CenterPoint/box_org.npy"

### transform
show_open3d_pcd(pcd_path,anno_path)
