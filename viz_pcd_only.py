import open3d as o3d
import numpy as np
import json
import math
import copy


### Annotation Rotation Function
def trans(offset , rotation):
    T = np.eye(4)
    if offset == 'x':
        T = [[1,  0,  0], [0,  np.cos(rotation),  -np.sin(rotation)], [0,  np.sin(rotation),  np.cos(rotation)]]

    elif offset == 'y':
        T = [[np.cos(rotation),  0,  -np.sin(rotation)],  [0,  1,  0],  [np.sin(rotation),  0,  np.cos(rotation)]]

    else:
        T = [[np.cos(rotation),  -np.sin(rotation),  0], [np.sin(rotation),  np.cos(rotation),  0], [0,  0,  1]]
    
    return T

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
    # import pdb; pdb.set_trace()

    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    # import pdb;pdb.set_trace()
    # import pdb; pdb.set_trace()
    # center[2]= gt_boxes[5]/2

    #### annotation rotate
    # return_x = center[0]
    # return_y = center[1]
    # return_z = center[2]
    # # # center[0]= 0
    # center[1]= 0
    # center[2]= 0
    # center = trans('x', - np.pi / 12) @ np.transpose(center) #@ np.transpose(rot)
    # center[1] = return_y
    # center[2] = return_z

    # ####Transform
    # T = np.eye(4)
    # # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, np.pi / 12, 0))
    # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((np.pi / 12, 0, 0))
    # # T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi / 12))

    # T[0, 3] = 0
    # T[1, 3] = 0
    # T[2, 3] = 5.5
    # pcd = copy.deepcopy(pcd).transform(T)

    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d

def show_open3d_pcd(pcd_path, show_origin=True, origin_size=3, show_grid=True):
    # import pdb; pdb.set_trace()
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.zeros(3)
    pcd = o3d.io.read_point_cloud(pcd_path)

    np_colors = np.array(pcd.colors) / 255
    pcd.colors = o3d.utility.Vector3dVector(np_colors)
    
    vis.add_geometry(pcd)
    ### add
    # np_pcd = np.asarray(copy.deepcopy(pcd.points))
    # ground = copy.deepcopy(pcd)
    # ground_x = np.random.uniform(-146,82,100000)
    # ground_y = np.random.uniform(-123,114,100000)
    # ground_z = np.full((100000,3),0)
    # ground_z[:,0], ground_z[:,1] = ground_x,ground_y
    # ground.points = o3d.utility.Vector3dVector(ground_z)
    # colors = np.zeros_like(ground.points)
    # ground.colors= o3d.utility.Vector3dVector(colors)
    # np_pcd = np.concatenate([np_pcd,ground_z])
    # pcd.points = o3d.utility.Vector3dVector(np_pcd)

    # np_pcd = np.asarray(pcd.points)

    # with open(anno_path,"r") as f:
    #     anno = json.load(f)
    # num_obj = anno['annotation_metadata']['object_num']
    # for i in range(num_obj):
    #     obj_attr = anno['annotation_metadata']['object_list'][i]
    #     h,w,l = obj_attr['bbox_size']
    #     x,y,z = obj_attr['bbox_center']
    #     lines = np.array(obj_attr['bbox_vertices'])
    #     orientation = math.atan2(lines[0][1]-lines[1][1],lines[0][0]-lines[1][0])
    #     gt_box = [x,y,z]+[h,w,l]+[orientation]
    #     line_set, box3d = translate_boxes_to_open3d_instance(gt_box)
    #     line_set.paint_uniform_color((0,1,0))
    #     vis.add_geometry(line_set)

    # pcd = copy.deepcopy(pcd).transform(T)
    vis.run()
    vis.destroy_window()
pcd_path = "./output.pcd"
# pcd_path = "../nuscenes/v1.0-trainval/samples/LIDAR/n008-2018-05-21-11-06-59-0400__LIDAR_TOP__1526915243047392.pcd.bin"
# anno_path = "/home/CenterPoint/box_11.npy"
# pcd_path = "./ark_scene_new/scene/scene_021/000737_1637802078.731925.pcd"
# anno_path = "/home/stitch/annotation_result_lidar/scene_001/lidar/008630_1637629936.465564.json"

# import pdb; pdb.set_trace()
show_open3d_pcd(pcd_path)