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
import pdb
from nuscenes.nuscenes import NuScenes
from layers import *
from utils import download_model_if_doesnt_exist

# Replace 'v1.0' with the appropriate version if needed

def quaternion_to_rotation_matrix(qut):
    # Extract quaternion elements
    # import pdb; pdb.set_trace()
    qw, qx, qy, qz = qut
    # qx, qy, qz, qw = qut

    # Calculate the elements of the rotation matrix
    R11 = 1 - 2*qy**2 - 2*qz**2
    R12 = 2*qx*qy - 2*qz*qw
    R13 = 2*qx*qz + 2*qy*qw
    R21 = 2*qx*qy + 2*qz*qw
    R22 = 1 - 2*qx**2 - 2*qz**2
    R23 = 2*qy*qz - 2*qx*qw
    R31 = 2*qx*qz - 2*qy*qw
    R32 = 2*qy*qz + 2*qx*qw
    R33 = 1 - 2*qx**2 - 2*qy**2

    return np.array([[R11, R12, R13, 0],
                     [R21, R22, R23, 0],
                     [R31, R32, R33, 0],
                     [0,   0,   0,   1]])


def create_extrinsic_matrix(translation, rotation_quaternion):
    # Convert quaternion to rotation matrix
    # import pdb; pdb.set_trace()
    rotation_matrix = quaternion_to_rotation_matrix(rotation_quaternion)
    # import pdb; pdb.set_trace()
    # Create the 4x4 extrinsic matrix
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix[:3, :3]
    extrinsic_matrix[:3, 3] = translation

    return extrinsic_matrix


# def pseudo_point_cloud(depth_map, intrinsic_matrix, extrinsic_matrix, ego_ext_inv):
#     # 이미지 크기
#     height, width = depth_map.shape

#     # Pseudo point cloud 생성을 위한 빈 배열
#     point_cloud = []

#     # Intrinsic matrix의 역행렬을 계산
#     K_inv = np.linalg.inv(intrinsic_matrix)
#     E_inv = np.linalg.inv(extrinsic_matrix)

#     # 이미지 상의 모든 픽셀을 순회하면서 depth map에서 깊이 정보를 얻음
#     for v in range(height):
#         for u in range(width):
#             depth = depth_map[v, u]

#             # depth가 0인 경우는 유효하지 않은 값으로 판단하고 건너뜀
#             if depth == 0:
#                 continue

#             # 카메라 좌표계에서의 2D 이미지 좌표를 homogeneous 좌표로 변환
#             p_cam_homogeneous = np.array([u, v, 1], dtype=np.float32)

#             # 깊이 정보를 추가하여 카메라 좌표계에서 3D 좌표를 얻음
#             p_cam_3d_homogeneous = depth * np.dot(K_inv, p_cam_homogeneous)
#             p_world_3d_homogeneous = np.dot(extrinsic_matrix, np.append(p_cam_3d_homogeneous, [1]))
#             # 카메라 좌표계에서 월드 좌표계로 변환
#             # p_world_3d_homogeneous = np.dot(E_inv, np.append(p_cam_3d_homogeneous, [1]))
#             # ego vehicle의 extrinsic matrix 적용
#             # p_world_3d_homogeneous = np.dot(ego_ext_inv, p_world_3d_homogeneous)
            
#             # homogeneous 좌표를 일반 좌표로 변환
#             p_world_3d = p_world_3d_homogeneous[:3]

#             # Pseudo point cloud에 좌표를 추가
#             point_cloud.append(p_world_3d)

#     return np.array(point_cloud)


def pseudo_point_cloud(depth_map, intrinsic_matrix, extrinsic_matrix):
    factor = 6
    # 이미지 크기
    height, width = depth_map.shape

    # 이미지 좌표 생성
    # u, v = np.meshgrid(np.linspace(0, 1600-1, width*factor), np.linspace(0, 900-1, height*factor))
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    # 이미지 좌표를 homogeneous 좌표로 변환
    p_cam_homogeneous = np.vstack((u, v, np.ones_like(u)))

    # 깊이 정보 추가하여 카메라 좌표계에서 3D 좌표 얻기
    depth = depth_map.flatten()
    # pdb.set_trace()
    # depth = np.meshgrid(np.linspace(0, 1600-1, width*factor))
    p_cam_3d_homogeneous = np.dot(np.linalg.inv(intrinsic_matrix), p_cam_homogeneous) * depth

    # 카메라 좌표계에서 월드 좌표계로 변환
    p_world_3d_homogeneous = np.dot(extrinsic_matrix, np.vstack((p_cam_3d_homogeneous, np.ones_like(u))))
    # p_world_3d_homogeneous = np.dot(np.linalg.inv(extrinsic_matrix), np.vstack((p_cam_3d_homogeneous, np.ones_like(u))))

    # homogeneous 좌표를 일반 좌표로 변환
    p_world_3d = p_world_3d_homogeneous[:3] / p_world_3d_homogeneous[3]

    # depth가 0인 경우에 대한 처리
    valid_indices = depth != 0
    p_world_3d = p_world_3d[:, valid_indices]

    return p_world_3d.T


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
    # import pdb; pdb.set_trace()
    # rot = o3d.geometry.get_rotation_matrix_from_quaternion(axis_angles)
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)
    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set, box3d

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    return parser.parse_args()

def resize_matrix(original_array):
    for i in range(original_array.shape[0]):
        for j in range(original_array.shape[1]):
            if original_array[i, j] != 0:
                new_i = int(i * row_scale)
                new_j = int(j * col_scale)
                # import pdb; pdb.set_trace()
                resized_array[new_i, new_j] = original_array[i, j]
    return resized_array

def test_simple(args, sample_data_list):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    # feed_height = 1600
    # feed_width = 900
    # import pdb; pdb.set_trace()
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()
    # import pdb; pdb.set_trace()
    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    image_path = os.path.join('../nuscenes/v1.0-trainval', sample_data_list['filename'])
    paths = [image_path]
    output_directory = os.path.dirname(image_path)
    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # pdb.set_trace()
            # Load image and preprocess
            input_image = pil.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)
            # PREDICTION
            input_image = input_image.to(device)
            # import pdb; pdb.set_trace()
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            # scaled_disp, depth_map = disp_to_depth(disp, 1, 40)
            scaled_disp, depth_map = disp_to_depth(disp, 0.1, 100)
            depth_map = depth_map.squeeze().cpu().numpy()
            disp = disp.squeeze().cpu().numpy()
            # import pdb; pdb.set_trace()
            camera_intrinsics = {
                'focal_length_x': cam_int[0][0],   # Focal length in X direction
                'focal_length_y': cam_int[1][1],   # Focal length in Y direction
                'principal_point_x': cam_int[0][2], # Principal point (X coordinate)
                'principal_point_y': cam_int[1][2]  # Principal point (Y coordinate)
            }

            # Example usage
            translation_vector = sample_data_list['cam_trans']
            rotation_quaternion = sample_data_list['cam_rot']

            extrinsic_matrix = create_extrinsic_matrix(translation_vector, rotation_quaternion)
            intrinsic_matrix = sample_data_list['cam_int']

            # Assuming you have a NumPy array representing the image with the shape (1, 1, 320, 1024)
            original_array = depth_map
            # original_array = 1/disp
            #shape: (320,1024)
            # import pdb; pdb.set_trace()

            # Define the new size
            new_size = (900, 1600)
            # new_size = (9000, 16000)

            # Calculate the row and column scaling factors
            row_scale = new_size[0] / original_array.shape[0]
            col_scale =  new_size[1]/  original_array.shape[1]
            # import pdb; pdb.set_trace()
            # Create the resized array and fill it with original values and zeros
            resized_array = np.zeros(new_size, dtype=original_array.dtype)

            # Iterate over each non-zero element in the original array and copy it to the corresponding position in the resized array
            for i in range(original_array.shape[0]):
                for j in range(original_array.shape[1]):
                    if original_array[i, j] != 0:
                        new_i = int(i * row_scale)
                        new_j = int(j * col_scale)

                        if (i%2 == 0) or (j%2 == 0) :
                            original_array[i, j] = 0
                        if (i%3 == 0) or (j%3 == 0) :
                            original_array[i, j] = 0
                        if (i%5 == 0) or (j%5 == 0) :
                            original_array[i, j] = 0
                        if (i%7 == 0) or (j%7 == 0) :
                            original_array[i, j] = 0
                        # import pdb; pdb.set_trace()
                        resized_array[new_i, new_j] = original_array[i, j]
            count = np.count_nonzero(resized_array)

            # Print the result
            print("Number of non-zero elements:", count)

            depth_map = resized_array
            # Pseudo point cloud 생성을 위한 빈 배열
            point_cloud = []
            ego_ext = create_extrinsic_matrix(sample_data_list['ego_trs'] ,sample_data_list['ego_rot'] )
            ego_ext_inv = np.linalg.inv(ego_ext)
            p_c = pseudo_point_cloud(depth_map, intrinsic_matrix, extrinsic_matrix)
            # p_c = ego_ext_inv @ p_c
            # print("pc: ",p_c.shape)  # Output: (900, 1600)

            # import pdb; pdb.set_trace()
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            name_dest_npy = os.path.join('./result', "{}_fusion.npy".format(output_name))
            np.save(name_dest_npy, p_c)

            # import pdb; pdb.set_trace()
            # 결과 출력
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(p_c)
            # # Visualize the PointCloud
            # o3d.visualization.draw_geometries([point_cloud])
    return p_c

if __name__ == '__main__':
    args = parse_args()
    nusc = NuScenes(version='v1.0-mini', dataroot='../nuscenes', verbose=True)
    
    my_scene = nusc.scene[2]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    
    sample_data_list = []
    npy_list = []
    # print(my_sample)
    
    sensor_list = ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']
    for sensor in tqdm(sensor_list):
        image_data = nusc.get('sample_data', my_sample['data'][sensor])
        # depth_gt = get_depth(nusc, my_sample)
        ego = nusc.get('ego_pose', image_data['ego_pose_token'])
        ego_rot = ego['rotation']
        ego_trs = ego['translation']
        # import pdb; pdb.set_trace()
        image_data['calibrated_sensor_token']
        cs = nusc.get('calibrated_sensor', image_data['calibrated_sensor_token'])
        cam_int = cs['camera_intrinsic']
        cam_rot = cs['rotation']
        cam_trans = cs['translation']
        sample_data_list = {'filename':image_data['filename'], 'channel':image_data['channel'], 'cam_int':cam_int,
        'cam_rot':cam_rot, 'cam_trans':cam_trans, 'ego_rot':ego_rot, 'ego_trs':ego_trs}#, 'depth':depth_gt}
        p_c = test_simple(args, sample_data_list)
        # sample_data_list.append({'filename':image_data['filename'], 'channel':image_data['channel'], cam_int':cam_int,
        # 'cam_rot':cam_rot, 'cam_trans':cam_trans})
        # print(sample_data_list)
        npy_list.append(p_c)
    # print(npy_list)
    # import pdb; pdb.set_trace()
    vis = o3d.visualization.Visualizer()
    # for my_annotation_token in my_sample['anns']:
    #     my_annotation_metadata =  nusc.get('sample_annotation', my_annotation_token)
    #     # w,l,h = my_annotation_metadata['size']
    #     w,h,l = my_annotation_metadata['size']
    #     x,y,z = my_annotation_metadata['translation']
    #     # z=z-2
    #     # x = -x
    #     orientation = my_annotation_metadata['rotation']
    #     gt_box = [x,y,z]+[h,w,l]+[orientation]+[ego_rot]
    #     line_set, _ = translate_boxes_to_open3d_instance(gt_box)
    #     line_set.paint_uniform_color((0,1,0))
    #     mat = o3d.visualization.rendering.MaterialRecord()
    #     mat.shader = "unlitLine"
    #     mat.line_width = 10
    #     vis.add_geometry(line_set)


    # vis.run()
    # # vis.run()
    # vis.destroy_window()
    # 6 npy concat 
    six_cam = np.concatenate((npy_list[0], npy_list[1], npy_list[2], npy_list[3], npy_list[4], npy_list[5]))
    
    ego_rot = quaternion_to_rotation_matrix(ego_rot)
    # import pdb; pdb.set_trace()
    ego_rot = ego_rot[:3, :3]
    ego_rot = np.linalg.inv(ego_rot)
    # six_cam = np.dot(six_cam,ego_rot)# @ ego_rot
    # six_cam = six_cam + ego_trs
    # six_cam[2] = six_cam[2] - 2
    # six_cam = six_cam @ ego_rot
    # kk = quaternion_to_rotation_matrix(ego_rot)
    # kk[:3,3] = ego_trs
    
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(six_cam)
    # Visualize the PointCloud
    # o3d.visualization.draw_geometries([point_cloud])

    # 시각화된 화면을 이미지 파일로 저장
    # o3d.visualization.draw_geometries([point_cloud], "visualization.png")
    # PointCloud를 시각화 창에 추가합니다.
    vis.create_window()
    vis.add_geometry(point_cloud)

    # 좌표계를 추가합니다.
    # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=ego_trs)
    # # import pdb; pdb.set_trace()
    # # coordinate_frame = coordinate_frame @ ego_rot
    # vis.add_geometry(coordinate_frame)

    # 시각화를 업데이트하고 보여줍니다.
    # vis.update_geometry()
    # vis.poll_events()
    # vis.update_renderer()
    vis.run()
    # o3d.visualization.draw_geometries([point_cloud])
    # test_simple(args, sample_data_list)