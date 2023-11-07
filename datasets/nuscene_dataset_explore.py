from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from PIL import Image
from pyquaternion import Quaternion
import numpy as np
import os
import os.path as osp
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix

nusc = NuScenes(version='v1.0-mini', dataroot='/share/nuscenes', verbose=True)
nusc_scenes = nusc.list_scenes()

current_scene = nusc.scene[0]
sensor = 'CAM_FRONT'
first_token = current_scene['first_sample_token']
first_token_meta = nusc.get('sample', first_token)
all_sensor_data = first_token_meta['data']
CAM_SAMPLE = nusc.get('sample_data', first_token_meta['data'][sensor])
CAM_EGO= CAM_SAMPLE['ego_pose_token']
EGO = nusc.get('ego_pose',token=CAM_EGO)

# map point cloud into image

def map_pointcloud_to_image(nusc,
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0,
                            render_intensity: bool = False,
                            show_lidarseg: bool = False):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.
    :param pointsensor_token: Lidar/radar sample_data token.
    :param camera_token: Camera sample_data token.
    :param min_dist: Distance from the camera below which points are discarded.
    :param render_intensity: Whether to render lidar intensity instead of point depth.
    :param show_lidarseg: Whether to render lidar intensity instead of point depth.
    :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
        or the list is empty, all classes will be displayed.
    :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                    predictions for the sample.
    :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
    """

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)
    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    if pointsensor['sensor_modality'] == 'lidar':
        # Ensure that lidar pointcloud is from a keyframe.
        assert pointsensor['is_key_frame'], \
            'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'
        pc = LidarPointCloud.from_file(pcl_path)
    else:
        pc = RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).
    depths = pc.points[2, :]

    if render_intensity:
        assert pointsensor['sensor_modality'] == 'lidar', 'Error: Can only render intensity for lidar, ' \
                                                          'not %s!' % pointsensor['sensor_modality']
        # Retrieve the color from the intensities.
        # Performs arbitary scaling to achieve more visually pleasing results.
        intensities = pc.points[3, :]
        intensities = (intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))
        intensities = intensities ** 0.1
        intensities = np.maximum(0, intensities - 0.5)
        coloring = intensities

    else:
        # Retrieve the color from the depth.
        coloring = depths

    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
    # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
    # casing for non-keyframes which are slightly out of sync.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
    points = points[:, mask]
    coloring = coloring[mask]

    return points, coloring, im

# lidar_token = nusc.get('sample_data', first_token_meta['data']['LIDAR_TOP'])
lidar_token = first_token_meta['data']['LIDAR_TOP']
# camera_token = nusc.get('sample_data', first_token_meta['data']['CAM_FRONT'])
camera_token = first_token_meta['data']['CAM_FRONT']
pts, depth, img = map_pointcloud_to_image(nusc, lidar_token, camera_token)
print("X")

def get_relative_pose(pose1, pose2):
    """
    calculate relative from pose1 to pose2 in the global frame
    :param from_pose:
    :param to_pose:
    :return:
    """
    H_pose1 = np.zeros((4, 4))
    H_pose2 = np.zeros((4, 4))
    H_pose1[3,3] = 1
    H_pose2[3, 3] = 1

    pose1_rot = Quaternion(pose1['rotation']).rotation_matrix
    pose1_tran = pose1['translation']
    H_pose1[0:3,0:3] = pose1_rot
    H_pose1[0:3,3] = pose1_tran



    pose2_rot = Quaternion(pose2['rotation']).rotation_matrix
    pose2_tran = pose2['translation']
    H_pose2[0:3,0:3] = pose2_rot
    H_pose2[0:3,3] = pose2_tran

    H_pose1_inv = np.linalg.inv(H_pose1)
    relative_pose_matrix = np.dot(H_pose1_inv, H_pose2)
    return relative_pose_matrix
    #
    # pose_1_inv = np.linalg.inv(pose1_rot)
    # rot_pose1_to_pose2 = np.dot(pose_1_inv, pose2_rot)
    # tran_pose1_to_pose2 = np.array(pose2_tran) - np.array(pose1_tran)
    # relative_pose_matrix[0:3, 0:3] = rot_pose1_to_pose2
    # relative_pose_matrix[3, 0:3] = tran_pose1_to_pose2
    # relative_pose_matrix[3,3] = 1
    # return relative_pose_matrix

CAM_SAMPLE = nusc.get('sample_data', first_token_meta['data'][sensor])
CAM_EGO= CAM_SAMPLE['ego_pose_token']
EGO = nusc.get('ego_pose',token=CAM_EGO)

CAM_SAMPLE_next_token = first_token_meta['next']
next_sample = nusc.get('sample', token=CAM_SAMPLE_next_token)
CAM_SAMPLE_next = nusc.get('sample_data', token=next_sample['data'][sensor])
CAM_EGO_next= CAM_SAMPLE_next['ego_pose_token']
EGO_next = nusc.get('ego_pose',token=CAM_EGO_next)

current_to_next = get_relative_pose(EGO,EGO_next)

H_pose1 = np.zeros((4, 4))
H_pose2 = np.zeros((4, 4))
H_pose1[3, 3] = 1
H_pose2[3, 3] = 1
pose1_rot = Quaternion(EGO['rotation']).rotation_matrix
pose1_tran = EGO['translation']
H_pose1[0:3, 0:3] = pose1_rot
H_pose1[0:3,3] = pose1_tran

pose2_rot = Quaternion(EGO_next['rotation']).rotation_matrix
pose2_tran = EGO_next['translation']
H_pose2[0:3, 0:3] = pose2_rot
H_pose2[0:3,3] = pose2_tran

H2_recovered = np.dot(H_pose1, current_to_next)



pose1_rot = Quaternion(pose1['rotation']).rotation_matrix
pose1_tran = pose1['translation']
H_pose1[0:3, 0:3] = pose1_rot
H_pose1[:, 3] = pose1_tran



pose2_rot = Quaternion(EGO_next['rotation']).rotation_matrix
recovered = np.dot(pose2_rot,current_to_next[0:3,0:3].T)
pose1_rot = Quaternion(EGO['rotation']).rotation_matrix
recovered_pose1_rot= list(Quaternion(matrix=recovered))


H1_global = np.zeros((4,4))
H1_global[0:3,0:3] = pose1_rot
H1_global[3,0:3] = EGO['translation']
H1_global[3,3] = 1


H2_global = np.zeros((4,4))
H2_global[0:3,0:3] = pose2_rot
H2_global[3,0:3] = EGO_next['translation']
H2_global[3,3] = 1

recovered_H1 = np.dot(H2_global,current_to_next.T)

# pt[0] horizontal index, < im.size[0]
# pt[1] vertical index, < im.size[1]
# depth, depth of the pcs
# for point cloud vertival range, 198-898
# image crop 1600x900 -> 1600x700
# for mono depth, the image must be of multiples of 32, the depth map and the image vertical size could be 1600 x 640,
# ie. crop image from 240 to 880
