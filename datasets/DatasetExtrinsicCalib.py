import json
import logging
import os
import sys
from math import radians

import cv2
import mathutils
import numpy as np
import pandas as pd
import pykitti
import pyntcloud
import torch
import yaml
from PIL import Image
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.calibration import get_calibration_config
from argoverse.utils.json_utils import read_json_file
from pandaset.geometry import _heading_position_to_mat
from torch.utils.data import Dataset
from torchvision import transforms

from utils import invert_pose, rotate_forward, to_rotation_matrix, quaternion_from_matrix, tvector2mat

logging.getLogger('argoverse').setLevel(logging.ERROR)

import open3d as o3d


def se3_exp_map(omega, v):
    """
    Computes the exponential map for SE(3).
    :param omega: Rotation vector (axis-angle), shape (3,) [radians]
    :param v: Translation vector component (in tangent space), shape (3,) [meters]
    :return: 4x4 Transformation Matrix
    """
    theta = np.linalg.norm(omega)
    K = np.zeros((3, 3))
    K[0, 1] = -omega[2]
    K[0, 2] = omega[1]
    K[1, 0] = omega[2]
    K[1, 2] = -omega[0]
    K[2, 0] = -omega[1]
    K[2, 1] = omega[0]

    I = np.eye(3)
    if theta < 1e-6:
        R = I + K
        V = I + 0.5 * K
    else:
        R = I + (np.sin(theta) / theta) * K + ((1 - np.cos(theta)) / (theta ** 2)) * (K @ K)
        V = I + ((1 - np.cos(theta)) / (theta ** 2)) * K + ((theta - np.sin(theta)) / (theta ** 3)) * (K @ K)

    t = V @ v
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return torch.tensor(T, dtype=torch.float32)


class ReadOpen3d:
    def __call__(self, file):
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        return points


class ReadBinWithTime:
    def __call__(self, file):
        dtype = np.dtype([
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("intensity", "<f4"),
            ("time_ns", "<u4"),
        ])
        data = np.fromfile(file, dtype=dtype)
        return np.stack([data["x"], data["y"], data["z"], data["intensity"]], axis=1)


def is_image(img):
    extensions = ['.jpg', '.png', '.tiff', '.jpeg', '.bmp']
    return os.path.splitext(img)[1] in extensions


def read_calib_file(filepath):
    """Read in a calibration file and parse into a dictionary."""
    data = {}

    with open(filepath, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    return data


# Generic point cloud reader from https://github.com/PRBonn/kiss-icp
def _get_point_cloud_reader(file_extension, first_scan_file):
        """Attempt to guess with try/catch blocks which is the best point cloud reader to use for
        the given dataset folder. Supported readers so far are:
            - np.fromfile
            - trimesh.load
            - PyntCloud
            - open3d[optional]
        """
        # This is easy, the old KITTI format
        if file_extension == "bin":
            print("[WARNING] Reading .bin files, the only format supported is the KITTI format")

            class ReadKITTI:
                def __call__(self, file):
                    return np.fromfile(file, dtype=np.float32).reshape((-1, 4))

            return ReadKITTI()

        print('Trying to guess how to read your data')
        # first try open3d
        try:

            try_pcd = o3d.io.read_point_cloud(first_scan_file)
            if try_pcd.is_empty():
                # open3d binding does not raise an exception if file is unreadable or extension is not supported
                raise Exception("Generic Dataloader| Open3d PointCloud file is empty")

            return ReadOpen3d()
        except:
            pass

        try:
            import trimesh

            trimesh.load(first_scan_file)

            class ReadTriMesh:
                def __call__(self, file):
                    return np.asarray(trimesh.load(file).vertices)

            return ReadTriMesh()
        except:
            pass

        try:
            from pyntcloud import PyntCloud

            PyntCloud.from_file(first_scan_file)

            class ReadPynt:
                def __call__(self, file):
                    return PyntCloud.from_file(file).points[["x", "y", "z"]].to_numpy()

            return ReadPynt()
        except:
            print("[ERROR], File format not supported")
            sys.exit(1)


def get_scan_kitti(path, cam='2', kitti=None):
    scan = np.fromfile(path, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    split_path = path.split('/')
    base_folder = os.path.join('/', *split_path[:-4])
    if kitti is None:
        kitti = pykitti.odometry(base_folder, split_path[-3])
    if cam == '2' or cam == '02':
        cam_to_velo = torch.from_numpy(kitti.calib.T_cam2_velo).double()
        calib = kitti.calib.K_cam2
    elif cam == '3' or cam == '03':
        cam_to_velo = torch.from_numpy(kitti.calib.T_cam3_velo).double()
        calib = kitti.calib.K_cam3
    calib = torch.tensor([calib[0, 0], calib[1, 1], calib[0, 2], calib[1, 2]]).float()
    return scan, cam_to_velo.float(), calib


def get_scan_argo(path, camera):
    data = pyntcloud.PyntCloud.from_file(os.fspath(path))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]
    lidar_intensity = np.array(data.points.intensity, dtype=np.float32)[:, np.newaxis]
    lidar_pts = np.concatenate((x, y, z, lidar_intensity), axis=1)

    splitted_path = path.split('/')

    calib = read_json_file(path.replace(f'{splitted_path[-2]}/{splitted_path[-1]}', 'vehicle_calibration_info.json'))
    calib = get_calibration_config(calib, camera)
    cam2_to_velo = torch.from_numpy(calib.extrinsic)
    intrinsics = torch.tensor([calib.intrinsic[0, 0], calib.intrinsic[1, 1],
                               calib.intrinsic[0, 2], calib.intrinsic[1, 2]])

    return lidar_pts, cam2_to_velo.float(), intrinsics.float()


def get_scan_pandaset(path, sensor_id=0):
    scan = pd.read_pickle(path)
    scan = scan.loc[scan['d'] == sensor_id]

    return scan.values[:, :4]


def get_extrinsic_pandaset(camera):
    with open(os.path.join(os.path.dirname(__file__), 'pandaset_extrinsic.yaml')) as f:
        file_data = yaml.safe_load(f)
    camera_pose = file_data[camera]['extrinsic']['transform']
    camera_translation = torch.tensor([camera_pose['translation']['x'], camera_pose['translation']['y'],
                                       camera_pose['translation']['z']])
    camera_quaternion = torch.tensor([camera_pose['rotation']['w'], camera_pose['rotation']['x'],
                                      camera_pose['rotation']['y'], camera_pose['rotation']['z']])
    camera_pose = to_rotation_matrix(camera_quaternion, camera_translation)
    return camera_pose


class DatasetGeneralExtrinsicCalib(Dataset):

    def __init__(self, dataset_dirs, transform=None, augmentation=False, use_reflectance=False, max_t=2., max_r=10.,
                 train=True, normalize_images=True, dataset='kitti', cam='2', change_frame=False,data_type='default', image_name='image_left', pcl_name='lidar', downsample=False, fix_error =False, error_file=None, error_idx=0):
        super(DatasetGeneralExtrinsicCalib, self).__init__()
        self.dataset = dataset
        self.data_type = data_type
        self.use_reflectance = use_reflectance
        self.max_r = max_r
        self.max_t = max_t
        self.fix_error = fix_error
        self.augmentation = augmentation
        self.root_dirs = dataset_dirs
        self.transform = transform
        self.train = train
        self.normalize_images = normalize_images
        self.maps_folder = None
        self.extension = None
        self.cam = str(cam)
        self.camera_folder = f'image_{cam}'
        self.change_frame = change_frame
        if dataset == 'kitti':
            self.maps_folder = 'velodyne'
            self.extension = '.bin'
        elif dataset == 'argoverse':
            self.maps_folder = 'lidar'
            self.extension = '.ply'
            self.sdbs = {}
        elif dataset == 'custom':
            self.maps_folder = pcl_name
            if downsample:
                self.camera_folder = 'Downsample/camera'
            else:
                self.camera_folder = 'camera'
        elif data_type == 'lg_custom':
            self.maps_folder = pcl_name
            self.camera_folder = image_name
            self.downsample = downsample
            self.data_type = data_type

        self.all_files = []
        self.synced_stamps = []

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for directory in dataset_dirs:

            if dataset == 'argoverse':
                for log_id in sorted(os.listdir(directory)):
                    self.sdbs[log_id] = SynchronizationDB(directory, collect_single_log_id=log_id)

                    point_cloud_folder = os.path.join(directory, log_id, self.maps_folder)

                    sorted_filenames = sorted(os.listdir(point_cloud_folder))
                    for filename in sorted_filenames:
                        self.all_files.append(os.path.join(point_cloud_folder, filename))

            if dataset == 'custom':
                if downsample:
                    with open(os.path.join(directory, 'Downsample/calibration.yaml')) as f:
                        file_data = yaml.safe_load(f)
                else:
                    with open(os.path.join(directory, 'calibration.yaml')) as f:
                        file_data = yaml.safe_load(f)
                self.camera_intrinsics = torch.tensor(
                    [file_data['fx'], file_data['fy'], file_data['cx'], file_data['cy']])
                self.initial_extrinsic = torch.tensor(file_data['initial_extrinsic'], dtype=torch.float).reshape(4, 4)
                first_scan = os.listdir(os.path.join(directory, self.maps_folder))
                first_scan = sorted(first_scan)[0]
                self.extension = os.path.splitext(first_scan)[1]
                self.point_cloud_reader = _get_point_cloud_reader(self.extension[1:],
                                                                  os.path.join(directory, self.maps_folder, first_scan))

            if dataset == 'kitti' or dataset == 'custom':
                img_folder = os.path.join(directory, self.camera_folder)
                point_cloud_folder = os.path.join(directory, self.maps_folder)

                sorted_filenames = sorted(os.listdir(img_folder))
                for filename in sorted_filenames:
                    filename_no_extension = os.path.splitext(filename)[0]
                    point_cloud_path = os.path.join(point_cloud_folder, filename_no_extension + self.extension)
                    if not os.path.exists(point_cloud_path):
                        continue
                    self.all_files.append(os.path.join(img_folder, filename))

            if data_type == 'lg_custom':
                with open(os.path.join(directory, '../..', 'calibration.yaml')) as f:
                    file_data = yaml.safe_load(f)

                self.camera_intrinsics = torch.tensor(
                    [file_data['fx'], file_data['fy'], file_data['cx'], file_data['cy']])
                self.initial_extrinsic = torch.tensor(file_data['initial_extrinsic'], dtype=torch.float).reshape(4, 4)
                first_scan = os.listdir(os.path.join(directory,'sensor_data', self.maps_folder))
                first_scan = sorted(first_scan)[0]
                self.extension = os.path.splitext(first_scan)[1]

                self.point_cloud_reader = ReadBinWithTime()
                
                img_folder = os.path.join(directory, self.camera_folder)
                point_cloud_folder = os.path.join(directory, self.maps_folder)

                synced_stamp_path = os.path.join(directory, 'synced_stamps', self.camera_folder + '_' + self.maps_folder + '.txt')
                with open(synced_stamp_path, 'r') as f:
                    for line in f.read().splitlines():
                        
                        # skip first line
                        if 'image' in line:
                            continue

                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        image_stamp, maps_stamp = parts[0], parts[1]
                        self.synced_stamps.append((directory, image_stamp, maps_stamp))

                    print(f"Loaded {len(self.synced_stamps)} synced stamps from {synced_stamp_path}")

        self.use_error_file = False
        if self.fix_error:
            if error_file is not None:
                with open(error_file, 'r') as f:
                    lines = f.readlines()
                    if 0 <= error_idx < len(lines):
                        values = list(map(float, lines[error_idx].strip().split()))
                        # Assume input is: omega_x, omega_y, omega_z (degree), v_x, v_y, v_z (meter)
                        # We store them as is for now, conversion happens in __getitem__ or se3_exp_map call
                        omega = np.array(values[:3])
                        v = np.array(values[3:])
                        self.fixed_errors = (omega, v)
                        self.use_error_file = True
                        print(f"Using FIXED SE(3) error from FILE {error_file} (idx {error_idx}): w={omega}(deg), v={v}(m)")
                    else:
                        raise ValueError(f"Invalid error_idx {error_idx} for file {error_file} with {len(lines)} lines")
            else:
                rotz = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                roty = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                rotx = np.random.uniform(-self.max_r, self.max_r) * (3.141592 / 180.0)
                transl_x = np.random.uniform(-self.max_t, self.max_t)
                transl_y = np.random.uniform(-self.max_t, self.max_t)
                transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))
                self.fixed_errors = (rotx, roty, rotz, transl_x, transl_y, transl_z)
                print(f"Using FIXED error for ALL frames : {self.fixed_errors} ...")

    def custom_transform(self, rgb, calib, img_rotation=0., flip=False):
        if self.train:
            color_transform = transforms.ColorJitter(0.2, 0.2, 0.2) # 밝기, 대비, 채도 변화 20% 수준
            rgb = color_transform(rgb)
        rgb = np.array(rgb)
        if self.train:
            if flip:
                rgb = cv2.flip(rgb, 1)
            height, width = rgb.shape[:2]
            matrix = cv2.getRotationMatrix2D(tuple(calib[2:].numpy()), img_rotation, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, dsize=(width, height))

        return torch.tensor(rgb).float()

    def __len__(self):
        if self.data_type == 'lg_custom':
            return len(self.synced_stamps)
        return len(self.all_files)

    def __getitem__(self, idx):
        if self.dataset == 'kitti' or self.dataset == 'custom':
            img_path = self.all_files[idx]
            extension = os.path.basename(img_path)
            extension = os.path.splitext(extension)[1]
            pc_path = img_path.replace(f'/{self.camera_folder}/', f'/{self.maps_folder}/').replace(extension,
                                                                                                   self.extension)
            
        elif self.data_type == 'lg_custom':
            # read stamps from self.synced_stamps
            directory, image_stamp, maps_stamp = self.synced_stamps[idx]
            img_path = os.path.join(directory, 'sensor_data', self.camera_folder, f'{image_stamp}.png')
            pc_path = os.path.join(directory, 'sensor_data', self.maps_folder,
                                   f'{maps_stamp}{self.extension}')
        elif self.dataset == 'argoverse':
            pc_path = self.all_files[idx]

            splitted_path = pc_path.split('/')
            lidar_stamp = int(splitted_path[-1][3:-4])
            log_id = splitted_path[-3]
            sdb = self.sdbs[log_id]

        if self.dataset == 'kitti':
            pc, cam2vel, calib = get_scan_kitti(pc_path, cam=self.cam)
        elif self.dataset == 'argoverse':
            cam_timestamp = sdb.get_closest_cam_channel_timestamp(lidar_stamp, self.cam, log_id)

            img_path = pc_path.replace('/' + self.maps_folder + '/', f'/{self.cam}/')
            img_path = img_path.replace(splitted_path[-1], f'{self.cam}_{cam_timestamp}.png')

            pc, cam2vel, calib = get_scan_argo(pc_path, self.cam)
        elif self.dataset == 'custom':
            pc = self.point_cloud_reader(pc_path)
            cam2vel = self.initial_extrinsic
            calib = self.camera_intrinsics.clone()
            if self.maps_folder == 'radar':
                if pc.shape[0] == 0 or pc.shape[1] < 3:
                    print(f"[WARNING] Empty or invalid point cloud at {pc_path}, resampling")
                    new_idx = np.random.randint(0, self.__len__())
                    return self.__getitem__(new_idx)
                valid_mask = pc[:, 2] >= -1.0
                if not np.any(valid_mask):
                    print(f"[WARNING] All points below z=-1.0 for {pc_path}, resampling")
                    new_idx = np.random.randint(0, self.__len__())
                    return self.__getitem__(new_idx)
                pc = pc[valid_mask]

            if pc.shape[1] == 3:
                pc = np.concatenate((pc, np.ones((pc.shape[0], 1))), 1)
            elif pc.shape[1] >= 4:
                pc = pc[:, :4]
            else:
                print("[ERROR], Point cloud has less than 3 channels")
                sys.exit(1)

        elif self.data_type == 'lg_custom':
            pc = self.point_cloud_reader(pc_path)
            cam2vel = self.initial_extrinsic
            calib = self.camera_intrinsics.clone()

            if pc.shape[1] == 3:
                pc = np.concatenate((pc, np.ones((pc.shape[0], 1))), 1)
            elif pc.shape[1] >= 4:
                pc = pc[:, :4]
            else:
                print("[ERROR], Point cloud has less than 3 channels")
                sys.exit(1)

            if not os.path.exists(img_path):
                print(f"[WARNING] Missing image for stamp {img_path}, resampling")
                new_idx = np.random.randint(0, self.__len__())
                return self.__getitem__(new_idx)
            if not os.path.exists(pc_path):
                print(f"[WARNING] Missing point cloud for stamp {pc_path}, resampling")
                new_idx = np.random.randint(0, self.__len__())
                return self.__getitem__(new_idx)

        if self.use_reflectance:
            reflectance = torch.from_numpy(pc[:, -1]).float()
        pc[:, -1] = 1

        pc_in = torch.from_numpy(pc.astype(np.float32))
        pc_in = torch.mm(cam2vel, pc_in.t())
        if self.change_frame:
            pc_in = pc_in[[2, 0, 1, 3], :]

        img = Image.open(img_path)
        h_mirror = False # 좌우 반전
        # if np.random.rand() > 0.5 and self.train:
        #     h_mirror = True
        #     if self.change_frame:
        #         pc_in[1, :] *= -1
        #     else:
        #         pc_in[0, :] *= -1
        #     calib[2] = img.size[0] - calib[2] # cx 변경

        img_rotation = 0.
        if self.train:
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, calib, img_rotation, h_mirror) # 이미지 flip 및 회전
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.train:
            if self.change_frame:
                R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            else:
                R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        if self.fix_error and self.use_error_file:
            omega_deg, v = self.fixed_errors
            omega_rad = omega_deg * (np.pi / 180.0)
            
            # dT = exp([omega, v]) in SE(3)
            dT = se3_exp_map(omega_rad, v) # 4x4 tensor
            
            # T_GT = cam2vel (assuming cam2vel takes point from Lidar/Sensor to Camera)
            # Check type of cam2vel
            T_GT = cam2vel
            if not isinstance(T_GT, torch.Tensor):
                T_GT = torch.tensor(T_GT, dtype=torch.float32)
            
            # M = T_GT * dT * T_GT^-1
            # extrinsic_error in evaluate code is M.
            # We need to return components of M^-1 (because code does RT1_inv = T*R, then extrinsic_error = RT1_inv.inverse())
            # So RT1_inv = M^-1
            
            M = torch.mm(T_GT, torch.mm(dT, T_GT.inverse()))
            M_inv = M.inverse()
            
            # Decompose M_inv into Translation T and Rotation R
            # M_inv = [R  t]
            #         [0  1]
            # But code expects RT1_inv = T_mat * R_mat (where T_mat is translation only, R_mat is rotation only)
            # T_mat * R_mat = [I t] * [R 0] = [R t]
            #                 [0 1]   [0 1]   [0 1]
            # So yes, standard decomposition fits.
            
            # Extract rotation and translation
            R_mat = M_inv[:3, :3] # 3x3
            T_vec = M_inv[:3, 3]  # 3
            
            R = quaternion_from_matrix(R_mat) # vector 4
            T = T_vec # vector 3
            
        elif self.fix_error:
            rotx, roty, rotz, transl_x, transl_y, transl_z = self.fixed_errors
            # Fallback to Euler perturbation logic
            if self.change_frame:
                R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
                T = mathutils.Vector((transl_x, transl_y, transl_z))
            else:
                R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
                T = mathutils.Vector((transl_y, transl_z, transl_x))
            R, T = invert_pose(R, T)
            R, T = torch.tensor(R), torch.tensor(T)
        else:
            max_angle = self.max_r
            rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
            transl_x = np.random.uniform(-self.max_t, self.max_t)
            transl_y = np.random.uniform(-self.max_t, self.max_t)
            transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))

            if self.change_frame:
                R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
                T = mathutils.Vector((transl_x, transl_y, transl_z))
            else:
                R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
                T = mathutils.Vector((transl_y, transl_z, transl_x))
            R, T = invert_pose(R, T)
            R, T = torch.tensor(R), torch.tensor(T)

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'tr_error': T,
                  'rot_error': R, 'rgb_name': img_path, 'idx': idx, 'cam2vel': cam2vel}
        if self.use_reflectance:
            sample['reflectance'] = reflectance

        return sample


class DatasetPandasetExtrinsicCalib(Dataset):

    def __init__(self, dataset_dirs, transform=None, augmentation=False, use_reflectance=False, max_t=2., max_r=10.,
                 train=True, normalize_images=True, sensor_id=0, camera='front_camera', change_frame=False):
        super(DatasetPandasetExtrinsicCalib, self).__init__()
        self.use_reflectance = use_reflectance
        self.max_r = max_r
        self.max_t = max_t
        self.augmentation = augmentation
        self.root_dirs = dataset_dirs
        self.transform = transform
        self.train = train
        self.normalize_images = normalize_images
        self.sensor_id = sensor_id
        self.maps_folder = 'lidar'
        self.extension = 'pkl'
        self.camera = camera

        self.all_files = []
        self.camera_poses = []
        self.camera_stamps = []

        self.change_frame = change_frame

        if not isinstance(dataset_dirs, list):
            dataset_dirs = [dataset_dirs]

        for directory in dataset_dirs:
            point_cloud_folder = os.path.join(directory, self.maps_folder)

            sorted_filenames = sorted(os.listdir(point_cloud_folder))
            for filename in sorted_filenames:
                if '.json' not in filename:
                    self.all_files.append(os.path.join(point_cloud_folder, filename))

            pose_file = os.path.join(directory, 'camera', camera, 'poses.json')
            timestamp_file = os.path.join(directory, 'camera', camera, 'timestamps.json')
            with open(pose_file, 'r') as f:
                file_data = json.load(f)
                for entry in file_data:
                    self.camera_poses.append(_heading_position_to_mat(entry['heading'], entry['position']))

            with open(timestamp_file, 'r') as f:
                file_data = json.load(f)
                for entry in file_data:
                    self.camera_stamps.append(entry)

    def custom_transform(self, rgb, calib, img_rotation=0., flip=False):
        if self.train:
            color_transform = transforms.ColorJitter(0.2, 0.2, 0.2)
            rgb = color_transform(rgb)
        rgb = np.array(rgb)
        if self.train:
            if flip:
                rgb = cv2.flip(rgb, 1)
            height, width = rgb.shape[:2]
            matrix = cv2.getRotationMatrix2D(tuple(calib[2:].numpy()), img_rotation, 1.0)
            rgb = cv2.warpAffine(rgb, matrix, dsize=(width, height))

        return torch.tensor(rgb).float()

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        pc_path = self.all_files[idx]
        img_path = pc_path.replace('/' + self.maps_folder + '/', f'/camera/{self.camera}/').replace(self.extension,'jpg')
        # Get the camera intrinsic parameters
        calib_file = os.path.dirname(img_path)
        calib_file = os.path.join(calib_file, 'intrinsics.json')
        with open(calib_file, 'r') as f:
            calib = json.load(f)
        calib = torch.tensor([calib['fx'], calib['fy'], calib['cx'], calib['cy']]).float()

        pc = get_scan_pandaset(pc_path, self.sensor_id)
        if self.use_reflectance:
            reflectance = torch.from_numpy(pc[:, -1]).float()
        pc[:, -1] = 1
        cam_pose = torch.from_numpy(self.camera_poses[idx]).float().inverse()
        pc_in = torch.from_numpy(pc.astype(np.float32))
        pc_in = torch.mm(cam_pose, pc_in.t())
        if self.change_frame:
            pc_in = pc_in[[2, 0, 1, 3], :]

        cam2vel = get_extrinsic_pandaset(self.camera)


        img = Image.open(img_path)
        h_mirror = False
        if np.random.rand() > 0.5 and self.train:
            h_mirror = True
            if self.change_frame:
                pc_in[1, :] *= -1
            else:
                pc_in[0, :] *= -1
            calib[2] = img.size[0] - calib[2]

        img_rotation = 0.
        if self.train:
            img_rotation = np.random.uniform(-5, 5)
        try:
            img = self.custom_transform(img, calib, img_rotation, h_mirror)
        except OSError:
            new_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(new_idx)

        # Rotate PointCloud for img_rotation
        if self.train:
            if self.change_frame:
                R = mathutils.Euler((radians(img_rotation), 0, 0), 'XYZ')
            else:
                R = mathutils.Euler((0, 0, radians(img_rotation)), 'XYZ')
            T = mathutils.Vector((0., 0., 0.))
            pc_in = rotate_forward(pc_in, R, T)

        max_angle = self.max_r
        rotz = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        roty = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        rotx = np.random.uniform(-max_angle, max_angle) * (3.141592 / 180.0)
        transl_x = np.random.uniform(-self.max_t, self.max_t)
        transl_y = np.random.uniform(-self.max_t, self.max_t)
        transl_z = np.random.uniform(-self.max_t, min(self.max_t, 1.))

        if self.change_frame:
            R = mathutils.Euler((rotx, roty, rotz), 'XYZ')
            T = mathutils.Vector((transl_x, transl_y, transl_z))
        else:
            R = mathutils.Euler((roty, rotz, rotx), 'XYZ')
            T = mathutils.Vector((transl_y, transl_z, transl_x))

        R, T = invert_pose(R, T)
        R, T = torch.tensor(R), torch.tensor(T)

        sample = {'rgb': img, 'point_cloud': pc_in, 'calib': calib, 'tr_error': T,
                  'rot_error': R, 'rgb_name': img_path, 'idx': idx, 'cam2vel': cam2vel}
        if self.use_reflectance:
            sample['reflectance'] = reflectance

        return sample
