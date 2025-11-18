import argparse
import gc
import math
import os
import random
import time
from matplotlib import cm, rcParams
from rich.console import Console
from rich.table import Table
from rich import box
from PIL import Image

import cv2
import numpy as np

import torch
import torch.nn.parallel
import torch.utils.data
import visibility
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from datasets.DatasetExtrinsicCalib import DatasetGeneralExtrinsicCalib, DatasetPandasetExtrinsicCalib
from camera_model import CameraModel

from models.get_model import get_model
from quaternion_distances import quaternion_loss

# import matplotlib
# matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from utils import (downsample_depth, merge_inputs, get_flow_zforward, quat2mat, tvector2mat,
                   quaternion_from_matrix, EndPointError, rotate_forward, quaternion_median,
                   average_quaternions, rotate_back, quaternion_mode, str2bool, overlay_imgs)

rcParams["figure.raise_window"] = False
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCH = 1

output_dir = 'output'


def _init_fn(worker_id, seed):
    seed = seed + worker_id + EPOCH * 100
    print(f"Init worker {worker_id} with seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def quaternion_distance(q, r):
    dist = quaternion_loss(q.unsqueeze(0), r.unsqueeze(0), q.device)

    dist = 180. * dist.item() / math.pi
    return dist


def prepare_input(cam_params, pc_rotated, real_shape, reflectance, _config, change_frame=False):
    cam_model = CameraModel()
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]
    uv, depth, _, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance, change_frame)
    uv = uv.t().int().contiguous()
    depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
    depth_img += 1000.
    depth_img = visibility.depth_image(uv, depth, depth_img, uv.shape[0], real_shape[1], real_shape[0])
    depth_img[depth_img == 1000.] = 0.

    depth_img_no_occlusion = torch.zeros_like(depth_img, device='cuda')
    depth_img_no_occlusion = visibility.visibility2(depth_img, cam_params, depth_img_no_occlusion,
                                                    depth_img.shape[1], depth_img.shape[0],
                                                    _config['occlusion_threshold'], _config['occlusion_kernel'])

    uv = uv.long()

    # Check valid indexes: as multiple points MIGHT be projected into the same pixel, keep only
    # the points that are actually projected
    indexes = depth_img_no_occlusion[uv[:, 1], uv[:, 0]] == depth
    if _config['use_reflectance']:
        refl_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
        refl_img[uv[indexes, 1], uv[indexes, 0]] = refl[0, indexes]

    depth_img_no_occlusion /= _config['max_depth']
    depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)

    if _config['use_reflectance']:
        depth_img_no_occlusion = torch.cat((depth_img_no_occlusion, refl_img.unsqueeze(0)))

    uv = uv[indexes]
    return depth_img_no_occlusion, uv, indexes, depth


def downsample_and_pad(_config, rgb, depth_img_no_occlusion, img_shape, real_shape, flow_img, flow_mask):
    shape_pad = [0, 0, 0, 0]

    if _config['dataset'] in ['argoverse', 'pandaset'] or _config['downsample']:
        rgb = nn.functional.interpolate(rgb.unsqueeze(0), scale_factor=0.5)[0]
        depth_img_no_occlusion = downsample_depth(depth_img_no_occlusion.permute(1, 2, 0).contiguous(), 2)
        depth_img_no_occlusion = depth_img_no_occlusion.permute(2, 0, 1)

        shape_pad[3] = (img_shape[0] - real_shape[0] // 2)  # // 2
        shape_pad[1] = (img_shape[1] - real_shape[1] // 2)  # // 2 + 1

        rgb = F.pad(rgb, shape_pad)
        depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)

        shape_pad[3] = (img_shape[0] * 2 - real_shape[0])  # // 2
        shape_pad[1] = (img_shape[1] * 2 - real_shape[1])  # // 2 + 1
        flow_img = F.pad(flow_img.permute(2, 0, 1), shape_pad).permute(1, 2, 0).contiguous()
        flow_mask = F.pad(flow_mask, shape_pad)

    else:
        shape_pad[3] = (img_shape[0] - real_shape[0])  # // 2
        shape_pad[1] = (img_shape[1] - real_shape[1])  # // 2 + 1

        rgb = F.pad(rgb, shape_pad)
        depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)
        flow_img = F.pad(flow_img.permute(2, 0, 1), shape_pad).permute(1, 2, 0).contiguous()
        flow_mask = F.pad(flow_mask, shape_pad)

    # Convert depth into fourier frequencies, similar to the positional encoding used in NERF
    if _config['fourier_levels'] >= 0:
        depth_img_no_occlusion = depth_img_no_occlusion.squeeze()
        mask = (depth_img_no_occlusion > 0).clone()
        fourier_feats = []
        for L in range(_config['fourier_levels']):
            fourier_feat = depth_img_no_occlusion * np.pi * 2 ** L
            fourier_feats.append(fourier_feat.sin())
            fourier_feats.append(fourier_feat.cos())
        depth_img_no_occlusion = torch.stack(fourier_feats + [depth_img_no_occlusion])
        depth_img_no_occlusion = depth_img_no_occlusion * mask.unsqueeze(0)

    return rgb, depth_img_no_occlusion, flow_img, flow_mask


def _to_numpy_image(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if image.dim() == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        return image.numpy()
    return np.asarray(image)


def _save_viz_image(image, save_path, title=None, figsize=(12, 8), dpi=150):
    np_image = _to_numpy_image(image)
    np_image = np.clip(np_image, 0.0, 1.0)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(np_image)
    ax.axis('off')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    del fig, ax, np_image
    gc.collect()


def _create_height_overlay_map(uv, depth_values, cam_params, image_shape, clip_percentiles=(2, 98)):
    """
    Build a 1x1xH xW tensor containing normalized heights so overlay_imgs can colorize by height.
    """
    if uv is None or depth_values is None:
        return None
    if uv.numel() == 0 or depth_values.numel() == 0:
        return None

    height, width = image_shape[0], image_shape[1]
    uv_int = uv.long()
    uv_float = uv.float()

    fy = cam_params[1]
    cy = cam_params[3]

    heights = (uv_float[:, 1] - cy) * depth_values / fy

    height_img = torch.zeros((height, width), device=uv.device, dtype=torch.float32)
    valid_mask = torch.zeros_like(height_img, dtype=torch.bool)

    v_coords = uv_int[:, 1]
    u_coords = uv_int[:, 0]
    height_img[v_coords, u_coords] = heights
    valid_mask[v_coords, u_coords] = True

    heights_cpu = heights.detach().cpu().numpy()
    if heights_cpu.size == 0:
        return height_img.unsqueeze(0).unsqueeze(0)

    low_percentile, high_percentile = clip_percentiles
    lower = np.percentile(heights_cpu, low_percentile)
    upper = np.percentile(heights_cpu, high_percentile)
    if not np.isfinite(lower):
        lower = heights_cpu.min()
    if not np.isfinite(upper):
        upper = heights_cpu.max()
    if upper - lower < 1e-4:
        upper = lower + 1e-4

    normalized_map = torch.zeros_like(height_img, dtype=torch.float32)
    normalized_values = (height_img[valid_mask] - lower) / (upper - lower)
    normalized_values = torch.clamp(normalized_values, 0., 1.)
    # Prevent valid pixels from being rounded to zero when converted to uint8.
    eps = 1.0 / 128.0
    normalized_values = normalized_values * (1.0 - eps) + eps
    normalized_map[valid_mask] = normalized_values

    return normalized_map.unsqueeze(0).unsqueeze(0)


def _visualize_temporal_aggregation(dataset, aggregation_rt, _config, seed, mean_torch, std_torch):
    if aggregation_rt is None or dataset is None:
        print("[viz_aggregation] Skipping visualization: no aggregated pose available.")
        return

    save_dir = os.path.join(output_dir, 'aggregation')
    os.makedirs(save_dir, exist_ok=True)

    def init_fn(worker_id):
        return _init_fn(worker_id, seed)

    agg_loader = torch.utils.data.DataLoader(dataset=dataset,
                                             shuffle=False,
                                             batch_size=1,
                                             num_workers=_config['num_worker'],
                                             worker_init_fn=init_fn,
                                             collate_fn=merge_inputs,
                                             drop_last=False,
                                             pin_memory=False)

    aggregation_rt = aggregation_rt.to(device).float()
    frame_counter = 0
    color_by_height = _config.get('color_height', False)

    with torch.no_grad():
        fixed_tr_error = None
        fixed_rot_error = None
        for sample in tqdm(agg_loader, desc='Temporal aggregation visualization', leave=False):
            if _config['fix_rt']:
                if fixed_tr_error is None:
                    fixed_tr_error = sample['tr_error'].clone()
                    fixed_rot_error = sample['rot_error'].clone()
                else:
                    sample['tr_error'] = fixed_tr_error.clone()
                    sample['rot_error'] = fixed_rot_error.clone()
            sample['tr_error'] = sample['tr_error'].cuda()
            sample['rot_error'] = sample['rot_error'].cuda()

            for idx in range(len(sample['rgb'])):
                # check if rot_error[idx] is in the correct format
                rot_err = sample['rot_error'][idx]
                # quaternion_distance expects a 1D quaternion tensor
                if rot_err.dim() > 1:
                    rot_err = rot_err.squeeze()
                    
                real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]

                point_cloud = sample['point_cloud'][idx].cuda()
                if _config['max_depth'] < 100.:
                    point_cloud = point_cloud[:, point_cloud[0, :] < _config['max_depth']]

                reflectance = None
                if _config['use_reflectance']:
                    reflectance = sample['reflectance'][idx].cuda()

                cam_params = sample['calib'][idx].cuda()
                cam_model = CameraModel()
                cam_model.focal_length = cam_params[:2]
                cam_model.principal_point = cam_params[2:]

                R = quat2mat(sample['rot_error'][idx])
                T = tvector2mat(sample['tr_error'][idx])
                initial_pose = torch.mm(T, R).inverse()
                pc_initial = rotate_forward(point_cloud.clone(), initial_pose)
                init_depth_img, init_uv, init_indexes, init_depth_vals = prepare_input(
                    cam_params, pc_initial, real_shape, reflectance, _config)
                init_depth_valid = init_depth_vals[init_indexes]

                cam2vel = sample['cam2vel'][idx].cuda().float()
                pc_vel = rotate_forward(point_cloud.clone(), cam2vel)
                pc_agg = rotate_back(pc_vel.clone(), aggregation_rt)
                agg_depth_img, agg_uv, agg_indexes, agg_depth_vals = prepare_input(
                    cam_params, pc_agg, real_shape, reflectance, _config)
                agg_depth_valid = agg_depth_vals[agg_indexes]

                rgb = sample['rgb'][idx].cuda()
                rgb = rgb / 255.
                if _config['normalize_images']:
                    rgb = (rgb - mean_torch) / std_torch
                rgb = rgb.permute(2, 0, 1)

                if color_by_height:
                    init_height_overlay = _create_height_overlay_map(init_uv.clone(), init_depth_valid,
                                                                     cam_params, real_shape)
                    agg_height_overlay = _create_height_overlay_map(agg_uv.clone(), agg_depth_valid,
                                                                    cam_params, real_shape)
                else:
                    init_height_overlay = None
                    agg_height_overlay = None

                init_overlay = init_height_overlay if init_height_overlay is not None \
                    else init_depth_img[-1].unsqueeze(0).unsqueeze(0)
                init_max_depth = 1.0 if init_height_overlay is not None else 0.5

                agg_overlay = agg_height_overlay if agg_height_overlay is not None \
                    else agg_depth_img[-1].unsqueeze(0).unsqueeze(0)
                agg_max_depth = 1.0 if agg_height_overlay is not None else (_config['max_depth'] / 2)

                viz_initial = overlay_imgs(rgb, init_overlay, max_depth=init_max_depth, close_thr=1000)
                viz_aggregated = overlay_imgs(rgb, agg_overlay, max_depth=agg_max_depth, close_thr=1000)

                viz_initial_np = _to_numpy_image(viz_initial)
                viz_aggregated_np = _to_numpy_image(viz_aggregated)

                fig, axes = plt.subplots(1, 2, figsize=(12, 8))
                axes[0].imshow(viz_initial_np)
                axes[0].set_title('Initial Calibration')
                axes[0].axis('off')
                axes[1].imshow(viz_aggregated_np)
                axes[1].set_title('Temporal Aggregation (Mean)')
                axes[1].axis('off')
                fig.tight_layout()

                base_name = f'frame_{frame_counter:06d}'
                if 'rgb_name' in sample:
                    rgb_entry = sample['rgb_name']
                    if isinstance(rgb_entry, (list, tuple)):
                        rgb_candidate = rgb_entry[idx]
                    else:
                        rgb_candidate = rgb_entry
                    if isinstance(rgb_candidate, str):
                        base_name = os.path.splitext(os.path.basename(rgb_candidate))[0]

                fig.savefig(os.path.join(save_dir, f'aggregation_result_{base_name}.png'), dpi=300)
                plt.close(fig)
                frame_counter += 1

                del (pc_initial, pc_vel, pc_agg, init_depth_img, agg_depth_img, viz_initial, viz_aggregated,
                     viz_initial_np, viz_aggregated_np)
            gc.collect()
# noinspection PyUnreachableCode
def evaluate_calibration(_config, seed):
    global EPOCH, output_dir

    np.random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    cv2.setRNGSeed(seed)
    if _config['deterministic']:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)

    checkpoint = torch.load(_config['weights'][0], map_location='cpu')

    if _config['viz']:
        print("output_dir:", output_dir)
        os.makedirs(os.path.join(output_dir, 'correspondence'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'output'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'init'), exist_ok=True)
        # plt.show(block=False)
        # plt.pause(1)

    _config['network'] = 'RAFT'
    _config['use_reflectance'] = checkpoint['config']['use_reflectance']
    _config['initial_pool'] = True
    _config['upsample_method'] = checkpoint['config']['upsample_method']
    _config['occlusion_kernel'] = 9
    _config['occlusion_threshold'] = 3.9999
    _config['scaled_gt'] = False
    _config['uncertainty'] = False
    _config['der_type'] = "NLL"
    _config['unc_freeze'] = False
    _config['normalize_images'] = checkpoint['config']['normalize_images']
    _config['max_depth'] = checkpoint['config']['max_depth']
    if 'uncertainty' in checkpoint['config']:
        _config['uncertainty'] = checkpoint['config']['uncertainty']
    _config['fourier_levels'] = -1
    if 'fourier_levels' in checkpoint['config']:
        _config['fourier_levels'] = checkpoint['config']['fourier_levels']
    _config['num_scales'] = 1
    if 'num_scales' in checkpoint['config']:
        _config['num_scales'] = checkpoint['config']['num_scales']
    if 'der_type' in checkpoint['config']:
        _config['der_type'] = checkpoint['config']['der_type']
    if 'unc_freeze' in checkpoint['config']:
        _config['unc_freeze'] = checkpoint['config']['unc_freeze']
    _config['context_encoder'] = 'rgb'
    if 'context_encoder' in checkpoint['config']:
        _config['context_encoder'] = checkpoint['config']['context_encoder']

    mean_torch = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std_torch = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Setup Dataset
    val_directories = []
    base_dir = _config['data_folder']
    if _config['dataset'] == 'argoverse':
        img_shape = (640, 1920 // 2)  # Multiple of 64, inference at scale 0.5
        if _config['cam'] is None:
            _config['cam'] = 'ring_front_center'
        else:
            assert _config['cam'] in ['ring_front_center'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif 'kitti' in _config['dataset']:
        img_shape = (384, 1280)  # Multiple of 64
        for subdir in ['00']:
            val_directories.append(os.path.join(base_dir, subdir))
        if _config['cam'] is None:
            _config['cam'] = '2'
        else:
            assert str(_config['cam']) in ['2', '3'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif _config['dataset'] == 'pandaset':
        img_shape = (576, 1920 // 2)  # Multiple of 64 inference at scale 0.5
        if _config['cam'] is None:
            _config['cam'] = 'front_camera'
        else:
            assert _config['cam'] in ['front_camera'], \
                f"Camera {_config['cam']} not supported for the {_config['dataset']} dataset"
    elif _config['dataset'] == 'custom':
        val_directories.append(base_dir)
        if _config['downsize']:
            first_camera_path = os.listdir(os.path.join(_config['data_folder'], 'Downsample/camera'))[0]
            first_camera_frame = np.asarray(Image.open(os.path.join(_config['data_folder'], 'Downsample/camera', first_camera_path)))
        else:
            first_camera_path = os.listdir(os.path.join(_config['data_folder'], 'camera'))[0]
            first_camera_frame = np.asarray(Image.open(os.path.join(_config['data_folder'], 'camera', first_camera_path)))
        img_shape = [first_camera_frame.shape[0], first_camera_frame.shape[1]]
        if _config['downsample']:
            img_shape = [img_shape[0] // 2, img_shape[1] // 2]
        if img_shape[0] % 64 > 0:
            img_shape[0] = 64 * ((img_shape[0] // 64) + 1)
        if img_shape[1] % 64 > 0:
            img_shape[1] = 64 * ((img_shape[1] // 64) + 1)
    else:
        raise RuntimeError("Dataset unknown")

    if _config['dataset'] == 'kitti':
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'])
    elif _config['dataset'] == 'argoverse':
        for subdir in ['train4']:
            val_directories.append(os.path.join(_config['data_folder'], subdir))
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'])
    elif _config['dataset'] == 'pandaset':
        for subdir in ['011', '122', '124', '030', '109', '043', '084', '115', '090']:
            val_directories.append(os.path.join(_config['data_folder'], subdir))
        dataset_val = DatasetPandasetExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                    max_t=_config['max_t'],
                                                    use_reflectance=_config['use_reflectance'],
                                                    normalize_images=_config['normalize_images'],
                                                    sensor_id=0, camera=_config['cam'])
    elif _config['dataset'] == 'custom':
        dataset_val = DatasetGeneralExtrinsicCalib(val_directories, train=False, max_r=_config['max_r'],
                                                   max_t=_config['max_t'], use_reflectance=_config['use_reflectance'],
                                                   normalize_images=_config['normalize_images'],
                                                   dataset=_config['dataset'], cam=_config['cam'], sensor_type=_config['sensor_type'], downsample=_config['downsize'])

    def init_fn(x):
        return _init_fn(x, seed)

    # Training and test set creation
    num_worker = _config['num_worker']
    batch_size = 1  # This code is designed for a batch size of 1, don't change this!

    TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                shuffle=False,
                                                batch_size=batch_size,
                                                num_workers=num_worker,
                                                worker_init_fn=init_fn,
                                                collate_fn=merge_inputs,
                                                drop_last=False,
                                                pin_memory=False)

    print(len(TestImgLoader))

    models = []
    print(f"Loading weights from {_config['weights']}")
    for i in range(len(_config['weights'])):
        checkpoint = torch.load(_config['weights'][i], map_location='cpu')
        model = get_model(_config, img_shape)
        saved_state_dict = checkpoint['state_dict']
        clean_state_dict = saved_state_dict
        model.load_state_dict(clean_state_dict, strict=False)
        model = model.to(device)
        model.eval()
        models.append(model)

    errors_r = []
    errors_t = []
    list_quats = []
    list_transl = []
    epe = []
    ransac_time = []
    inference_time = []
    final_calib_RTs = []
    for i in range(len(_config['weights']) + 1):
        errors_r.append([])
        errors_t.append([])
        list_quats.append([])
        list_transl.append([])
        epe.append([])
        final_calib_RTs.append([])
    tbar = tqdm(TestImgLoader)
    idex = 0
    # first frame initial pose error
    fixed_tr_error = None
    fixed_rot_error = None
    for batch_idx, sample in enumerate(tbar):
        idex += 1
        lidar_input = []
        rgb_input = []
        if _config['fix_rt']:
            if batch_idx == 0:
                fixed_tr_error = sample['tr_error'].clone()
                fixed_rot_error = sample['rot_error'].clone()
                print(f"fixed_tr_error: {fixed_tr_error}")
                print(f"fixed_rot_error: {fixed_rot_error}")
            else:
                sample['tr_error'] = fixed_tr_error.clone()
                sample['rot_error'] = fixed_rot_error.clone()

        sample['tr_error'] = sample['tr_error'].cuda()
        sample['rot_error'] = sample['rot_error'].cuda()

        for idx in range(len(sample['rgb'])):
            # check if rot_error[idx] is in the correct format
            rot_err = sample['rot_error'][idx]
            # quaternion_distance expects a 1D quaternion tensor
            if rot_err.dim() > 1:
                rot_err = rot_err.squeeze()
            errors_r[0].append(quaternion_distance(sample['rot_error'][idx],
                                                   torch.tensor([1., 0., 0., 0.], device=sample['rot_error'].device)))

            errors_t[0].append(sample['tr_error'][idx].norm().item())
            list_quats[0].append(sample['rot_error'][idx].cpu().numpy())
            list_transl[0].append(sample['tr_error'][idx].cpu().numpy())

            # ProjectPointCloud in RT-pose

            real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]

            sample['point_cloud'][idx] = sample['point_cloud'][idx].cuda()
            if _config['max_depth'] < 100.:
                sample['point_cloud'][idx] = sample['point_cloud'][idx][:,
                                             sample['point_cloud'][idx][0, :] < _config['max_depth']]

            pc_rotated = sample['point_cloud'][idx].clone()
            reflectance = None
            if _config['use_reflectance']:
                reflectance = sample['reflectance'][idx].cuda()

            R = quat2mat(sample['rot_error'][idx])
            T = tvector2mat(sample['tr_error'][idx])
            RT1_inv = torch.mm(T, R)
            extrinsic_error = [RT1_inv.clone().inverse()]
            extrinsic_prediction = [torch.eye(4)]

            pc_rotated = rotate_forward(pc_rotated, extrinsic_error[0])

            # Project point cloud into virtual image plane placed at random 'initial_calib'
            cam_params = sample['calib'][idx].cuda()
            depth_img_no_occlusion, uv, indexes, depth = prepare_input(cam_params, pc_rotated, real_shape,
                                                                       reflectance, _config)
            depth_valid = depth[indexes]
            if _config['viz'] and _config.get('color_height', False):
                initial_height_overlay = _create_height_overlay_map(
                    uv.clone(), depth_valid, cam_params, real_shape)
            else:
                initial_height_overlay = None
            cam_model = CameraModel()
            cam_model.focal_length = cam_params[:2]
            cam_model.principal_point = cam_params[2:]

            flow, points_3D, new_indexes = get_flow_zforward(uv.float(), depth_valid, RT1_inv, cam_model,
                                                             [real_shape[0], real_shape[1], 3],
                                                             scale_flow=False, reverse=False,
                                                             get_valid_indexes=True)

            uv = uv[new_indexes].clone()
            flow = flow[new_indexes].clone()

            rgb = sample['rgb'][idx].cuda()

            # Normalize image
            rgb = rgb / 255.
            if _config['normalize_images']:
                rgb = (rgb - mean_torch) / std_torch
            rgb = rgb.permute(2, 0, 1)
            sample['rgb'][idx] = rgb

            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device='cuda', dtype=torch.float)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            # flow_mask containts 1 in pixels that have a point projected
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1

            if _config['viz']:
                if _config.get('color_height', False) and initial_height_overlay is not None:
                    lidar_for_overlay = initial_height_overlay
                    overlay_max_depth = 1.0
                else:
                    lidar_for_overlay = depth_img_no_occlusion[-1].unsqueeze(0).unsqueeze(0)
                    overlay_max_depth = 0.5
                viz_initial = overlay_imgs(
                    rgb,
                    lidar_for_overlay,
                    max_depth=overlay_max_depth,
                    close_thr=1000
                )
                viz_initial_np = _to_numpy_image(viz_initial)
                _save_viz_image(viz_initial_np,
                                os.path.join(output_dir, 'init', f'init_{idex}_.png'),
                                title='Initial Calibration')
                del viz_initial
            else:
                viz_initial_np = None

            points_3D = points_3D[new_indexes].clone()
            rgb, depth_img_no_occlusion, flow_img, flow_mask = downsample_and_pad(_config, rgb, depth_img_no_occlusion,
                                                                                  img_shape, real_shape, flow_img,
                                                                                  flow_mask)

            rgb_input.append(rgb)
            lidar_input.append(depth_img_no_occlusion)

        lidar_input = torch.stack(lidar_input)
        rgb_input = torch.stack(rgb_input)

        for iteration in range(len(_config['weights'])):
            torch.cuda.synchronize()
            time1 = time.time()
            # Predict 'flow': dense lidar depth map to rgb pixel displacements
            with torch.no_grad():
                predicted_flow = models[iteration](rgb_input, lidar_input)
                torch.cuda.synchronize()
                time2 = time.time()
                predicted_flow, predicted_uncertainty = predicted_flow
                inference_time.append(time2 - time1)
                # Upsample if necessary
                if _config['dataset'] in ['argoverse', 'pandaset'] or _config['downsample']:
                    predicted_flow = list(predicted_flow)
                    predicted_flow[-1] *= 2
                    predicted_flow[-1] = F.interpolate(predicted_flow[-1], scale_factor=2, mode='bilinear')
                    if _config['uncertainty']:
                        predicted_uncertainty[-1] = F.interpolate(predicted_uncertainty[-1], scale_factor=2,
                                                                  mode='bilinear')

            up_flow = predicted_flow[-1]

            # EPE
            gt = flow_img.clone().permute(2, 0, 1)
            gt = torch.cat((gt, flow_mask.unsqueeze(0).float()))
            gt = gt.unsqueeze(0)
            epe[iteration].append(EndPointError(up_flow, gt).item())

            up_flow = up_flow[0].permute(1, 2, 0)

            new_uv = uv.float() + up_flow[uv[:, 1], uv[:, 0]]

            valid_indexes = flow_mask[uv[:, 1], uv[:, 0]] == 1

            if flow_mask.sum() < 10:
                break

            if _config['uncertainty']:
                sum_uncertainty = predicted_uncertainty[-1][0, 0] + predicted_uncertainty[-1][0, 1]
                mean_uncertainty = sum_uncertainty * flow_mask
                try:
                    mean_uncertainty = np.quantile(mean_uncertainty[flow_mask != 0].detach().cpu().numpy(),
                                                   _config['quantile'])
                except:
                    pass
                valid_indexes = valid_indexes & (sum_uncertainty[uv[:, 1], uv[:, 0]] < mean_uncertainty)

            # Check only pixels that are within the image border
            valid_indexes = valid_indexes & (new_uv[:, 0] < flow_mask.shape[1])
            valid_indexes = valid_indexes & (new_uv[:, 1] < flow_mask.shape[0])
            valid_indexes = valid_indexes & (new_uv[:, 0] >= 0)
            valid_indexes = valid_indexes & (new_uv[:, 1] >= 0)
            new_uv = new_uv[valid_indexes]

            valid_indexes2 = torch.ones(new_uv.shape[0], dtype=torch.bool).cuda()

            new_uv = new_uv[valid_indexes2]

            points_2d = new_uv.cpu().numpy()
            obj_coord = points_3D[valid_indexes][valid_indexes2][:, :3].cpu().numpy()
            obj_coord_zforward = np.zeros(obj_coord.shape)
            obj_coord_zforward[:, 0] = obj_coord[:, 0]
            obj_coord_zforward[:, 1] = obj_coord[:, 1]
            obj_coord_zforward[:, 2] = obj_coord[:, 2]
            cam_mat = cam_model.get_matrix()

            torch.cuda.synchronize()

            time1 = time.time()
            if obj_coord_zforward.shape[0] < 10:
                for left_iter in range(iteration + 1):
                    errors_t[left_iter].pop(-1)
                    errors_r[left_iter].pop(-1)
                    list_transl[left_iter].pop(-1)
                    list_quats[left_iter].pop(-1)
                break

            if _config['quantile'] < 1.0 and not _config['uncertainty']:
                num_corr = points_2d.shape[0]
                corr_to_keep = list(range(num_corr))
                random.shuffle(corr_to_keep)
                num_corr_to_keep = int(num_corr * _config['quantile'])
                if num_corr_to_keep > 10:
                    points_2d = points_2d[corr_to_keep[:num_corr_to_keep]]
                    obj_coord_zforward = obj_coord_zforward[corr_to_keep[:num_corr_to_keep]]

            if _config['viz']:
                std = [0.229, 0.224, 0.225]
                mean = [0.485, 0.456, 0.406]

                vis_img = sample['rgb'][idx].clone().cpu().permute(1, 2, 0).numpy()
                vis_img = vis_img * std + mean

                h, w, _ = vis_img.shape
                # float 타입으로 생성해야 0.0 ~ 1.0 사이의 알파값을 다룰 수 있습니다.
                overlay_img = np.zeros((h, w, 4), dtype=np.float32)

                # --- 3. 오버레이 이미지에 대응점 그리기 ---
                # 깊이(z값)와 컬러맵을 준비합니다.
                depths = obj_coord_zforward[:, 2]
                vmax = np.percentile(depths, 95)  # 극단적인 값에 의한 왜곡 방지
                norm_depths = np.clip(depths / vmax, 0, 1)
                cmap = cm.get_cmap('jet')  # overlay_imgs에서 사용된 'jet' 컬러맵

                b_uv = uv.float()
                b_uv = b_uv[valid_indexes]
                b_uv = b_uv[valid_indexes2]
                points_2d_before = b_uv.cpu().numpy()
                # 각 대응점에 깊이 색상과 투명도를 적용하여 원을 그립니다.
                dw_ratio = 3
                for i in range(0,len(points_2d), dw_ratio):
                    pt = tuple(map(int, points_2d[i]))
                    b_pt = tuple(map(int, points_2d_before[i]))

                    # 깊이에 해당하는 색상을 cmap에서 가져옵니다 (R, G, B, A 순서).
                    color_rgba = cmap(norm_depths[i])

                    # cv2.circle은 (B, G, R, A) 순서를 사용하므로 채널 순서를 변경합니다.
                    # 또한, 투명도를 0.8로 고정하여 더 잘 보이게 합니다.
                    color_bgra = (color_rgba[2], color_rgba[1], color_rgba[0], 0.8)

                    # cv2.line(overlay_img, pt, b_pt, color=color_bgra, thickness=1)
                    cv2.arrowedLine(overlay_img, b_pt, pt, color=color_bgra, thickness=1, tipLength=0.05)
                    # cv2.circle(overlay_img, b_pt, radius=1, color=color_bgra, thickness=-1)

                # --- 4. 배경과 오버레이 합성 (Alpha Blending) ---
                # overlay_imgs의 합성 공식과 동일한 방식입니다.
                alpha = overlay_img[:, :, 3:]  # 알파 채널 (H, W, 1)
                foreground = overlay_img[:, :, :3]  # RGB 채널 (H, W, 3)

                # blended_img = foreground * alpha + background * (1 - alpha)
                blended_img = foreground * alpha + vis_img * (1. - alpha)
                blended_img = np.clip(blended_img, 0, 1)

                _save_viz_image(blended_img,
                                os.path.join(output_dir, 'correspondence',
                                             f'comparison_result_{idex}_{iteration}.png'),
                                title='Correspondences')
                del overlay_img, alpha, foreground, blended_img, vis_img, points_2d_before
                # plt.draw()
                # plt.pause(5)

            # Predict relative transformation based on CMRNext correspondences
            # for iterative refinement
            cuda_pnp = cv2.pythoncuda.cudaPnP(obj_coord_zforward.astype(np.float32).copy(),
                                              points_2d.astype(np.float32).copy(), obj_coord_zforward.shape[0],
                                              200, 2., cam_mat.astype(np.float32)
                                              )

            transl = cuda_pnp[0, [0, 1, 2]]
            rot_mat = cuda_pnp[:, 3:6].T
            rot_mat, _ = cv2.Rodrigues(rot_mat)

            torch.cuda.synchronize()
            time2 = time.time()
            # print("Ransac: ", time2-time1)
            ransac_time.append(time2 - time1)

            transl = torch.tensor(transl).float().squeeze().cuda()

            rot_mat = torch.tensor(rot_mat)

            pred_quaternion = quaternion_from_matrix(rot_mat)

            R_predicted = quat2mat(pred_quaternion).cuda()
            T_predicted = tvector2mat(transl)
            RT_predicted = torch.mm(T_predicted, R_predicted)
            composed = torch.mm(extrinsic_error[iteration], RT_predicted.inverse())
            extrinsic_error.append(composed)
            extrinsic_prediction.append(torch.mm(extrinsic_prediction[iteration], RT_predicted.inverse().cpu()))

            T_composed = composed[:3, 3]
            R_composed = quaternion_from_matrix(composed)
            errors_r[iteration + 1].append(quaternion_distance(R_composed,
                                                               torch.tensor([1., 0., 0., 0.],
                                                                            device=R_composed.device)))
            errors_t[iteration + 1].append(T_composed.norm().item())

            list_transl[iteration + 1].append(T_composed.cpu().numpy())
            list_quats[iteration + 1].append(R_composed.cpu().numpy())

            # Compute final cam lidar predicted matrix
            points_3D_orig = torch.mm(extrinsic_prediction[-2].to(points_3D.device), points_3D[valid_indexes].T).T
            points_3D_orig = torch.mm(extrinsic_error[0], points_3D_orig.T).T
            # points_3D_orig = rotate_back(points_3D_orig, sample['cam2vel'][0].to(points_3D_orig.device))
            points_3D_orig = rotate_forward(points_3D_orig, sample['cam2vel'][0].to(points_3D_orig.device))
            final_correspondences = new_uv, points_3D_orig
            cuda_pnp_final = cv2.pythoncuda.cudaPnP(
                final_correspondences[1][:, :3].cpu().numpy().astype(np.float32).copy(),
                final_correspondences[0].cpu().numpy().astype(np.float32).copy(),
                points_3D_orig.shape[0], 200, 2., cam_mat.astype(np.float32))
            transl_final = cuda_pnp_final[0, :3]
            rot_mat_final = cuda_pnp_final[:, 3:6].T
            rot_mat_final, _ = cv2.Rodrigues(rot_mat_final)
            rot_mat_final = torch.tensor(rot_mat_final)
            T_predicted_final = tvector2mat(torch.tensor(transl_final))
            R_predicted_final = torch.eye(4)
            R_predicted_final[:3, :3] = rot_mat_final.clone().detach()
            # Final prediction by CMRNet (for each iteration)
            RT_predicted_final = torch.mm(T_predicted_final, R_predicted_final)
            # final_calib_RTs[iteration + 1].append(RT_predicted_final.inverse())
            final_calib_RTs[iteration + 1].append(RT_predicted_final)

            if T_composed.norm().item() > 4.:
                # Prediction has failed for this frame
                for left_iteration in range(iteration + 2, len(_config['weights']) + 1):
                    errors_t[left_iteration].append(T_composed.norm().item())
                    errors_r[left_iteration].append(errors_r[iteration + 1][-1])
                break

            # Rotate point cloud based on predicted pose, and generate new
            # inputs for the next iteration
            rotated_point_cloud = rotate_forward(sample['point_cloud'][idx], extrinsic_error[-1])

            depth_img_no_occlusion, uv, indexes, depth = prepare_input(cam_params, rotated_point_cloud, real_shape,
                                                                       reflectance, _config)
            depth_valid_next = depth[indexes]
            final_height_overlay = None
            if _config['viz'] and _config.get('color_height', False) and iteration == len(_config['weights']) - 1:
                final_height_overlay = _create_height_overlay_map(
                    uv.clone(), depth_valid_next, cam_params, real_shape)

            flow, points_3D, new_indexes = get_flow_zforward(uv.float(), depth_valid_next, extrinsic_error[-1].inverse(),
                                                             cam_model, [real_shape[0], real_shape[1], 3],
                                                             scale_flow=False, reverse=False,
                                                             get_valid_indexes=True)

            uv = uv[new_indexes].clone()
            flow = flow[new_indexes].clone()

            rgb = sample['rgb'][idx].cuda()
            flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device='cuda', dtype=torch.float)
            flow_img[uv[:, 1], uv[:, 0]] = flow
            flow_mask = torch.zeros((real_shape[0], real_shape[1]), device='cuda', dtype=torch.int)
            flow_mask[uv[:, 1], uv[:, 0]] = 1

            points_3D = points_3D[new_indexes].clone()
            rgb, depth_img_no_occlusion, flow_img, flow_mask = downsample_and_pad(_config, rgb, depth_img_no_occlusion,
                                                                                  img_shape, real_shape,
                                                                                  flow_img, flow_mask)

            rgb_input = rgb.unsqueeze(0)
            lidar_input = depth_img_no_occlusion.unsqueeze(0)

            if _config['viz'] and iteration == len(_config['weights']) - 1:
                gt_uv, gt_depth, _, _ = cam_model.project_pytorch(rotated_point_cloud, real_shape, reflectance)
                gt_uv = gt_uv.t().int().contiguous()

                new_depth_img = torch.zeros(real_shape[:2], device='cuda', dtype=torch.float)
                new_depth_img += 1000.
                new_depth_img = visibility.depth_image(gt_uv.int().contiguous(), gt_depth, new_depth_img,
                                                       gt_uv.shape[0],
                                                       real_shape[1], real_shape[0])
                new_depth_img[new_depth_img == 1000.] = 0.

                new_depth_img_no_occlusion = torch.zeros_like(new_depth_img, device='cuda')
                new_depth_img_no_occlusion = visibility.visibility2(new_depth_img, cam_params,
                                                                    new_depth_img_no_occlusion,
                                                                    new_depth_img.shape[1], new_depth_img.shape[0],
                                                                    _config['occlusion_threshold'],
                                                                    _config['occlusion_kernel'])

                if _config.get('color_height', False) and final_height_overlay is not None:
                    lidar_flow = final_height_overlay
                    overlay_max_depth = 1.0
                else:
                    lidar_flow = new_depth_img_no_occlusion.unsqueeze(0).unsqueeze(0)
                    overlay_max_depth = _config['max_depth'] / 2
                viz_final = overlay_imgs(
                    sample['rgb'][idx].cuda(),
                    lidar_flow,
                    max_depth=overlay_max_depth,
                    close_thr=1000
                )
                viz_final_np = _to_numpy_image(viz_final)
                comparison_fig, comparison_axes = plt.subplots(1, 2, figsize=(12, 8))
                comparison_axes[0].imshow(viz_initial_np if viz_initial_np is not None else viz_final_np)
                comparison_axes[0].set_title('Initial Calibration')
                comparison_axes[0].axis('on')
                comparison_axes[1].imshow(viz_final_np)
                comparison_axes[1].set_title('CMRNext Estimated Calibration')
                comparison_axes[1].axis('on')
                comparison_fig.tight_layout()
                comparison_fig.savefig(
                    os.path.join(output_dir, 'output', f'comparison_result_{idex}_{iteration}.png'),
                    dpi=300
                )
                plt.close(comparison_fig)
                del comparison_fig, comparison_axes, viz_final, viz_final_np, new_depth_img, new_depth_img_no_occlusion, lidar_flow
                gc.collect()
                viz_initial_np = None

            try:
                if _config['dataset'] != 'custom':
                    tbar.set_postfix(t_mean=torch.tensor(errors_t[-1]).mean().item(),
                                     t_median=torch.tensor(errors_t[-1]).median().item(),
                                     r_mean=torch.tensor(errors_r[-1]).mean().item(),
                                     r_median=torch.tensor(errors_r[-1]).median().item(),
                                     epe_mean=torch.tensor(epe[0]).mean().item())
            except:
                pass

    print(f'Network Time: ', torch.tensor(inference_time).mean())
    print(f'PnP+RANSAC Time: ', torch.tensor(ransac_time).mean())
    for iteration in range(len(_config['weights']) + 1):
        errors_t[iteration] = torch.tensor(errors_t[iteration])
        errors_r[iteration] = torch.tensor(errors_r[iteration])

    console = Console()
    table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_HEAVY_HEAD, title_style="bold red")
    table.title = f"CMRNext Results on {_config['dataset']}, camera {_config['cam']}"
    table.add_column("Iteration")
    table.add_column("Median Translation error (cm)", justify="center", max_width=20)
    table.add_column("Median Rotation error (˚)", justify="center", max_width=20)
    table.add_row(
        f"Initial Pose",
        f"{errors_t[0].median().item() * 100:.2f}",
        f"{errors_r[0].median().item():.2f}"
    )
    for iteration in range(1, len(_config['weights']) + 1):
        table.add_row(
            f"Iteration {iteration}",
            f"{errors_t[iteration].median().item() * 100:.2f}",
            f"{errors_r[iteration].median().item():.2f}"
        )
    print("")
    print("")
    console.print(table)

    table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL_HEAVY_HEAD, title_style="bold red")
    table.title = f"Temporal Aggregation Results on {_config['dataset']}"
    table.add_column("Aggregation Measure", max_width=13)
    table.add_column("Translation Error (cm)", justify="center")
    table.add_column("Rotation Error (˚)", justify="center")

    iteration = len(_config['weights'])
    aggregation_pose = None
    final_stack = torch.stack(final_calib_RTs[iteration]) if len(final_calib_RTs[iteration]) > 0 else None
    if final_stack is not None:
        final_quats = np.stack([quaternion_from_matrix(t) for t in final_calib_RTs[iteration]])
        avg_quat_np = average_quaternions(final_quats)
        avg_quat_tensor = torch.from_numpy(avg_quat_np)
        aggregation_pose = quat2mat(avg_quat_tensor.float())
        aggregation_pose[:3, 3] = final_stack[:, :3, 3].mean(0)

        r_error_avg = quaternion_distance(
            avg_quat_tensor,
            quaternion_from_matrix(sample['cam2vel'][0])
        )
        r_error_mode = quaternion_distance(
            quaternion_mode(final_quats, 4),
            quaternion_from_matrix(sample['cam2vel'][0])
        )
        if r_error_mode > quaternion_distance(
                quaternion_mode(final_quats, 3),
                quaternion_from_matrix(sample['cam2vel'][0])
        ):
            r_error_mode = quaternion_distance(
                quaternion_mode(final_quats, 3),
                quaternion_from_matrix(sample['cam2vel'][0])
            )

        t_error_avg = (final_stack[:, :3, 3].mean(0)
                       - sample['cam2vel'][0][:3, 3]).norm() * 100.
        t_error_median = (final_stack[:, :3, 3].median(0)[0]
                          - sample['cam2vel'][0][:3, 3]).norm() * 100.
        t_error_mode = (quaternion_mode(final_stack[:, :3, 3], 2)
                        - sample['cam2vel'][0][:3, 3]).norm() * 100.
        if t_error_mode > (quaternion_mode(final_stack[:, :3, 3], 1)
                           - sample['cam2vel'][0][:3, 3]).norm() * 100.:
            t_error_mode = (quaternion_mode(final_stack[:, :3, 3], 1)
                            - sample['cam2vel'][0][:3, 3]).norm() * 100.
        table.add_row(
            "Mean",
            f"[bold green]{t_error_avg.item():.2f}[/bold green]" if t_error_avg.item() <= t_error_median.item() and
                                                                    t_error_avg.item() <= t_error_mode.item() else
            f"{t_error_avg.item():.2f}",
            f"[bold green]{r_error_avg:.2f}[/bold green]" if r_error_avg <= r_error_mode else
            f"{r_error_avg:.2f}",
        )
        table.add_row(
            "Median",
            f"[bold green]{t_error_median.item():.2f}[/bold green]" if t_error_median.item() <= t_error_avg.item() and
                                                                       t_error_median.item() <= t_error_mode.item() else
            f"{t_error_median.item():.2f}",
            f"---"
        )
        table.add_row(
            "Mode",
            f"[bold green]{t_error_mode.item():.2f}[/bold green]" if t_error_mode.item() <= t_error_avg.item() and
                                                                     t_error_mode.item() <= t_error_median.item() else
            f"{t_error_mode.item():.2f}",
            f"[bold green]{r_error_mode:.2f}[/bold green]" if r_error_mode <= r_error_avg else
            f"{r_error_mode:.2f}"
        )
        print("")
        print("")
        console.print(table)
    else:
        print("[Temporal Aggregation] No final calibration estimates were collected; skipping summary table.")

    if _config.get('viz_aggregation', False):
        _visualize_temporal_aggregation(dataset_val, aggregation_pose, _config, seed, mean_torch, std_torch)

    if _config['save_file'] is not None:
        torch.save(errors_t, f'./{_config["save_file"]}_errors_t.torch')
        torch.save(errors_r, f'./{_config["save_file"]}_errors_r.torch')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitti')
    parser.add_argument('--data_folder', type=str, default='/data/KITTI/sequences/')
    parser.add_argument('--cam', type=str, nargs='?', default=None)
    parser.add_argument('--max_t', type=float, default=1.5)
    parser.add_argument('--max_r', type=float, default=20.)
    parser.add_argument('--fix_rt', type=str2bool, nargs='?', const=True, default=False) # fix initial RT error by using the first frame
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--weights', type=str, nargs='+', default=None)
    parser.add_argument('--img_shape', type=int, nargs=1, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--save_file', type=str, nargs='?', default=None)
    parser.add_argument('--quantile', type=float, default=1.0)
    parser.add_argument('--downsample', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--viz', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--viz_aggregation', type=str2bool, nargs='?', const=True, default=False,
                        help='Visualize per-frame projections using temporal aggregation results.')
    parser.add_argument('--dataset_name', type=str, default='KITTI')
    parser.add_argument('--data_id', type=str, default='00')
    parser.add_argument('--test_topics', type=str, default='default')
    parser.add_argument('--sensor_type', type=str, default='lidar')
    parser.add_argument('--downsize', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--color_height', type=str2bool, nargs='?', const=True, default=False,
                        help='Color lidar projections by height instead of depth in visualizations.')


    args = parser.parse_args()
    _config = vars(args)

    global output_dir
    output_dir = os.path.join('/ws/output/', args.dataset_name, args.data_id, args.test_topics)
    print(f"Results will be saved in {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    evaluate_calibration(_config, _config['seed'])


if __name__ == '__main__':
    main()
