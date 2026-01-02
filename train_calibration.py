import argparse
from functools import partial
from itertools import chain

import os
import random
import time

import mathutils
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import visibility
import wandb
from skimage import io
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp
from matplotlib import cm

from datasets.DatasetExtrinsicCalib import DatasetGeneralExtrinsicCalib, DatasetPandasetExtrinsicCalib
from camera_model import CameraModel
from flow_losses import RAFT_loss2
from utils import resize_dense_vector
from models.get_model import get_model
from utils import merge_inputs, rotate_back, get_flow_zforward, downsample_flow_and_mask, \
    init_logger, downsample_depth, get_ECE
from flow_vis import flow_to_color

torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled = False


EPOCH = 1


def _init_fn(worker_id, epoch=0, seed=0):
    seed = seed + worker_id + epoch * 100
    seed = seed % (2 ** 32 - 1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def uncertainty_to_color(_tensor, mask=None):
    """
    Args:
        _tensor: (2, H, W)

    Returns:
        color: (H, W, 3)
    """
    total_uncertainty = _tensor[0, :, :] + _tensor[1, :, :]
    total_uncertainty = torch.from_numpy(total_uncertainty)
    if mask is not None:
        total_uncertainty = total_uncertainty * mask[0]
    total_uncertainty -= torch.min(total_uncertainty)
    total_uncertainty /= torch.max(total_uncertainty)
    jet = cm.get_cmap('jet')
    color = jet(total_uncertainty)
    color = color[:, :, :3]
    return color


def prepare_input(_config, device, idx, img_shape, mean, sample, std, aligned=False):
    # 샘플 기본 정보와 포인트클라우드 준비
    real_shape = [sample['rgb'][idx].shape[0], sample['rgb'][idx].shape[1], sample['rgb'][idx].shape[2]]
    sample['point_cloud'][idx] = sample['point_cloud'][idx].to(device)
    pc_rotated = sample['point_cloud'][idx].clone()
    reflectance = None
    if _config['use_reflectance']:
        reflectance = sample['reflectance'][idx].to(device)

    # RT(오차 extrinsic) 구성: rot_error/tr_error → 4×4 변환행렬
    R = mathutils.Quaternion(sample['rot_error'][idx]).to_matrix()
    R.resize_4x4()
    T = mathutils.Matrix.Translation(sample['tr_error'][idx])
    try:
        RT = T @ R
    except:
        RT = T * R

    if not aligned:
        pc_rotated = rotate_back(pc_rotated, RT)

    rgb = sample['rgb'][idx].to(device)
    if _config['normalize_images']:
        rgb = rgb / 255.
        rgb = (rgb - mean) / std
    rgb = rgb.permute(2, 0, 1)

    cam_params = sample['calib'][idx].to(device)

    cam_model = CameraModel()
    cam_model.focal_length = cam_params[:2]
    cam_model.principal_point = cam_params[2:]

    # Scale(Upsample or downsample) for Hercules dataset according to focal length
    if _config['dataset'] == 'hercules' and _config['downsize'] == True:
        target_fx, target_fy = 718.5377, 718.5377
        scale_x = float(target_fx) / float(cam_params[0])
        scale_y = float(target_fy) / float(cam_params[1])
        scale = (scale_x + scale_y) * 0.5
        cam_model.focal_length = cam_params[:2] * scale
        cam_model.principal_point = cam_params[2:] * scale
        if scale != 1.0:
            resize_shape = (int(round(real_shape[1] * scale)), int(round(real_shape[0] * scale)))  # (width, height)

            # image resize
            rgb = rgb.unsqueeze(0)
            rgb = F.interpolate(rgb, size=(resize_shape[1], resize_shape[0]), mode='bilinear', align_corners=True)[0]

            # crop to original image size
            if resize_shape[0] >= real_shape[1]:
                crop_w_start = (resize_shape[0] - real_shape[1]) // 2
                rgb = rgb[:, :, crop_w_start:crop_w_start + real_shape[1]]
                real_shape[1] = real_shape[1]
                cam_model.principal_point[0] -= crop_w_start
            else:
                real_shape[1] = resize_shape[0]

            if resize_shape[1] >= real_shape[0]:
                crop_h_start = (resize_shape[1] - real_shape[0]) // 2
                rgb = rgb[:, crop_h_start:crop_h_start + real_shape[0], :]
                real_shape[0] = real_shape[0]
                cam_model.principal_point[1] -= crop_h_start
            else:
                real_shape[0] = resize_shape[1]
                

    uv_lidar, depth, _, refl = cam_model.project_pytorch(pc_rotated, real_shape, reflectance)
    uv_lidar = uv_lidar.t().int().contiguous()

    depth_img = torch.zeros(real_shape[:2], device=device, dtype=torch.float)
    depth_img += 1000.
    depth_img = visibility.depth_image(uv_lidar, depth, depth_img, uv_lidar.shape[0], real_shape[1],
                                       real_shape[0])
    temp_index = (depth_img == 1000.)
    depth_img[temp_index] = 0.
    depth_img_no_occlusion = depth_img

    # “진짜로 남은 점”만 필터링 (z-buffer 일치 검사)
    uv_lidar = uv_lidar.long()
    indexes = depth_img_no_occlusion[uv_lidar[:, 1], uv_lidar[:, 0]] == depth
    if _config['use_reflectance']:
        refl_img = torch.zeros(real_shape[:2], device=device, dtype=torch.float)
        refl_img[uv_lidar[indexes, 1], uv_lidar[indexes, 0]] = refl[0, indexes]
    
    # depth 정규화 및 채널 구성
    depth_img_no_occlusion /= _config['max_depth']
    depth_img_no_occlusion = depth_img_no_occlusion.unsqueeze(0)
    if _config['use_reflectance']:
        depth_img_no_occlusion = torch.cat((depth_img_no_occlusion, refl_img.unsqueeze(0)))


    # GT flow 만들기: RT가 유발한 픽셀 이동량 계산
    uv_lidar = uv_lidar[indexes]
    flow, _, new_indexes = get_flow_zforward(uv_lidar.float(), depth[indexes], RT, cam_model,
                                             [real_shape[0], real_shape[1], 3],
                                             scale_flow=False, reverse=False,
                                             get_valid_indexes=True)
    uv_flow = uv_lidar
    uv_flow = uv_flow[new_indexes].clone()
    flow = flow[new_indexes].clone()

    # dense flow 이미지와 마스크로 “뿌리기”
    flow_img = torch.zeros((real_shape[0], real_shape[1], 2), device=device, dtype=torch.float)
    flow_img[uv_flow[:, 1], uv_flow[:, 0]] = flow
    flow_mask = torch.zeros((real_shape[0], real_shape[1]), device=device, dtype=torch.int)
    flow_mask[uv_flow[:, 1], uv_flow[:, 0]] = 1

    # Argoverse(1920 width)면 half-scale 처리
    # Scale by half if the image is from the ARGO dataset
    if real_shape[1] == 1920 and _config['subset_argoverse']:
        flow_img, flow_mask = downsample_flow_and_mask(flow_img, flow_mask, 2, scale_flow=True)
        rgb = nn.functional.interpolate(rgb.unsqueeze(0), scale_factor=0.5)[0]
        depth_img_no_occlusion = downsample_depth(depth_img_no_occlusion.permute(1, 2, 0).contiguous(), 2)
        depth_img_no_occlusion = depth_img_no_occlusion.permute(2, 0, 1)
        real_shape[0] = real_shape[0] // 2
        real_shape[1] = real_shape[1] // 2

    # crop (mode 0: random, mode 1: valid-area random)
    if img_shape is not None:
        th, tw = img_shape
        h, w = real_shape[:2]
        crop_mode = _config.get('crop_mode', 0)

        # crop이 가능한 경우에만 수행하고, 작은 경우는 pad에서 처리
        if (h >= th) and (w >= tw):
            if crop_mode == 1:
                valid_mask = (flow_mask > 0)
                valid_ys, valid_xs = torch.where(valid_mask)
                if valid_ys.numel() > 0:
                    rand_idx = torch.randint(0, valid_ys.numel(), (1,))
                    cy = valid_ys[rand_idx].item()
                    cx = valid_xs[rand_idx].item()
                    top = cy - th // 2
                    left = cx - tw // 2
                else:
                    top = random.randint(0, h - th)
                    left = random.randint(0, w - tw)
            else:
                top = random.randint(0, h - th)
                left = random.randint(0, w - tw)

            top = max(0, min(top, h - th))
            left = max(0, min(left, w - tw))

            rgb = rgb[:, top:top + th, left:left + tw]
            depth_img_no_occlusion = depth_img_no_occlusion[:, top:top + th, left:left + tw]
            flow_img = flow_img[top:top + th, left:left + tw, :]
            flow_mask = flow_mask[top:top + th, left:left + tw]
            real_shape[0] = th
            real_shape[1] = tw


    # pad (오른쪽/아래만)로 목표 크기 맞춤
    if img_shape is not None and (img_shape[0] >= real_shape[0] or img_shape[1] >= real_shape[1]):
        # PAD ONLY ON RIGHT AND BOTTOM SIDE, IN ORDER TO BE CONSISTENT WITH FLOW
        shape_pad = [0, 0, 0, 0]

        shape_pad[3] = max(0, (img_shape[0] - real_shape[0]))  # // 2
        shape_pad[1] = max(0, (img_shape[1] - real_shape[1]))  # // 2 + 1

        rgb = F.pad(rgb, shape_pad)
        depth_img_no_occlusion = F.pad(depth_img_no_occlusion, shape_pad)
        flow_img = F.pad(flow_img.permute(2, 0, 1), shape_pad).permute(1, 2, 0)
        flow_mask = F.pad(flow_mask, shape_pad)

    # Fourier feature로 depth를 채널 확장 (옵션)
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
    return depth_img_no_occlusion, flow_img, flow_mask, rgb


# CNN training
def train(model, optimizer, scaler, rgb_img, lidar_img, target_flow, target_mask, _config):
    model.train()

    optimizer.zero_grad()

    with amp.autocast(enabled=_config['amp']):
        # Run model
        predicted_flow = model(rgb_img, lidar_img)

        predicted_flow, predicted_uncertainty = predicted_flow

        # Calculate Loss
        flow_loss, metrics = RAFT_loss2(predicted_flow, predicted_uncertainty, target_flow[0], target_mask[0],
                                        upsample=False,
                                        weight_nll=_config['weight_nll'], unc_type=_config['der_type'])

    #  EPE
    epe = metrics['epe']

    total_loss = flow_loss

    # Backpropagation
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    scaler.step(optimizer)
    scaler.update()

    del predicted_flow
    return total_loss.detach(), epe.detach()


# CNN test
def test(model, rgb_img, lidar_img, target_flow, target_mask, log_image, _config):
    global EPOCH
    model.eval()

    # Run model
    with torch.no_grad():
        with amp.autocast(enabled=False):
            predicted_flow = model(rgb_img, lidar_img)

    predicted_flow, predicted_uncertainty = predicted_flow

    # Calculate Loss
    flow_loss, metrics = RAFT_loss2(predicted_flow, predicted_uncertainty, target_flow[0], target_mask[0],
                                    upsample=False,
                                    weight_nll=_config['weight_nll'], unc_type=_config['der_type'])

    total_loss = flow_loss

    #  EPE
    gt = torch.cat((target_flow[0], target_mask[0][:, 0:1, :, :]), dim=1)
    epe = metrics['epe']
    f1 = metrics['f1']

    # Expected Calibration Error (Uncertainty Estimation)
    ece_dict, ece_u, ece_v = None, None, None
    if _config['uncertainty']:
        ece_u, ece_v = get_ECE(predicted_flow[-1], predicted_uncertainty[-1], target_flow[0], target_mask[0],
                               loss_type=_config['der_type'])

    # Log images in Weights&Biases
    if log_image:
        rgb_images = []
        predicted_flow_images = []
        target_flow_images = []
        uncertainties = []
        uncertainties2 = []
        upsampled = resize_dense_vector(predicted_flow[-1], gt.size(2), gt.size(3))
        for i in range(min(rgb_img.shape[0], 16)):
            img_show = rgb_img[i].permute(1, 2, 0).cpu().numpy()
            std = [0.229, 0.224, 0.225]
            mean = [0.485, 0.456, 0.406]
            if _config['normalize_images']:
                img_show = img_show * std + mean
            else:
                img_show = img_show / 255.
            img_show = img_show.clip(0, 1)
            if predicted_uncertainty[-1] is not None:
                rgb_images.append(wandb.Image(img_show, grouping=5))
            else:
                rgb_images.append(wandb.Image(img_show, grouping=3))

            flowcolored_pred = flow_to_color(upsampled[i].permute(1, 2, 0).cpu().numpy())
            predicted_flow_images.append(wandb.Image(flowcolored_pred))

            flowcolored_gt = flow_to_color((gt[i].permute(1, 2, 0)[:, :, :2]).cpu().numpy())
            target_flow_images.append(wandb.Image(flowcolored_gt))
            if predicted_uncertainty[-1] is not None:
                uncertainties.append(wandb.Image(uncertainty_to_color((predicted_uncertainty[-1][i]).cpu().numpy())))
                uncertainties2.append(
                    wandb.Image(uncertainty_to_color((target_mask[0][i] * predicted_uncertainty[-1][i]).cpu().numpy())))
        if predicted_uncertainty[-1] is not None:
            images_for_wandb = list(chain.from_iterable(zip(rgb_images, predicted_flow_images,
                                                            target_flow_images, uncertainties, uncertainties2)))
        else:
            images_for_wandb = list(chain.from_iterable(zip(rgb_images, predicted_flow_images,
                                                            target_flow_images)))
        wandb.log({'examples': images_for_wandb}, commit=False)

    del predicted_flow
    return total_loss.detach(), epe.detach(), ece_u, ece_v, ece_dict, f1


def main(gpu, _config, common_seed, world_size):
    global EPOCH
    rank = gpu

    # Setup Distributed Training
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Set random seeds
    local_seed = (common_seed + common_seed ** gpu) ** 2
    local_seed = local_seed % (2 ** 32 - 1)
    np.random.seed(common_seed)
    torch.random.manual_seed(common_seed)
    torch.cuda.set_device(gpu)
    device = torch.device(gpu)
    print(f"Process {rank}, seed {common_seed}")

    # Setup Weights&Biases
    wandb_run_id = 'remove'
    if _config['wandb'] and rank == 0:
        if _config['resume_id'] is None:
            wandb.init(config=_config)
        else:
            wandb.init(id=_config['resume_id'], resume="must")
        wandb_run_id = wandb.run.id
        print("RUN ID: ", wandb_run_id)

    if rank == 0:
        logger = init_logger(f'/tmp/{wandb_run_id}.log', _config['resume'], _config['wandb'])

    img_shape = _config['img_shape']

    if not os.path.exists(_config["savemodel"]) and rank == 0:
        os.mkdir(_config["savemodel"])
    _config["savemodel"] = os.path.join(_config["savemodel"], wandb_run_id)
    if not os.path.exists(_config["savemodel"]) and rank == 0:
        os.mkdir(_config["savemodel"])

    # Training and test set creation
    num_worker = _config['num_worker']
    batch_size = _config['batch_size']
    starting_epoch = 0

    model = get_model(_config, img_shape)

    # Load weights if resuming training
    if _config['weights'] is not None:
        if rank == 0:
            logger.info(f"Loading weights from {_config['weights']}")
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        saved_state_dict = checkpoint['state_dict']
        clean_state_dict = saved_state_dict
        if not _config['unc_freeze'] or _config['finetune']:
            model.load_state_dict(clean_state_dict, strict=True)
        else:
            model.load_state_dict(clean_state_dict, strict=False)
            for name, param in model.named_parameters():
                if 'update_block_unc' not in name:
                    param.requires_grad = False

    model.train()
    model = DistributedDataParallel(model.to(device), device_ids=[rank], output_device=rank,
                                    find_unused_parameters=_config['find_unused_parameter'])

    # Setup Optimizer and Scheduler
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.Adam(parameters, lr=_config['BASE_LEARNING_RATE'], weight_decay=5e-6)
    scheduler = None

    scaler = amp.GradScaler(enabled=_config['amp'])
    if rank == 0 and _config['amp']:
        logger.info("Using Mixed Precision")

    # Load optimizer and scheduler state if resuming training
    if _config['weights'] is not None and _config['resume']:
        checkpoint = torch.load(_config['weights'], map_location='cpu')
        opt_state_dict = checkpoint['optimizer']
        optimizer.load_state_dict(opt_state_dict)
        starting_epoch = checkpoint['epoch'] + 1

    if rank == 0:
        logger.info('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    if _config['wandb'] and rank == 0:
        wandb.watch(model)

    start_full_time = time.time()
    BEST_VAL_EPE = 10000.
    old_save_filename = None

    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    debug_save_dir = _config['save_dir'] + "/"
    debug_save_dir += _config['dataset'] + "/train/"
    if not os.path.exists(debug_save_dir):
        os.makedirs(debug_save_dir)
    else:
        import shutil
        shutil.rmtree(debug_save_dir)
        os.makedirs(debug_save_dir)
    debug_input_data_dir = debug_save_dir + "input_data/"
    os.makedirs(debug_input_data_dir)

    # total_iter = starting_epoch * len(dataset)
    total_iter = 0
    dataset_custom = None
    for epoch in range(starting_epoch, _config['epochs']):

        if _config['custom']:
            train_directories_custom = []
            for subdir in ['library_1', 'library_3', 'parking_lot_4', 'SC_1', 'SC_3', 'island_1', 'island_2', 'parking_lot_1']:
                train_directories_custom.append(os.path.join(os.path.join(_config['data_folder_custom'], subdir), 'CMRNext'))

            dataset_custom = DatasetGeneralExtrinsicCalib(train_directories_custom, train=True, max_r=_config['max_r'],
                                                         max_t=_config['max_t'],
                                                         use_reflectance=_config['use_reflectance'],
                                                         normalize_images=_config['normalize_images'],
                                                         dataset='custom', sensor_type=_config['sensor_type'], downsample=_config['downsize'])

            dataset_train = dataset_custom

        elif _config['dataset'] == 'hercules':
            train_directories_hercules = []
            # for subdir in ['library_1', 'library_3', 'parking_lot_1', 'parking_lot_4', 'SC_1', 'SC_3', 'island_1', 'island_2']:
            for subdir in ['parking_lot_1']:
                train_directories_hercules.append(os.path.join(os.path.join(_config['data_folder_custom'], subdir), 'offline'))

            dataset_hercules = DatasetGeneralExtrinsicCalib(train_directories_hercules, train=True, max_r=_config['max_r'],
                                                         max_t=_config['max_t'],
                                                         use_reflectance=_config['use_reflectance'],
                                                         normalize_images=_config['normalize_images'],
                                                         dataset='hercules', sensor_type=_config['sensor_type'], downsample=_config['downsize'])
            dataset_train = dataset_hercules

        else:
            train_directories_kitti = []
            for subdir in ['03', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                           '20', '21']:
                train_directories_kitti.append(os.path.join(_config['data_folder_kitti'], subdir))
            train_directories_argo = []
            base_dir = _config['data_folder_argo']
            for subdir in ['train1', 'train2', 'train3']:
                train_directories_argo.append(os.path.join(base_dir, subdir))
            train_directories_pandaset = []
            base_dir = _config['data_folder_panda']
            for subdir in os.listdir(base_dir):
                seq_num = int(subdir)
                if (57 <= int(seq_num) <= 78) or int(seq_num) == 149:
                    continue
                if subdir in ['011', '122', '124', '030', '109', '043', '084', '115', '090']:
                    continue
                train_directories_pandaset.append(os.path.join(base_dir, subdir))

            dataset_kitti = DatasetGeneralExtrinsicCalib(train_directories_kitti, train=True, max_r=_config['max_r'],
                                                         max_t=_config['max_t'],
                                                         use_reflectance=_config['use_reflectance'],
                                                         normalize_images=_config['normalize_images'],
                                                         dataset='kitti')
            if not _config['kitti_only']:
                dataset_argo = DatasetGeneralExtrinsicCalib(train_directories_argo, train=True, max_r=_config['max_r'],
                                                            max_t=_config['max_t'],
                                                            use_reflectance=_config['use_reflectance'],
                                                            normalize_images=_config['normalize_images'],
                                                            dataset='argoverse', cam='ring_front_center')
                dataset_pandaset = DatasetPandasetExtrinsicCalib(train_directories_pandaset, train=True,
                                                                 max_r=_config['max_r'], max_t=_config['max_t'],
                                                                 use_reflectance=_config['use_reflectance'],
                                                                 normalize_images=_config['normalize_images'],
                                                                 sensor_id=0, camera='front_camera')
                dataset_pandaset2 = DatasetPandasetExtrinsicCalib(train_directories_pandaset, train=True,
                                                                  max_r=_config['max_r'], max_t=_config['max_t'],
                                                                  use_reflectance=_config['use_reflectance'],
                                                                  normalize_images=_config['normalize_images'],
                                                                  sensor_id=1, camera='front_camera')
                assert len(dataset_argo) != 0 and len(dataset_kitti) != 0 and len(dataset_pandaset) != 0 and len(
                    dataset_pandaset2) != 0, "Something wrong with the dataset"

            # print("Len Kitti Dataset: ", len(dataset_kitti))
            # print("Len Argo Dataset: ", len(dataset_argo))
            # print("Len Panda Dataset: ", 2*len(dataset_pandaset))

            if _config['kitti_only']:
                dataset_train = dataset_kitti
            else:
                kitti_idxs = np.arange(0, len(dataset_kitti))
                np.random.shuffle(kitti_idxs)
                dataset_kitti = torch.utils.data.Subset(dataset_kitti, kitti_idxs[:len(dataset_pandaset) * 2])
                argo_idxs = np.arange(0, len(dataset_argo))
                np.random.shuffle(argo_idxs)
                dataset_argo = torch.utils.data.Subset(dataset_argo, argo_idxs[:len(dataset_pandaset) * 2])
                dataset_train = torch.utils.data.ConcatDataset(
                    [dataset_argo, dataset_kitti, dataset_pandaset, dataset_pandaset2])
                if len(dataset_train) != 36000 and rank == 0:
                    logger.warning(f"Dataset size is different than what is should be:\n"
                                   f"Expected size: 36000, Current size: {len(dataset_train)}")


        # Validation set creation
        if _config['custom']:
            test_directories_custom = []
            for subdir in ['parking_lot_2']:
                test_directories_custom.append(os.path.join(os.path.join(_config['data_folder_custom'], subdir), 'CMRNext'))

            dataset_val_custom = DatasetGeneralExtrinsicCalib(test_directories_custom, train=True, max_r=_config['max_r'],
                                                          max_t=_config['max_t'],
                                                          use_reflectance=_config['use_reflectance'],
                                                          normalize_images=_config['normalize_images'],
                                                          dataset='custom',sensor_type=_config['sensor_type'], downsample=_config['downsize'])

            dataset_val = dataset_val_custom

            print ("Len Custom Val Dataset: ", len(dataset_val))
        
        elif _config['dataset'] == 'hercules':
            test_directories_hercules = []
            # for subdir in ['parking_lot_2']:
            for subdir in ['parking_lot_1']:
                test_directories_hercules.append(os.path.join(os.path.join(_config['data_folder_custom'], subdir), 'offline'))

            dataset_val_hercules = DatasetGeneralExtrinsicCalib(test_directories_hercules, train=True, max_r=_config['max_r'],
                                                          max_t=_config['max_t'],
                                                          use_reflectance=_config['use_reflectance'],
                                                          normalize_images=_config['normalize_images'],
                                                          dataset='hercules',sensor_type=_config['sensor_type'], downsample=_config['downsize'])

            dataset_val = dataset_val_hercules

            print ("Len Hercules Val Dataset: ", len(dataset_val))

        else:
            test_directories_kitti = []
            for subdir in ['00']:
                test_directories_kitti.append(os.path.join(_config['data_folder_kitti'], subdir))
            test_directories_argo = []
            base_dir = _config['data_folder_argo']
            for subdir in ['train1', 'train2', 'train3']:
                test_directories_argo.append(os.path.join(base_dir, subdir))
            test_directories_pandaset = []
            base_dir = _config['data_folder_panda']
            for subdir in ['011', '122', '124', '030', '109', '043', '084', '115', '090']:
                test_directories_pandaset.append(os.path.join(base_dir, subdir))

            dataset_val_kitti = DatasetGeneralExtrinsicCalib(test_directories_kitti, train=False, max_r=_config['max_r'],
                                                             max_t=_config['max_t'],
                                                             use_reflectance=_config['use_reflectance'],
                                                             normalize_images=_config['normalize_images'],
                                                             dataset='kitti')

            dataset_val = dataset_val_kitti

        train_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train,
            num_replicas=world_size,
            rank=rank,
            seed=common_seed,
            shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_val,
            num_replicas=world_size,
            rank=rank,
            seed=common_seed
        )
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        init_fn = partial(_init_fn, epoch=epoch, seed=local_seed)
        TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train,
                                                     shuffle=False,
                                                     batch_size=batch_size,
                                                     num_workers=num_worker,
                                                     worker_init_fn=init_fn,
                                                     collate_fn=merge_inputs,
                                                     drop_last=True,
                                                     sampler=train_sampler,
                                                     pin_memory=True)

        TestImgLoader = torch.utils.data.DataLoader(dataset=dataset_val,
                                                    shuffle=False,
                                                    batch_size=_config['eval_batch_size'],
                                                    num_workers=num_worker,
                                                    worker_init_fn=init_fn,
                                                    collate_fn=merge_inputs,
                                                    drop_last=True,
                                                    sampler=val_sampler,
                                                    pin_memory=True)

        if rank == 0:
            logger.info(f'Len Train: {len(TrainImgLoader)}')
            logger.info(f'Len Test: {len(TestImgLoader)}')
            logger.info(f'This is {epoch}-th epoch')


        if epoch == starting_epoch:
            if starting_epoch == 0:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, _config['BASE_LEARNING_RATE'],
                                                                epochs=_config['epochs'],
                                                                steps_per_epoch=len(dataset_train) // (
                                                                        batch_size * world_size),
                                                                pct_start=0.4, div_factor=10,
                                                                final_div_factor=100000)
            else:
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, _config['BASE_LEARNING_RATE'],
                                                                epochs=_config['epochs'] + 1,
                                                                steps_per_epoch=len(dataset_train) // (
                                                                        batch_size * world_size),
                                                                pct_start=0.4, div_factor=10,
                                                                final_div_factor=100000, last_epoch=starting_epoch * (
                            len(dataset_train) // (batch_size * world_size)))
            total_iter = starting_epoch * len(dataset_train)

        EPOCH = epoch
        epoch_start_time = time.time()
        total_train_loss = 0
        local_loss = 0.
        local_epe = 0.
        total_train_epe = 0.
        if epoch != starting_epoch:
            if _config['wandb'] and rank == 0:
                wandb.log({'LR': scheduler.get_last_lr()[0]}, commit=False)

        time_for_N_it = time.time()
        batch_idx = 0
        # Training #
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            lidar_input = []
            rgb_input = []

            target_flow1 = []
            target_flow2 = []
            target_flow3 = []
            target_flow4 = []
            target_flow5 = []
            target_flow6 = []
            target_mask1 = []
            target_mask2 = []
            target_mask3 = []
            target_mask4 = []
            target_mask5 = []
            target_mask6 = []

            sample['tr_error'] = sample['tr_error'].to(device)
            sample['rot_error'] = sample['rot_error'].to(device)

            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose

                depth_img_no_occlusion, flow_img, flow_mask, rgb = prepare_input(_config, device,
                                                                                 idx, img_shape, mean, sample, std, aligned=False)
                if _config['debug']:
                    aligned_img_no_occlusion, _, _, aligned_rgb = prepare_input(_config, device,
                                                                    idx, img_shape, mean, sample, std, aligned=True)
                flow_img = flow_img.contiguous()
                flow_mask = flow_mask.contiguous()
                rgb_input.append(rgb)
                lidar_input.append(depth_img_no_occlusion)
                target_flow1.append(flow_img.permute(2, 0, 1).clone())
                target_mask1.append(flow_mask.repeat(2, 1, 1).float().clone())

                down_flow2, down_mask2 = downsample_flow_and_mask(flow_img, flow_mask, 4)
                down_flow3, down_mask3 = downsample_flow_and_mask(down_flow2, down_mask2, 2)
                down_flow4, down_mask4 = downsample_flow_and_mask(down_flow3, down_mask3, 2)
                down_flow5, down_mask5 = downsample_flow_and_mask(down_flow4, down_mask4, 2)
                down_flow6, down_mask6 = downsample_flow_and_mask(down_flow5, down_mask5, 2)

                target_flow2.append(down_flow2.permute(2, 0, 1).clone())
                target_mask2.append(down_mask2.repeat(2, 1, 1).float().clone())
                target_flow3.append(down_flow3.permute(2, 0, 1).clone())
                target_mask3.append(down_mask3.repeat(2, 1, 1).float().clone())
                target_flow4.append(down_flow4.permute(2, 0, 1).clone())
                target_mask4.append(down_mask4.repeat(2, 1, 1).float().clone())
                target_flow5.append(down_flow5.permute(2, 0, 1).clone())
                target_mask5.append(down_mask5.repeat(2, 1, 1).float().clone())
                target_flow6.append(down_flow6.permute(2, 0, 1).clone())
                target_mask6.append(down_mask6.repeat(2, 1, 1).float().clone())

                if _config['debug']:
                    def _depth_overlay(depth_tensor, base_rgb):
                        depth_np = depth_tensor[0].detach().cpu().numpy()
                        depth_mask = depth_np > 0
                        if depth_mask.any():
                            depth_vals = depth_np[depth_mask]
                            lo = np.percentile(depth_vals, 1)
                            hi = np.percentile(depth_vals, 99)
                            if hi <= lo:
                                lo, hi = depth_vals.min(), depth_vals.max()
                            depth_norm = (depth_np - lo) / (hi - lo + 1e-8)
                            depth_norm = np.clip(depth_norm, 0, 1)
                            depth_colors = (cm.jet(depth_norm)[:, :, :3] * 255.0).astype(np.uint8)
                            overlay = base_rgb.copy()
                            overlay[depth_mask] = depth_colors[depth_mask]
                            return overlay
                        return base_rgb
                    
                    def _flow_arrows(flow_tensor, mask_tensor, base_rgb, step=20):
                        arrows = base_rgb.copy()
                        flow_np = flow_tensor.detach().cpu().numpy()
                        mask_np = mask_tensor.detach().cpu().numpy()
                        h, w = flow_np.shape[:2]
                        for y in range(0, h, step):
                            for x in range(0, w, step):
                                if mask_np[y, x] == 0:
                                    continue
                                dx, dy = flow_np[y, x]
                                if dx == 0 and dy == 0:
                                    continue
                                pt1 = (int(x), int(y))
                                pt2 = (int(x + dx), int(y + dy))
                                cv2.arrowedLine(arrows, pt1, pt2, color=(0, 255, 0), thickness=1, tipLength=0.3)
                        return arrows

                    # print("RGB")
                    # io.imshow(rgb.permute(1, 2, 0).cpu().numpy())
                    # print("Flow Mask")
                    # io.imshow(flow_mask.cpu().numpy())
                    # io.show()
                    # print("Flow image")
                    # io.imshow(flow_to_color(flow_img.cpu().numpy()))
                    # io.show()
                    # print("Depth image")
                    # io.imshow(depth_img_no_occlusion[0].cpu().numpy())
                    # io.show()

                    # save images for debug
                    base_name = f"rank{rank}_e{epoch}_b{batch_idx}_i{idx}"

                    rgb_np = rgb.permute(1, 2, 0).detach().cpu().numpy()
                    aligned_rgb_np = aligned_rgb.permute(1, 2, 0).detach().cpu().numpy()
                    if _config['normalize_images']:
                        raw_rgb = (rgb_np * std.cpu().numpy()) + mean.cpu().numpy()
                        raw_rgb = np.clip(raw_rgb * 255.0, 0, 255).astype(np.uint8)
                        aligned_raw_rgb = (aligned_rgb_np * std.cpu().numpy()) + mean.cpu().numpy()
                        aligned_raw_rgb = np.clip(aligned_raw_rgb * 255.0, 0, 255).astype(np.uint8)
                    else:
                        raw_rgb = np.clip(rgb_np, 0, 255).astype(np.uint8)
                        aligned_raw_rgb = np.clip(aligned_rgb_np, 0, 255).astype(np.uint8)
                    pc_overlay = _depth_overlay(depth_img_no_occlusion, raw_rgb)
                    aligned_overlay = _depth_overlay(aligned_img_no_occlusion, aligned_raw_rgb)
                    flow_arrows = _flow_arrows(flow_img, flow_mask, raw_rgb)
                    concat_img = np.concatenate([aligned_overlay, pc_overlay, flow_arrows], axis=1)
                    io.imsave(os.path.join(debug_input_data_dir, f"{base_name}_debug_concat.png"), concat_img)


            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            target_flow1 = torch.stack(target_flow1)
            target_flow2 = torch.stack(target_flow2)
            target_flow3 = torch.stack(target_flow3)
            target_flow4 = torch.stack(target_flow4)
            target_flow5 = torch.stack(target_flow5)
            target_flow6 = torch.stack(target_flow6)
            target_mask1 = torch.stack(target_mask1)
            target_mask2 = torch.stack(target_mask2)
            target_mask3 = torch.stack(target_mask3)
            target_mask4 = torch.stack(target_mask4)
            target_mask5 = torch.stack(target_mask5)
            target_mask6 = torch.stack(target_mask6)

            loss, epe = train(model, optimizer, scaler, rgb_input, lidar_input,
                              [target_flow1, target_flow2, target_flow3, target_flow4, target_flow5, target_flow6],
                              [target_mask1, target_mask2, target_mask3, target_mask4, target_mask5, target_mask6],
                              _config)

            dist.barrier()
            dist.reduce(loss, 0)
            dist.reduce(epe, 0)
            if _config['scheduler'].startswith('cycle'):
                scheduler.step()
            if rank == 0:
                loss = loss / world_size
                epe = epe / world_size
                local_loss += loss.item()
                local_epe += epe.item()

                if batch_idx % _config['print_every'] == 0 and batch_idx != 0:
                    logger.info('Iter %d/%d training loss = %.3f , epe = %.3f, time = %.2f, time for %d it = %.2f' %
                                (batch_idx,
                                 len(TrainImgLoader),
                                 local_loss / _config['print_every'],
                                 local_epe / _config['print_every'],
                                 (time.time() - start_time) / lidar_input.shape[0],
                                 _config['print_every'],
                                 (time.time() - time_for_N_it)))
                    time_for_N_it = time.time()
                    if _config['wandb']:
                        wandb.log({'Loss': local_loss / _config['print_every']}, step=total_iter)
                        wandb.log({'EPE': local_epe / _config['print_every']}, step=total_iter)
                    local_loss = 0.
                    local_epe = 0.
                total_train_loss += loss.item()
                total_train_epe += epe.item()
                total_iter += len(sample['rgb']) * world_size
            del loss, epe
            del target_mask1, target_mask2, target_mask3, target_mask4, target_mask5, target_mask6
            del target_flow1, target_flow2, target_flow3, target_flow4, target_flow5, target_flow6
            del rgb_input, lidar_input

        if rank == 0:
            logger.info("------------------------------------")
            logger.info('epoch %d total training loss = %.3f, total training epe = %.3f' %
                        (epoch, total_train_loss / batch_idx, total_train_epe / batch_idx))
            logger.info('Total epoch time = %.2f' % (time.time() - epoch_start_time))
            logger.info("------------------------------------")
            if _config['wandb']:
                wandb.log({'Total training loss': total_train_loss / batch_idx, 'epoch': epoch}, commit=False)
                wandb.log({'Total training EPE': total_train_epe / batch_idx}, commit=False)

        ## Test ##
        total_test_loss = 0.
        total_test_epe = 0.
        total_test_f1 = 0.
        total_test_ece_u = 0.
        total_test_ece_v = 0.

        local_loss = 0.0
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            lidar_input = []
            rgb_input = []

            target_flow1 = []
            target_flow2 = []
            target_flow3 = []
            target_flow4 = []
            target_flow5 = []
            target_flow6 = []
            target_mask1 = []
            target_mask2 = []
            target_mask3 = []
            target_mask4 = []
            target_mask5 = []
            target_mask6 = []

            sample['tr_error'] = sample['tr_error'].to(device)
            sample['rot_error'] = sample['rot_error'].to(device)

            for idx in range(len(sample['rgb'])):
                # ProjectPointCloud in RT-pose

                depth_img_no_occlusion, flow_img, flow_mask, rgb = prepare_input(_config, device,
                                                                                 idx, img_shape, mean, sample, std)

                flow_img = flow_img.contiguous()
                flow_mask = flow_mask.contiguous()
                rgb_input.append(rgb)
                lidar_input.append(depth_img_no_occlusion)

                target_flow1.append(flow_img.permute(2, 0, 1).clone())
                target_mask1.append(flow_mask.repeat(2, 1, 1).float().clone())

                down_flow2, down_mask2 = downsample_flow_and_mask(flow_img, flow_mask, 4)
                down_flow3, down_mask3 = downsample_flow_and_mask(down_flow2, down_mask2, 2)
                down_flow4, down_mask4 = downsample_flow_and_mask(down_flow3, down_mask3, 2)
                down_flow5, down_mask5 = downsample_flow_and_mask(down_flow4, down_mask4, 2)
                down_flow6, down_mask6 = downsample_flow_and_mask(down_flow5, down_mask5, 2)

                target_flow2.append(down_flow2.permute(2, 0, 1).clone())
                target_mask2.append(down_mask2.repeat(2, 1, 1).float().clone())
                target_flow3.append(down_flow3.permute(2, 0, 1).clone())
                target_mask3.append(down_mask3.repeat(2, 1, 1).float().clone())
                target_flow4.append(down_flow4.permute(2, 0, 1).clone())
                target_mask4.append(down_mask4.repeat(2, 1, 1).float().clone())
                target_flow5.append(down_flow5.permute(2, 0, 1).clone())
                target_mask5.append(down_mask5.repeat(2, 1, 1).float().clone())
                target_flow6.append(down_flow6.permute(2, 0, 1).clone())
                target_mask6.append(down_mask6.repeat(2, 1, 1).float().clone())

            lidar_input = torch.stack(lidar_input)
            rgb_input = torch.stack(rgb_input)
            target_flow1 = torch.stack(target_flow1)
            target_flow2 = torch.stack(target_flow2)
            target_flow3 = torch.stack(target_flow3)
            target_flow4 = torch.stack(target_flow4)
            target_flow5 = torch.stack(target_flow5)
            target_flow6 = torch.stack(target_flow6)
            target_mask1 = torch.stack(target_mask1)
            target_mask2 = torch.stack(target_mask2)
            target_mask3 = torch.stack(target_mask3)
            target_mask4 = torch.stack(target_mask4)
            target_mask5 = torch.stack(target_mask5)
            target_mask6 = torch.stack(target_mask6)

            save_images = (batch_idx == 0) and _config['wandb'] and rank == 0

            loss, epe, ece_u, ece_v, ece_dict, f1 = test(model, rgb_input, lidar_input,
                                                         [target_flow1, target_flow2, target_flow3, target_flow4,
                                                          target_flow5, target_flow6],
                                                         [target_mask1, target_mask2, target_mask3, target_mask4,
                                                          target_mask5, target_mask6],
                                                         save_images, _config)
            dist.barrier()
            dist.reduce(loss, 0)
            dist.reduce(epe, 0)
            if f1 is not None:
                dist.reduce(f1, 0)
            if _config['uncertainty']:
                dist.reduce(ece_u, 0)
                dist.reduce(ece_v, 0)
            if rank == 0:
                loss = loss / world_size
                epe = epe / world_size
                if f1 is not None:
                    f1 = f1 / world_size
                if _config['uncertainty']:
                    ece_u = ece_u / world_size
                    ece_v = ece_v / world_size
                local_loss += loss.item()

                if batch_idx % 50 == 0 and batch_idx != 0:
                    logger.info('Iter %d test loss = %.3f , time = %.2f' %
                                (batch_idx, local_loss / 50,
                                 (time.time() - start_time) / lidar_input.shape[0]))
                    local_loss = 0.0
                total_test_loss += loss.item()
                total_test_epe += epe.item()
                if f1 is not None:
                    total_test_f1 += f1.item()
                if _config['uncertainty']:
                    total_test_ece_u += ece_u.item()
                    total_test_ece_v += ece_v.item()
            del loss, epe
            del target_mask1, target_mask2, target_mask3, target_mask4, target_mask5, target_mask6
            del target_flow1, target_flow2, target_flow3, target_flow4, target_flow5, target_flow6
            del rgb_input, flow_img, flow_mask, rgb, lidar_input
            del down_flow2, down_flow3, down_flow4, down_flow5, down_flow6
            del down_mask2, down_mask3, down_mask4, down_mask5, down_mask6
            del depth_img_no_occlusion

        if rank == 0:
            logger.info("------------------------------------")
            logger.info('total test loss = %.3f' % (total_test_loss / batch_idx))
            logger.info('total test epe = %.3f' % (total_test_epe / batch_idx))
            logger.info("------------------------------------")

            if _config['wandb']:
                wandb.log({'Val Loss': total_test_loss / batch_idx,
                           'Val EPE': total_test_epe / batch_idx}, commit=False)
                if f1 is not None:
                    wandb.log({'Val F1': total_test_f1 / batch_idx}, commit=False)
                if _config['uncertainty']:
                    wandb.log({'ECE u': total_test_ece_u / batch_idx,
                               'ECE v': total_test_ece_v / batch_idx}, commit=False)

        # SAVE
        val_epe = total_test_epe / batch_idx
        if rank == 0 and _config['wandb']:
            wandb.save(f'/tmp/{wandb_run_id}.log')
            torch.save({
                'config': _config,
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(dataset_val),
                'test_loss': total_test_loss / len(dataset_val),
                'train_epe': total_train_epe / len(dataset_val),
                'test_epe': total_test_epe / len(dataset_val),
            }, f'{_config["savemodel"]}/last_iter_checkpoint.tar')
        if val_epe < BEST_VAL_EPE and rank == 0:
            BEST_VAL_EPE = val_epe
            savefilename = f'{_config["savemodel"]}/checkpoint_{epoch}_{val_epe:.3f}.tar'
            torch.save({
                'config': _config,
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': total_train_loss / len(dataset_val),
                'test_loss': total_test_loss / len(dataset_val),
                'train_epe': total_train_epe / len(dataset_val),
                'test_epe': total_test_epe / len(dataset_val),
            }, savefilename)
            logger.info(f'Model saved as {savefilename}')
            if _config['wandb']:
                wandb.run.summary["Best Val EPE"] = BEST_VAL_EPE
                torch.save({
                    'config': _config,
                    'epoch': epoch,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'train_loss': total_train_loss / len(dataset_val),
                    'test_loss': total_test_loss / len(dataset_val),
                    'train_epe': total_train_epe / len(dataset_val),
                    'test_epe': total_test_epe / len(dataset_val),
                }, './best_model_so_far.tar')
                wandb.save('./best_model_so_far.tar')
            if old_save_filename is not None:
                if os.path.exists(old_save_filename):
                    os.remove(old_save_filename)
            old_save_filename = savefilename

        # Cleanup
        del sample, dataset_train, dataset_val, TrainImgLoader

    if rank == 0:
        logger.info('full training time = %.2f HR' % ((time.time() - start_full_time) / 3600))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def real_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--savemodel', type=str, default='/media/RAIDONE/CATTANEOD/regnet/checkpoints_CMRFlowNet/')
    parser.add_argument('--data_folder_argo', type=str, default='/media/DATA/ARGO/only_center_camera/')
    parser.add_argument('--data_folder_kitti', type=str, default='/home/cattaneod/Datasets/KITTI/sequences/')
    parser.add_argument('--data_folder_panda', type=str, default='/media/RAIDONE/DATASETS/pandaset')
    parser.add_argument('--data_folder_custom', type=str, default='/ws/data/LG_Innotek/VIVID/CMRNext')
    parser.add_argument('--custom', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--use_reflectance', action='store_true', default=False)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--BASE_LEARNING_RATE', type=float, default=3e-4)
    parser.add_argument('--max_t', type=float, default=1.5)
    parser.add_argument('--max_r', type=float, default=20.)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--num_worker', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--max_depth', type=float, default=160.)
    parser.add_argument('--resume', action='store_true', default=False)
    parser.add_argument('--resume_id', type=str, default=None)
    parser.add_argument('--no_scheduler', default=False, action='store_true')
    parser.add_argument('--scheduler', type=str, default="cycle_one")
    parser.add_argument('--upsample_method', type=str, default="transposed")
    parser.add_argument('--img_shape', type=int, nargs=2, default=[320, 960])
    parser.add_argument('--wandb', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--not_normalize_images', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--master_port', type=str, default=None)
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--gpu_count', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--amp', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--print_every', type=int, default=500)
    parser.add_argument('--uncertainty', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--kitti_only', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--fourier_levels', type=int, default=12)
    parser.add_argument('--weight_nll', type=float, default=-1.0)
    parser.add_argument('--der_type', type=str, default="NLL",
                        choices=["NLL"])  # Type of uncertainty
    parser.add_argument('--unc_freeze', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--finetune', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--find_unused_parameter', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--context_encoder', type=str, default="lidar", choices=["lidar"])
    parser.add_argument('--sensor_type', type=str, default="lidar")
    parser.add_argument('--downsize', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--crop_mode', type=int, default=0)
    parser.add_argument('--dataset', type=str, default="kitti")
    parser.add_argument('--save_dir', type=str, default='./logs/')

    args = parser.parse_args()
    # print(args)
    _config = vars(args)
    if _config['uncertainty']:
        raise NotImplementedError("Training with uncertainty is still untested.")
    if _config['no_scheduler']:
        _config['scheduler'] = False
    _config['normalize_images'] = not _config['not_normalize_images']
    _config['subset_argoverse'] = True

    if _config['gpu_count'] == -1:
        _config['gpu_count'] = torch.cuda.device_count()
    world_size = _config['gpu_count']
    os.environ['MASTER_ADDR'] = 'localhost'
    _config['MASTER_PORT'] = f'{np.random.randint(8000, 9999)}'
    if args.master_port is not None:
        _config['MASTER_PORT'] = args.master_port
    os.environ['MASTER_PORT'] = _config['MASTER_PORT']
    if _config['gpu'] == -1:
        mp.spawn(main, nprocs=world_size, args=(_config, _config['seed'], world_size,))
    else:
        main(_config['gpu'], _config, _config['seed'], world_size)


if __name__ == '__main__':
    real_main()