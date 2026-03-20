# Optional W&B logging (uncomment to enable):
    # --wandb \
    # --wandb_project cmrnext-calib \
    # --wandb_entity LGIT_calib \
    # --wandb_group hercules_finetune \
    # --wandb_name cmrnext_ft_lidar_G2_r20_t1.5_260201 \
    # --wandb_tags hercules,finetune,lidar,r20,t1.5 \
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python3 train_calibration.py \
    --savemodel /ws/data/Checkpoint \
    --data_folder_custom /ws/data/PublicDataset \
    --max_r 20 \
    --max_t 1.5 \
    --weights /ws/data/Checkpoint/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter1.tar \
    --finetune \
    --epochs 50 \
    --BASE_LEARNING_RATE 3e-5 \
    --batch_size 4 \
    --crop_mode 1 \
    --resize_mode 2  \
    --save_dir /ws/output  \
    --save_model_name cmrnext_kittiArgoversePandasetHerculesLidar_20_1.5_260216_woDownsize \
    --dataset hercules  \
    --data_type lg_custom \
    --image_name image_left \
    --pcl_name lidar_Aeva \
    --wandb \
    --wandb_project cmrnext-calib \
    --wandb_entity LGIT_calib \
    --wandb_group hercules_finetune \
    --wandb_name cmrnext_ft_lidar_G2_resize2_r20_t1.5_260216 \
    --wandb_tags hercules,finetune,lidar,r20,t1.5 \