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
    --max_r 5 \
    --max_t 0.5 \
    --epochs 150 \
    --BASE_LEARNING_RATE 3e-4 \
    --batch_size 4 \
    --crop_mode 1 \
    --resize_mode 2  \
    --save_dir /ws/output  \
    --save_model_name cmrnext_HerculesRadar_5_0.5_260320_woDownsize \
    --dataset hercules  \
    --data_type lg_custom \
    --image_name image_left \
    --pcl_name radar_Continental \
    --num_worker 4 \
    --wandb \
    --wandb_project cmrnext-calib \
    --wandb_entity LGIT_calib \
    --wandb_group hercules_fulltrain \
    --wandb_name cmrnext_full_radar_G4_r5_t0.5_260320_woDownsize_trainOPT \
    --wandb_tags hercules,fulltrain,radar,r5,t0.5 \