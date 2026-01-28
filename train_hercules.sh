clear

python3 train_calibration.py \
        --savemodel /ws/data/checkpoints  \
        --data_folder_custom /ws/data/hercules  \
        --max_r 5  \
        --max_t 0.5  \
        --batch_size 1  \
        --eval_batch_size 1  \
        --epochs 1  \
        --fourier_levels -1 \
        --not_normalize_images False \
        --crop_mode 1 \
        --resize_mode 1  \
        --save_dir /ws/results  \
        --dataset hercules  \
        --data_type lg_custom \
        --image_name image_left \
        --pcl_name radar_Continental \
        --debug