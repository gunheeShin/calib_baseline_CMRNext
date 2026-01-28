
python3 train_calibration.py \
    --savemodel /ws/data/checkpoints \
    --data_folder_custom /ws/data \
    --max_r 5 \
    --max_t 0.5 \
    --weights /ws/data/checkpoints/cmrnext_kittiArgoversePandasetHercules_5_0.5_251110.tar \
    --finetune \
    --epochs 25 \
    --BASE_LEARNING_RATE 3e-5 \
    --batch_size 4 \
    --crop_mode 1 \
    --resize_mode 1  \
    --save_dir /ws/output  \
    --save_model_name cmrnext_kittiArgoversePandasetHerculesRadar_5_0.5_251110_woDownsize \
    --dataset hercules  \
    --data_type lg_custom \
    --image_name image_left \
    --pcl_name radar_Continental \
