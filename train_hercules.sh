clear

python3 train_calibration.py --savemodel /ws/data/checkpoints --data_folder_custom /ws/data/hercules --save_dir /ws/data/cmrnext_results --dataset hercules --max_r 20 --max_t 0.5 --batch_size 1 --eval_batch_size 1 --sensor_type lidar_Aeva --downsize --epochs 1 --debug --crop_mode 1