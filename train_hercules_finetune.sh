python3 train_calibration.py --savemodel /ws/data/iter1/ --data_folder_custom /ws/data --custom --max_r 20 --max_t 1.5 --weights /ws/data/KITTI/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter1.tar --finetune --sensor_type radar --batch_size 4

python3 train_calibration.py --savemodel /ws/data/iter1/ --data_folder_custom /ws/data --custom --max_r 1 --max_t 0.1 --weights /ws/data/KITTI/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter1.tar --finetune --sensor_type radar --batch_size 4

python3 train_calibration.py --savemodel /ws/data/iter1/ --data_folder_custom /ws/data --custom --max_r 0.2 --max_t 0.05 --weights /ws/data/KITTI/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter1.tar --finetune --sensor_type radar --batch_size 4
