python3 train_calibration.py --savemodel /ws/data/iter1/ --data_folder_custom /ws/data/LG_Innotek/CustomDataset/VIVID --custom --max_r 20 --max_t 1.5 --weights /ws/data/LG_Innotek/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter1.tar --finetune --epochs 50 --BASE_LEARNING_RATE 1e-5

python3 train_calibration.py --savemodel /ws/data/iter5/ --data_folder_custom /ws/data/LG_Innotek/CustomDataset/VIVID --custom --max_r 1 --max_t 0.1 --weights /ws/data/LG_Innotek/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter5.tar --finetune --epochs 50 --BASE_LEARNING_RATE 1e-5

python3 train_calibration.py --savemodel /ws/data/iter6/ --data_folder_custom /ws/data/LG_Innotek/CustomDataset/VIVID --custom --max_r 0.2 --max_t 0.05 --weights /ws/data/LG_Innotek/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter6.tar --finetune --epochs 50 --BASE_LEARNING_RATE 1e-5
