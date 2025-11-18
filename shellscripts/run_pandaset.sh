    python3 ./../evaluate_flow_calibration.py \
    --weights /ws/data/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter1.tar \
     /ws/data/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter5.tar \
     /ws/data/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter6.tar \
    --data_folder /ws/data/Pandaset/PandaSet \
    --dataset pandaset \
    --dataset_name pandaset \
    --test_topic default \
    --viz \
