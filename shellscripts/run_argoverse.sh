    python3 ./../evaluate_flow_calibration.py \
    --weights /ws/data/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter1.tar \
     /ws/data/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter5.tar \
     /ws/data/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter6.tar \
    --data_folder /ws/data/Argoverse/tracking_train4_v1.1/argoverse-tracking \
    --dataset argoverse \
    --dataset_name argoverse \
    --test_topic default \
    --viz \
