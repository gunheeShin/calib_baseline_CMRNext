    python3 ./../evaluate_flow_calibration.py \
    --weights /ws/data/KITTI/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter1.tar \
     /ws/data/KITTI/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter5.tar \
     /ws/data/KITTI/cmrnext_calibration_weights/cmrnext-calib-LEnc-iter6.tar \
    --data_folder /ws/data/CMRNext \
    --dataset custom \
    --num_worker 1 \
    --quantile 1.0 \
    --dataset_name hercules \
    --data_id parking_lot_1  \
    --test_topic default \
    --viz \
    --max_r 0 \
    --max_t 0 \