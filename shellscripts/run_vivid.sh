    python3 ./../evaluate_flow_calibration.py \
    --weights /ws/data/cmrnext_calibration_weights/vivid_checkpoints/iter1/remove/checkpoint_11_29.705.tar \
    --data_folder /ws/data/VIVID/campus_day1/CMRNext \
    --dataset custom \
    --num_worker 1 \
    --quantile 1.0 \
    --downsample \
    --dataset_name vivid \
    --data_id campus_day1 \
    --test_topic default \
    --max_r 20 \
    --max_t 1.5 \
    --sensor_type lidar\
    --seed 42 \
    

    #      /ws/data/cmrnext_calibration_weights/vivid_checkpoints/iter5/remove/checkpoint_44_4.419.tar \
    #  /ws/data/cmrnext_calibration_weights/vivid_checkpoints/iter6/remove/checkpoint_24_2.275.tar \