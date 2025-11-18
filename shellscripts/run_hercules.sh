    python3 ./../evaluate_flow_calibration.py \
    --weights /ws/data/cmrnext_calibration_weights/hercules_finetuned_r5t50/checkpoint_16_9.348.tar \
    --data_folder /ws/data/hercules/parking_lot_2/CMRNext_kissicp \
    --dataset custom \
    --num_worker 1 \
    --quantile 1.0 \
    --downsample \
    --dataset_name hercules \
    --data_id parking_lot_2  \
    --test_topic viz_aggregation \
    --max_r 5 \
    --max_t 0.5 \
    --sensor_type radar \
    --seed 40 \
    --fix_rt \
    --color_height \
    --viz_aggregation

# lidar (Aeva) 
    # python3 ./../evaluate_flow_calibration.py \
    # --weights /ws/data/cmrnext_calibration_weights/hercules_finetuned_r5t50/checkpoint_16_9.348.tar \
    # --data_folder /ws/data/hercules/parking_lot_2/CMRNext \
    # --dataset custom \
    # --num_worker 1 \
    # --quantile 1.0 \
    # --downsample \
    # --dataset_name hercules \
    # --data_id parking_lot_2  \
    # --test_topic default \
    # --max_r 5 \
    # --max_t 0.5 \
    # --sensor_type lidar \
    # --viz