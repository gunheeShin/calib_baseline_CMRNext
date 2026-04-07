python3 evaluate_flow_calibration.py \
	    --weights /ws/data/Checkpoint/cmrnext_lg_full_lidar_G3_r5_t0.5_260405_a87vkeno/cmrnext_LgInnotek_5_0.5_260405_148_1.501.tar \
	    --data_folder /ws/data/CustomDataset \
	    --quantile 1.0 \
	    --max_t 0.5 \
	    --max_r 5 \
		--dataset lg_innotek \
	    --test_topic default \
		--data_type lg_custom \
	    --image_name image_Cam0 \
	    --pcl_name lidar_Hesai \
		--fix_error False \
		--resize_mode 2 \
		--img_shape 536 960 \

