docker run -it --rm \
    --gpus all \
    --shm-size 64G \
    --cpus=$(nproc) \
    --ipc=host \
    --pid=host \
    -e DISPLAY=unix$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,source=$(realpath ../calib_baseline_CMRNext),target=/ws/external \
    --mount type=bind,source=/media/chan/LGIT_chan/Dataset/LG_Innotek/PublicDataset/,target=/ws/data \
    --mount type=bind,source=/home/chan/Documents/Project/LG_Innotek/CMRNext/output,target=/ws/output \
    cmrnext:latest
