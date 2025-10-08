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
    -v $(realpath ../cmrnext):/ws/external \
    -v /media/LTDataset:/ws/data \
    wanheekim/cmrnext:latest
