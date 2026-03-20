
docker run -it --rm \
    --gpus '"device=0,1,2,3"' \
    --shm-size 64G \
    --cpus=$(nproc) \
    --ipc=host \
    --pid=host \
    -e DISPLAY=unix$DISPLAY \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(realpath .):/ws/external \
    -v /media/TrainDataset/LG_Innotek:/ws/data \
    wanheekim/cmrnext:latest
# '"device=0"'
