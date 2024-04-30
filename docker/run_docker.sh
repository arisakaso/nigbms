#!/bin/bash

DEVICE_NUM=$1
CONTAINER_NAME="$(whoami)_nigbms_${DEVICE_NUM}"

docker run -itd --rm --shm-size=128g --gpus "device=$DEVICE_NUM" \
    --workdir="/home/$USER" \
    -v /home/$USER/nigbms:/home/$USER/nigbms \
    -v /home/$USER/.gitconfig:/home/$USER/.gitconfig \
    -v /home/$USER/.bashrc:/home/$USER/.bashrc \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/gshadow:/etc/gshadow:ro" \
    --user $(id -u):$(id -g) \
    -e TUNNEL_NAME=gear_nigbms_${DEVICE_NUM} \
    --name $CONTAINER_NAME \
    nigbms tmux

docker exec -it $CONTAINER_NAME bash ~/nigbms/docker/tunnel.sh
