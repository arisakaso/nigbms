#!/bin/bash

DEVICE_NUM=$1
CONTAINER_NAME="sohei_test_$DEVICE_NUM"

docker run -it --rm --shm-size=128g --gpus "device=$DEVICE_NUM" test /bin/bash 