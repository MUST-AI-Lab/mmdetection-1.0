#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

CONFIG=$1
PORT=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    tools/train.py $CONFIG --launcher pytorch --validate --seed 1 ${@:3}
