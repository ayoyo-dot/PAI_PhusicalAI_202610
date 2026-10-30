#!/usr/bin/env bash
cd "$(dirname "$0")/.."
DATA=$1; NAME=$2; STEPS=${3:-4000}
mkdir -p outputs/logs
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e USER=runner -e LOGNAME=runner -e TORCH_HOME=/workspace/outputs/.cache/torch \
  -e PYTHONPATH=/workspace/source -v "$PWD/source:/workspace/source:ro" -v "$PWD/outputs:/workspace/outputs" libero-ros:policy \
  python3 -W ignore /workspace/source/train.py --dataset $DATA --name $NAME --steps $STEPS 2>&1 | grep -v -i "warn"
