#!/usr/bin/env bash
cd "$(dirname "$0")/.."
NAME=$1; N=${2:-3}; SEED=${3:-1000}
mkdir -p outputs/logs
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e PYTHONPATH=/workspace/src:/workspace/source \
  -v "$PWD/src:/workspace/src:ro" -v "$PWD/scenes:/workspace/scenes:ro" -v "$PWD/outputs:/workspace/outputs" -v "$PWD/source:/workspace/source" \
  libero-ros:sim python3 -W ignore /workspace/source/collect.py $NAME $N $SEED 2>&1 | grep -v -i "gym\|migration\|robosuite\|warn"
