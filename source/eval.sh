#!/usr/bin/env bash
cd "$(dirname "$0")/.."
NAME=$1; shift
mkdir -p outputs/logs
timeout 3000 docker run --rm --network actnet --user $(id -u):$(id -g) -e HOME=/tmp -e NUMBA_CACHE_DIR=/tmp/numba -e PYTHONPATH=/workspace/src:/workspace/source \
  -v "$PWD/src:/workspace/src:ro" -v "$PWD/scenes:/workspace/scenes:ro" -v "$PWD/outputs:/workspace/outputs" -v "$PWD/source:/workspace/source:ro" \
  libero-ros:sim python3 -W ignore /workspace/source/eval_policy.py http://act-server:8000 $NAME "$@" 2>&1 | grep -v -i "gym\|migration\|robosuite\|warn"
