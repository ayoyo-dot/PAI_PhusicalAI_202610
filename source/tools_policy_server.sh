#!/usr/bin/env bash
cd "$(dirname "$0")/.."
U="--user $(id -u):$(id -g) -e HOME=/tmp -e USER=runner -e LOGNAME=runner"
if [ "$1" = "stop" ]; then docker rm -f act-server >/dev/null 2>&1 && echo "止めました"; exit 0; fi
POL=${2:-my_pilot1}; STEP=$(printf "%06d" ${3:-4000})
CK=/workspace/outputs/policies/$POL/step_$STEP
[ -f "outputs/policies/$POL/step_$STEP/model.safetensors" ] || { echo "チェックポイントがありません: outputs/policies/$POL/step_$STEP"; exit 1; }
docker network create actnet >/dev/null 2>&1
docker rm -f act-server >/dev/null 2>&1
docker run -d --name act-server --network actnet $U -e TORCH_HOME=/workspace/outputs/.cache/torch -e PYTHONPATH=/workspace/src \
  -v "$PWD/src:/workspace/src:ro" -v "$PWD/scripts:/workspace/scripts:ro" -v "$PWD/outputs:/workspace/outputs" libero-ros:policy \
  python3 /workspace/scripts/policy_server.py --ckpt $CK --port 8000 --threads 8 >/dev/null
for i in $(seq 1 60); do docker logs act-server 2>&1 | grep -q serving && { echo "起動しました: $POL / step_$STEP (http://act-server:8000)"; exit 0; }; sleep 2; done
echo "起動に失敗しました: docker logs act-server"; exit 1
