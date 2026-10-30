# LIBERO × ACT: ロボットアームが、物をつかんで、持ち上げる

講座「PhysicalAI応用編1」の最終課題。シミュレーション(LIBERO)の中で、ロボットアーム(Franka Panda)が、机の上の物をつかんで、持ち上げる。
物の近くまでは、シミュレータの真値を使った自作の制御で手を動かし、つかんで持ち上げる動作は、模倣学習(ACT)で行う。
GPU なし(CPU のみ、Docker、`MUJOCO_GL=osmesa`)で、学習も評価も動かしている。

## ディレクトリ構成

```
source/        　自分で実装した部分。
  approach.py     手先の位置制御(P 制御、上 → 横 → 下の経路)
  expert.py       手本(つかんで持ち上げる)
  collect.py      デモ収集(物の真値 + 最大 3cm のずれの真上へ寄ってから、記録を始める)
  train.py        ACT の学習(lerobot)
  handoff.py      切り替え判定と、ポリシーを動かすループ
  provided.py     環境の部品(AI 作): 場面の作成、手先の位置・向き、物体の表(把持の高さ・幅)
  eval_policy.py  評価の道具(AI 作): 成功率、Wilson 区間、切り替え判定の TP/FP/FN/TN、動画

# 以下はAIが構築
src/libero_ros/ 環境まわりの部品: 手先の制御と座標(control)、つかみ(grasp)、配置(layout)、デモの記録(demos)、動画(video)
scripts/        policy_server.py: 学習した ACT を HTTP で呼べるようにするサーバー
docker/         イメージの定義(sim = LIBERO 入り、policy = lerobot 入り)
scenes/         シーン定義(.bddl)
```

## 動かし方

前提: Docker(GPU 不要)。

```bash
docker build -f docker/Dockerfile --target sim -t libero-ros:sim .       # LIBERO 入り(MuJoCo は CPU 描画: OSMesa)
docker build -f docker/Dockerfile.policy -t libero-ros:policy .          # lerobot 入り(ACT の学習・推論用)
mkdir -p outputs                                                          # 出力先(git 管理外)

bash source/collect.sh my_demos 150 1000                 # デモを 150 本集める(outputs/demos/my_demos)
bash source/train.sh my_demos my_pilot1 4000             # ACT を学習(CPU、約 32 分)
bash source/tools_policy_server.sh start my_pilot1 4000  # ポリシーのサーバーを起動
bash source/eval.sh final --n 12 --delta 3               # 評価(物体ごとに 12 回、ずれ最大 3cm)
bash source/tools_policy_server.sh stop
```
