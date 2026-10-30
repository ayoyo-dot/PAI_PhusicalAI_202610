import argparse
import glob
import json
import os
import time

import numpy as np

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy 

IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)


def load_episodes(dataset):
    eps = []
    for f in sorted(glob.glob(f"/workspace/outputs/demos/{dataset}/success/*.npz")):
        z = np.load(f, allow_pickle=False)
        eps.append({k: z[k] for k in ("agentview", "wrist", "state", "action")})
    return eps


def to_torch_batch(raw):
    import torch

    def img(x):
        x = x.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return torch.from_numpy((x - IMNET_MEAN) / IMNET_STD)
    return {"observation.images.agentview": img(raw["agentview"]), "observation.images.wrist": img(raw["wrist"]),
            "observation.state": torch.from_numpy(raw["state"].astype(np.float32)),
            "action": torch.from_numpy(raw["action"].astype(np.float32)),
            "action_is_pad": torch.from_numpy(raw["is_pad"])}


def forward_loss(policy, batch):
    o = policy.forward(batch)
    return o[0] if isinstance(o, tuple) else o["loss"]


def compute_stats(episodes):
    LOW_THR = 1e-3

    state_dict = dict()

    state_list = []
    action_list = []
    for episode in episodes: 
        state_list.append(episode["state"])      
        action_list.append(episode["action"])

    state_concatenate_list = np.concatenate(state_list,axis=0)
    action_concatenate_list = np.concatenate(action_list,axis=0)
    state_dict["state_mean"] = np.mean(state_concatenate_list,axis=0).tolist()
    state_dict["state_std"]  = np.clip(np.std(state_concatenate_list,axis=0),a_min = LOW_THR,a_max = None).tolist()
    state_dict["action_mean"]  = np.mean(action_concatenate_list,axis=0).tolist()
    state_dict["action_std"]  = np.clip(np.std(action_concatenate_list,axis=0),a_min=LOW_THR,a_max = None).tolist()  

    
    return(state_dict)

def sample_batch(episodes, stats, batch_size, chunk, rng):

    lens = []
    for episode in episodes:
        lens.append(len(episode["action"]))

    lens = np.array(lens)
    episode_num = rng.choice(len(lens),size=batch_size,p=lens/lens.sum()) 
    ag_list, wr_list, st_list, ac_list, pad_list = [], [], [], [], [] 
    for i in episode_num :
        e = episodes[i]
        T=len(e["action"])
        t = rng.integers(0,T) 
        idx = np.minimum(np.arange(t, t+chunk), T-1) 
        ag_list.append( e["agentview"][t] )
  
        wr_list.append(e["wrist"][t])
        st_list.append( e["state"][t] )
        ac_list.append(e["action"][idx])
        pad_list.append(np.arange(t,t+chunk) >= T)  
    ag_list_stack = np.stack(ag_list )
    wr_list_stack = np.stack(wr_list)
    st_list_stack = np.stack(st_list)
    ac_list_stack = np.stack(ac_list)
    pad_list_stack = np.stack( pad_list )

    state = (st_list_stack - np.array(stats["state_mean"]))/np.array(stats["state_std"])
    action = (ac_list_stack - np.array(stats["action_mean"]))/np.array(stats["action_std"])

    return(dict(
        agentview = ag_list_stack,
        wrist = wr_list_stack,
        state = state,
        action = action,
        is_pad = pad_list_stack
    
    ))


def make_policy(state_dim, image_hw, chunk=30, n_action_steps=10, lr=1e-4, lr_backbone=1e-5, pretrained=True):

    f_state = PolicyFeature(type=FeatureType.STATE, shape=(state_dim,))
    f_agentview = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_hw))
    f_wrist = PolicyFeature(type=FeatureType.VISUAL, shape=(3, *image_hw))
    f_action = PolicyFeature(type=FeatureType.ACTION,shape=(7,))

    input_features = {
        "observation.state": f_state,
        "observation.images.agentview":f_agentview,
        "observation.images.wrist":f_wrist,
    }
    output_features = {"action":f_action}
    p = None
    if pretrained:
        p = "ResNet18_Weights.IMAGENET1K_V1"
    cfg = ACTConfig(
        input_features = input_features,
        output_features = output_features,
        chunk_size = chunk,
        n_action_steps = n_action_steps,
        use_vae = False,
        device = "cpu",
        pretrained_backbone_weights = p,
        optimizer_lr = lr,
        optimizer_lr_backbone = lr_backbone
        )

    policy = ACTPolicy(cfg)
    policy.train()


    return(policy,cfg)


def train(dataset, name, steps, batch=8, chunk=30, n_action_steps=10, lr=1e-4, lr_backbone=1e-5, save_every=500, threads=16, seed=0):

    import torch

    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    episodes = load_episodes(dataset) 
    state_dim = episodes[0]["state"].shape[1]
    image_hw = episodes[0]["agentview"].shape[1:3]
    len_all = 0
    for e in episodes:
        len_all += len(e["action"])
    print(f"num of episodes : {len(episodes)} , 全ステップ数 : {len_all}" )
    
    stats = compute_stats(episodes)

    json_output_dir = f'/workspace/outputs/policies/{name}/'
    os.makedirs(json_output_dir,exist_ok=True)
    with open(f'{json_output_dir}stats.json', 'w', encoding='utf-8') as f:
        stats["mask_state_idx"] = []
        json.dump(stats,f)

    policy,cfg = make_policy(state_dim, image_hw,chunk=chunk,n_action_steps=n_action_steps,lr = lr,lr_backbone=lr_backbone)
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=lr, weight_decay=cfg.optimizer_weight_decay)

    loss_list = []
    start_time = time.time()
    for i in range(steps):
        raw = sample_batch(episodes, stats, batch, chunk, rng)
        torch_batch = to_torch_batch(raw)
        loss = forward_loss(policy, torch_batch)
        opt.zero_grad() 
        loss.backward() 
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        opt.step()
        loss_list.append(float(loss))
        if (i+1) % 50 == 0:
            now = time.time()
            print(f"loss : {np.mean(loss_list)} , time : {now - start_time}",flush=True)
            loss_list = []
        
        if (i+1) % save_every == 0:
            step_output_dir=f"{json_output_dir}/step_{i+1:06d}/"
            os.makedirs(step_output_dir,exist_ok=True)
            policy.save_pretrained(step_output_dir)
            with open(f'{step_output_dir}stats.json', 'w', encoding='utf-8') as f:
                stats["mask_state_idx"] = []
                json.dump(stats,f)

    step_output_dir=f"{json_output_dir}/step_{i+1:06d}/"
    os.makedirs(step_output_dir,exist_ok=True)
    policy.save_pretrained(step_output_dir)
    with open(f'{step_output_dir}stats.json', 'w', encoding='utf-8') as f:
        stats["mask_state_idx"] = []
        json.dump(stats,f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=30)
    ap.add_argument("--n-action-steps", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.dataset, a.name, a.steps, a.batch, a.chunk, a.n_action_steps, save_every=a.save_every, threads=a.threads, seed=a.seed)


if __name__ == "__main__":
    main()
