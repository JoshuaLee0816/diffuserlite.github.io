#!/usr/bin/env python3
"""
本機端 Inference 腳本
用於從 Hugging Face 下載 checkpoint 並執行推論

使用方式：
    python inference_local.py

需要先安裝環境：
    pip install -e .
    pip install huggingface_hub
    # MuJoCo 和 D4RL 的安裝請參考 README
"""

import os
import sys
import argparse
from pathlib import Path

# ========== 設定 ==========
REPO_ID = "JoshuaLee0816/diffuserlite-results"
ENV_NAME = "halfcheetah-medium-expert-v2"
DEVICE = "cpu"  # 本機端用 CPU
NUM_ENVS = 5    # 減少環境數量加速測試
NUM_EPISODES = 1  # 減少 episode 數量加速測試


def download_checkpoints():
    """從 Hugging Face 下載 checkpoint"""
    from huggingface_hub import hf_hub_download, list_repo_files
    import shutil

    # 本機 checkpoint 存放位置
    local_dir = Path(__file__).parent / "checkpoints" / ENV_NAME
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"從 Hugging Face 下載 checkpoint...")
    print(f"Repo: {REPO_ID}")
    print(f"目標目錄: {local_dir}")

    try:
        files = list_repo_files(REPO_ID)
        # 只下載 latest checkpoint
        checkpoint_files = [f for f in files if ENV_NAME in f and 'latest' in f and f.endswith('.pt')]

        if not checkpoint_files:
            # 如果沒有 latest，下載所有 checkpoint
            checkpoint_files = [f for f in files if ENV_NAME in f and f.endswith('.pt')]

        if not checkpoint_files:
            print(f"Hugging Face 上沒有找到 {ENV_NAME} 的 checkpoint")
            return None

        print(f"找到 {len(checkpoint_files)} 個 checkpoint 檔案")

        for f in checkpoint_files:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=f)
            filename = os.path.basename(f)
            dest_path = local_dir / filename
            shutil.copy(local_path, dest_path)
            print(f"  {filename}")

        print(f"\n已下載到: {local_dir}")
        return local_dir

    except Exception as e:
        print(f"下載失敗: {e}")
        return None


def setup_mujoco_shim():
    """設定 mujoco_py shim（相容層）"""
    import types
    import mujoco
    import numpy as np

    shim = types.ModuleType('mujoco_py')

    class ShimModel:
        def __init__(self, m):
            self._m = m
        @property
        def actuator_ctrlrange(self):
            return self._m.actuator_ctrlrange.copy()
        @property
        def nq(self): return self._m.nq
        @property
        def nv(self): return self._m.nv
        def __getattr__(self, name):
            return getattr(self._m, name)

    class ShimData:
        def __init__(self, d):
            self._d = d
        @property
        def qpos(self): return self._d.qpos
        @qpos.setter
        def qpos(self, v): self._d.qpos[:] = v
        @property
        def qvel(self): return self._d.qvel
        @qvel.setter
        def qvel(self, v): self._d.qvel[:] = v
        @property
        def ctrl(self): return self._d.ctrl
        @ctrl.setter
        def ctrl(self, v): self._d.ctrl[:] = v
        def __getattr__(self, name):
            return getattr(self._d, name)

    class ShimSim:
        def __init__(self, model):
            self._m = model._m
            self._d = mujoco.MjData(self._m)
            self.model = model
            self.data = ShimData(self._d)
        def step(self):
            mujoco.mj_step(self._m, self._d)
        def forward(self):
            mujoco.mj_forward(self._m, self._d)
        def get_state(self):
            return type('S',(),{'time':self._d.time,'qpos':self._d.qpos.copy(),'qvel':self._d.qvel.copy(),'act':np.array([]),'udd_state':{}})()
        def set_state(self, s):
            self._d.time = s.time
            self._d.qpos[:] = s.qpos
            self._d.qvel[:] = s.qvel
            mujoco.mj_forward(self._m, self._d)

    class MjViewer:
        def __init__(self, sim): pass
        def render(self): pass

    shim.load_model_from_path = lambda p: ShimModel(mujoco.MjModel.from_xml_path(p))
    shim.MjSim = lambda m: ShimSim(m)
    shim.MjViewer = MjViewer
    shim.MujocoException = Exception
    shim.ignore_mujoco_warnings = type('ctx',(),{'__enter__':lambda s:None,'__exit__':lambda s,*a:None})
    shim.__path__ = []

    for sub in ['cymj','builder','generated','generated.const']:
        sys.modules[f'mujoco_py.{sub}'] = types.ModuleType(f'mujoco_py.{sub}')
    sys.modules['mujoco_py'] = shim

    print("mujoco_py shim 已設定")


def run_inference(checkpoint_dir: Path):
    """執行推論"""
    import torch
    import numpy as np
    import gym
    import d4rl

    from cleandiffuser.dataset.d4rl_mujoco_dataset import MultiHorizonD4RLMuJoCoDataset
    from cleandiffuser.diffusion import ContinuousRectifiedFlow
    from cleandiffuser.invdynamic import FancyMlpInvDynamic
    from cleandiffuser.nn_condition import MLPCondition
    from cleandiffuser.nn_diffusion import DiT1d
    from cleandiffuser.utils import DD_RETURN_SCALE, set_seed

    set_seed(42)

    print(f"\n{'='*50}")
    print(f"執行 Inference")
    print(f"{'='*50}")
    print(f"環境: {ENV_NAME}")
    print(f"裝置: {DEVICE}")
    print(f"環境數量: {NUM_ENVS}")
    print(f"Episode 數量: {NUM_EPISODES}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"{'='*50}\n")

    # ========== 環境設定 ==========
    env = gym.make(ENV_NAME)
    scale = DD_RETURN_SCALE[ENV_NAME]

    # 取得 dataset 來建立 normalizer
    dataset = MultiHorizonD4RLMuJoCoDataset(
        env.get_dataset(),
        horizons=[5, 5, 9],  # 預設的 temporal horizons
        terminal_penalty=-100,
        discount=0.997
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    normalizer = dataset.get_normalizer()

    # ========== 模型設定 ==========
    planning_horizons = [5, 5, 9]
    n_levels = len(planning_horizons)

    # 建立模型架構
    fix_masks = [torch.zeros((h, obs_dim)) for h in planning_horizons]
    loss_weights = [torch.ones((h, obs_dim)) for h in planning_horizons]
    for i in range(n_levels):
        fix_idx = 0 if i == 0 else [0, -1]
        fix_masks[i][fix_idx, :] = 1.
        loss_weights[i][1, :] = 10.0  # next_obs_loss_weight

    nn_diffusions = [
        DiT1d(obs_dim, emb_dim=256, d_model=256, n_heads=8, depth=2, timestep_emb_type="fourier")
        for _ in range(n_levels)
    ]
    nn_conditions = [
        MLPCondition(1, 256, hidden_dims=[256])
        for _ in range(n_levels)
    ]

    diffusions = [
        ContinuousRectifiedFlow(
            nn_diffusions[i], nn_conditions[i], fix_masks[i], loss_weights[i],
            ema_rate=0.9999, device=DEVICE)
        for i in range(n_levels)
    ]

    invdyn = FancyMlpInvDynamic(obs_dim, act_dim, 256, torch.nn.Tanh(), add_dropout=True, device=DEVICE)

    # ========== 載入 checkpoint ==========
    print("載入 checkpoint...")
    for i in range(n_levels):
        ckpt_path = checkpoint_dir / f'diffusion{i}_ckpt_latest.pt'
        if ckpt_path.exists():
            diffusions[i].load(str(ckpt_path))
            print(f"  diffusion{i}: {ckpt_path.name}")
        else:
            print(f"  找不到: {ckpt_path}")
            return None
        diffusions[i].eval()

    invdyn_path = checkpoint_dir / 'invdyn_ckpt_latest.pt'
    if invdyn_path.exists():
        invdyn.load(str(invdyn_path))
        print(f"  invdyn: {invdyn_path.name}")
    else:
        print(f"  找不到: {invdyn_path}")
        return None
    invdyn.eval()

    print("模型載入完成\n")

    # ========== Inference ==========
    # 從 task config 取得 target_return
    target_return = 1.0  # R1 的 target_return（已正規化）
    w_cfg = 1.2  # R1 的 w_cfg

    env_eval = gym.vector.make(ENV_NAME, NUM_ENVS)
    episode_rewards = []

    priors = [torch.zeros((NUM_ENVS, planning_horizons[i], obs_dim), device=DEVICE)
              for i in range(n_levels)]
    condition = torch.ones((NUM_ENVS, 1), device=DEVICE) * target_return

    for ep in range(NUM_EPISODES):
        print(f"Episode {ep+1}/{NUM_EPISODES}")
        obs, ep_reward, cum_done, t = env_eval.reset(), 0., None, 0

        while t < 1000:
            obs_tensor = torch.tensor(normalizer.normalize(obs), device=DEVICE, dtype=torch.float32)

            priors[0][:, 0] = obs_tensor
            for j in range(n_levels):
                traj, _ = diffusions[j].sample(
                    priors[j],
                    n_samples=NUM_ENVS,
                    sample_steps=3,  # R1 用 3 步
                    use_ema=True,
                    condition_cfg=condition,
                    w_cfg=w_cfg,
                    temperature=0.5,
                    sample_step_schedule="quad_continuous"
                )
                if j < n_levels - 1:
                    priors[j + 1][:, [0, -1]] = traj[:, [0, 1]]

            with torch.no_grad():
                act = invdyn(traj[:, 0], traj[:, 1]).cpu().numpy()

            obs, rew, done, info = env_eval.step(act)

            t += 1
            if cum_done is None:
                cum_done = done
            else:
                cum_done = np.logical_or(cum_done, done)

            ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew

            if t % 100 == 0:
                print(f"  [t={t}] reward so far: {ep_reward.mean():.2f}")

            if np.all(cum_done):
                break

        episode_rewards.append(ep_reward)
        print(f"  Episode {ep+1} done, reward: {ep_reward.mean():.2f}")

    # ========== 計算 Normalized Score ==========
    episode_rewards = np.array(episode_rewards)
    normalized_scores = [[env.get_normalized_score(r) for r in ep] for ep in episode_rewards]
    normalized_scores = np.array(normalized_scores) * 100  # 轉成百分比

    mean_score = normalized_scores.mean()
    std_score = normalized_scores.std()

    print(f"\n{'='*50}")
    print(f"結果")
    print(f"{'='*50}")
    print(f"Normalized Score: {mean_score:.2f} +/- {std_score:.2f}")
    print(f"論文 R1 結果: 91.9")
    print(f"{'='*50}")

    return {
        'env_name': ENV_NAME,
        'normalized_score_mean': mean_score,
        'normalized_score_std': std_score,
        'episode_rewards': episode_rewards.tolist(),
        'normalized_scores': normalized_scores.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='本機端 Inference')
    parser.add_argument('--skip-download', action='store_true', help='跳過下載，使用本地 checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='指定 checkpoint 目錄')
    args = parser.parse_args()

    # 設定 mujoco_py shim
    setup_mujoco_shim()

    # 下載或使用本地 checkpoint
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    elif args.skip_download:
        checkpoint_dir = Path(__file__).parent / "checkpoints" / ENV_NAME
    else:
        checkpoint_dir = download_checkpoints()

    if checkpoint_dir is None or not checkpoint_dir.exists():
        print("無法找到 checkpoint，請先下載或指定路徑")
        sys.exit(1)

    # 執行推論
    results = run_inference(checkpoint_dir)

    if results:
        # 儲存結果
        import json
        results_file = Path(__file__).parent / "inference_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n結果已儲存到: {results_file}")


if __name__ == "__main__":
    main()
