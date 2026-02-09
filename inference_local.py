#!/usr/bin/env python3
"""
本機端 Inference 腳本
從 Hugging Face 下載所有 checkpoint 並執行推論
輸出結果表格和合併影片

使用方式：
    python inference_local.py                           # 預設 halfcheetah
    python inference_local.py --env walker2d-medium-expert-v2
    python inference_local.py --env hopper-medium-expert-v2
    python inference_local.py --skip-download           # 跳過下載

需要先安裝環境：
    pip install -e .
    pip install huggingface_hub imageio[ffmpeg]
"""

import os
import sys
import re
import argparse
from pathlib import Path

# ========== 設定 ==========
REPO_ID = "JoshuaLee0816/diffuserlite-results"
DEVICE = "cpu"
NUM_ENVS = 1
NUM_EPISODES = 1
RECORD_VIDEO = True
MAX_STEPS_PER_VIDEO = 200


def get_output_dir(env_name):
    """取得輸出目錄（依環境分類）"""
    output_dir = Path(__file__).parent / "results" / "inference" / env_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def download_all_checkpoints(env_name):
    """從 Hugging Face 下載所有 checkpoint"""
    from huggingface_hub import hf_hub_download, list_repo_files
    import shutil

    local_dir = Path(__file__).parent / "checkpoints" / env_name
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"從 Hugging Face 下載 checkpoint...")
    print(f"Repo: {REPO_ID}")
    print(f"環境: {env_name}")
    print(f"目標目錄: {local_dir}")

    try:
        files = list_repo_files(REPO_ID)
        # 下載該環境的 checkpoint
        checkpoint_files = [f for f in files if f.endswith('.pt') and env_name in f]

        if not checkpoint_files:
            print(f"Hugging Face 上沒有找到 {env_name} 的 checkpoint")
            return None, []

        print(f"找到 {len(checkpoint_files)} 個 checkpoint 檔案")

        for f in checkpoint_files:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=f)
            filename = os.path.basename(f)
            dest_path = local_dir / filename
            if not dest_path.exists():
                shutil.copy(local_path, dest_path)
            print(f"  {filename}")

        # 找出所有可用的步數
        steps = set()
        for f in os.listdir(local_dir):
            match = re.search(r'_ckpt_(\d+)\.pt', f)
            if match:
                steps.add(int(match.group(1)))

        if (local_dir / 'diffusion0_ckpt_latest.pt').exists():
            steps.add('latest')

        steps = sorted([s for s in steps if isinstance(s, int)]) + (['latest'] if 'latest' in steps else [])

        print(f"\n可用的 checkpoint 步數: {steps}")
        return local_dir, steps

    except Exception as e:
        print(f"下載失敗: {e}")
        return None, []


def setup_mujoco_shim():
    """設定 mujoco_py shim"""
    import types
    import mujoco
    import numpy as np

    shim = types.ModuleType('mujoco_py')

    class ShimModel:
        def __init__(self, m):
            self._m = m
            self._camera_name2id = {}
            for i in range(m.ncam):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
                if name:
                    self._camera_name2id[name] = i
        def camera_name2id(self, name):
            return self._camera_name2id.get(name, -1)
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
        def reset(self):
            mujoco.mj_resetData(self._m, self._d)
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

    class MjRenderContextOffscreen:
        def __init__(self, sim, device_id=-1):
            self._sim = sim
            self._m = sim._m
            self._d = sim._d
            self._width = 640
            self._height = 480
            self._renderer = mujoco.Renderer(self._m, height=self._height, width=self._width)
            self.cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self.cam)
            self.cam.distance = 6.0
            self.cam.elevation = -20
            self.cam.azimuth = 90
        def render(self, width, height, camera_id=None):
            mujoco.mj_forward(self._m, self._d)
            if len(self._d.qpos) > 0:
                self.cam.lookat[0] = self._d.qpos[0]
                self.cam.lookat[1] = 0
                self.cam.lookat[2] = 0.5
            if camera_id is not None and camera_id >= 0:
                self._renderer.update_scene(self._d, camera=camera_id)
            else:
                self._renderer.update_scene(self._d, camera=self.cam)
            return self._renderer.render()
        def read_pixels(self, width, height, depth=False):
            img = self.render(width, height)
            if depth:
                return img, np.zeros((height, width), dtype=np.float32)
            return img

    from collections import namedtuple
    MjSimState = namedtuple('MjSimState', ['time', 'qpos', 'qvel', 'act', 'udd_state'])

    shim.load_model_from_path = lambda p: ShimModel(mujoco.MjModel.from_xml_path(p))
    shim.MjSim = lambda m: ShimSim(m)
    shim.MjViewer = MjViewer
    shim.MjRenderContextOffscreen = MjRenderContextOffscreen
    shim.MujocoException = Exception
    shim.MjSimState = MjSimState
    shim.ignore_mujoco_warnings = type('ctx',(),{'__enter__':lambda s:None,'__exit__':lambda s,*a:None})
    shim.__path__ = []

    for sub in ['cymj','builder','generated','generated.const']:
        sys.modules[f'mujoco_py.{sub}'] = types.ModuleType(f'mujoco_py.{sub}')
    sys.modules['mujoco_py'] = shim

    print("mujoco_py shim 已設定")


def run_inference_single(checkpoint_dir, step, env_name, env, dataset, normalizer,
                         diffusions, invdyn, obs_dim, planning_horizons, n_levels):
    """對單一 checkpoint 執行推論"""
    import torch
    import numpy as np
    import gym

    step_str = step if step == 'latest' else str(step)

    for i in range(n_levels):
        ckpt_path = checkpoint_dir / f'diffusion{i}_ckpt_{step_str}.pt'
        if not ckpt_path.exists():
            return None, []
        diffusions[i].load(str(ckpt_path))
        diffusions[i].eval()

    invdyn_path = checkpoint_dir / f'invdyn_ckpt_{step_str}.pt'
    if not invdyn_path.exists():
        return None, []
    invdyn.load(str(invdyn_path))
    invdyn.eval()

    target_return = 1.0
    w_cfg = 1.2
    frames = []

    env_eval = gym.vector.make(env_name, NUM_ENVS, asynchronous=False)

    priors = [torch.zeros((NUM_ENVS, planning_horizons[i], obs_dim), device=DEVICE)
              for i in range(n_levels)]
    condition = torch.ones((NUM_ENVS, 1), device=DEVICE) * target_return

    obs = env_eval.reset()
    ep_reward = np.zeros(NUM_ENVS)
    cum_done, t = None, 0

    while t < 1000:
        obs_tensor = torch.tensor(normalizer.normalize(obs), device=DEVICE, dtype=torch.float32)

        priors[0][:, 0] = obs_tensor
        for j in range(n_levels):
            traj, _ = diffusions[j].sample(
                priors[j],
                n_samples=NUM_ENVS,
                sample_steps=3,
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

        if RECORD_VIDEO and t < MAX_STEPS_PER_VIDEO:
            try:
                inner_env = env_eval.envs[0]
                unwrapped = inner_env.unwrapped if hasattr(inner_env, 'unwrapped') else inner_env

                if hasattr(unwrapped, 'sim') and hasattr(unwrapped.sim, 'data'):
                    x_pos = unwrapped.sim.data.qpos[0]
                    if hasattr(unwrapped, '_get_viewer'):
                        viewer = unwrapped._get_viewer('rgb_array')
                        if viewer is not None and hasattr(viewer, 'cam'):
                            viewer.cam.lookat[0] = x_pos
                            viewer.cam.lookat[1] = 0
                            viewer.cam.lookat[2] = 0.5
                            viewer.cam.distance = 5.0
                            viewer.cam.elevation = -20
                            viewer.cam.azimuth = 90

                frame = inner_env.render(mode='rgb_array')
                if frame is not None:
                    frame = np.flipud(frame)
                    frames.append(frame)
            except Exception as e:
                if t == 0:
                    print(f"  錄影設定: {e}")
                try:
                    frame = inner_env.render(mode='rgb_array')
                    if frame is not None:
                        frame = np.flipud(frame)
                        frames.append(frame)
                except:
                    pass

        t += 1
        if cum_done is None:
            cum_done = done
        else:
            cum_done = np.logical_or(cum_done, done)

        ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew

        if np.all(cum_done):
            break

    env_eval.close()

    normalized_score = env.get_normalized_score(ep_reward[0]) * 100

    return normalized_score, frames


def add_text_to_frame(frame, text, position=(10, 30)):
    """在 frame 上加上文字標籤"""
    import numpy as np
    try:
        from PIL import Image, ImageDraw, ImageFont
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
        except:
            font = ImageFont.load_default()
        draw.rectangle([position[0]-5, position[1]-5, position[0]+200, position[1]+30], fill='black')
        draw.text(position, text, font=font, fill='white')
        return np.array(img)
    except ImportError:
        return frame


def main():
    import json

    parser = argparse.ArgumentParser(description='本機端 Inference - 測試所有 checkpoint')
    parser.add_argument('--env', type=str, default='halfcheetah-medium-expert-v2',
                        choices=['halfcheetah-medium-expert-v2', 'walker2d-medium-expert-v2', 'hopper-medium-expert-v2'],
                        help='環境名稱')
    parser.add_argument('--skip-download', action='store_true', help='跳過下載，使用本地 checkpoint')
    args = parser.parse_args()

    env_name = args.env
    output_dir = get_output_dir(env_name)

    print(f"\n{'='*60}")
    print(f"DiffuserLite Inference")
    print(f"環境: {env_name}")
    print(f"輸出目錄: {output_dir}")
    print(f"{'='*60}\n")

    setup_mujoco_shim()

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

    # 下載 checkpoint
    if args.skip_download:
        checkpoint_dir = Path(__file__).parent / "checkpoints" / env_name
        steps = set()
        if checkpoint_dir.exists():
            for f in os.listdir(checkpoint_dir):
                match = re.search(r'_ckpt_(\d+)\.pt', f)
                if match:
                    steps.add(int(match.group(1)))
            if (checkpoint_dir / 'diffusion0_ckpt_latest.pt').exists():
                steps.add('latest')
        steps = sorted([s for s in steps if isinstance(s, int)]) + (['latest'] if 'latest' in steps else [])
    else:
        checkpoint_dir, steps = download_all_checkpoints(env_name)

    if checkpoint_dir is None or not checkpoint_dir.exists():
        print("無法找到 checkpoint")
        sys.exit(1)

    if not steps:
        print("沒有找到任何 checkpoint")
        sys.exit(1)

    numeric_steps = [s for s in steps if isinstance(s, int)]
    print(f"\n將測試的 checkpoint 步數: {numeric_steps}")

    set_seed(42)

    env = gym.make(env_name)
    scale = DD_RETURN_SCALE[env_name]

    dataset = MultiHorizonD4RLMuJoCoDataset(
        env.get_dataset(),
        horizons=[5, 5, 9],
        terminal_penalty=-100,
        discount=0.997
    )
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim
    normalizer = dataset.get_normalizer()

    planning_horizons = [5, 5, 9]
    n_levels = len(planning_horizons)

    fix_masks = [torch.zeros((h, obs_dim)) for h in planning_horizons]
    loss_weights = [torch.ones((h, obs_dim)) for h in planning_horizons]
    for i in range(n_levels):
        fix_idx = 0 if i == 0 else [0, -1]
        fix_masks[i][fix_idx, :] = 1.
        loss_weights[i][1, :] = 10.0

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

    # 執行推論
    results = []
    all_frames = []

    print(f"\n{'='*60}")
    print(f"開始測試各階段 checkpoint")
    print(f"{'='*60}\n")

    for step in numeric_steps:
        print(f"測試 step {step}...")
        score, frames = run_inference_single(
            checkpoint_dir, step, env_name, env, dataset, normalizer,
            diffusions, invdyn, obs_dim, planning_horizons, n_levels
        )

        if score is not None:
            results.append({'step': step, 'score': score})
            print(f"  Step {step}: Score = {score:.2f}")

            if frames:
                labeled_frames = [add_text_to_frame(f, f"Step: {step}") for f in frames]
                all_frames.extend(labeled_frames)
        else:
            print(f"  Step {step}: 無法載入 checkpoint")

    # 輸出結果表格
    print(f"\n{'='*60}")
    print(f"學習曲線結果 ({env_name})")
    print(f"{'='*60}")
    print(f"{'Step':>10} | {'Normalized Score':>20}")
    print(f"{'-'*10}-+-{'-'*20}")
    for r in results:
        print(f"{r['step']:>10} | {r['score']:>20.2f}")
    print(f"{'='*60}")

    # 儲存結果到環境專屬目錄
    results_file = output_dir / "learning_curve_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n結果已儲存到: {results_file}")

    # 儲存合併影片
    if RECORD_VIDEO and all_frames:
        import imageio
        video_path = output_dir / "learning_progress.mp4"
        print(f"\n儲存合併影片到: {video_path}")
        print(f"共 {len(all_frames)} 幀")
        imageio.mimsave(str(video_path), all_frames, fps=30)
        print(f"影片儲存完成！")

    # 畫學習曲線圖（不顯示論文分數）
    try:
        import matplotlib.pyplot as plt

        steps_plot = [r['step'] for r in results]
        scores_plot = [r['score'] for r in results]

        plt.figure(figsize=(10, 6))
        plt.plot(steps_plot, scores_plot, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Training Steps', fontsize=12)
        plt.ylabel('Normalized Score', fontsize=12)
        plt.title(f'DiffuserLite Learning Curve ({env_name})', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = output_dir / "learning_curve.png"
        plt.savefig(fig_path, dpi=150)
        print(f"學習曲線圖已儲存到: {fig_path}")
        plt.close()
    except ImportError:
        print("無法繪製圖表（需要 matplotlib）")


if __name__ == "__main__":
    main()
