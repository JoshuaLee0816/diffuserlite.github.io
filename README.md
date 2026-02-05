# DiffuserLite 實驗復現

基於 [DiffuserLite](https://github.com/diffuserlite/diffuserlite.github.io) 論文 "DiffuserLite: Towards Real-time Diffusion Planning" 的實驗復現。

---

## 一、最小可行性工作流 (Colab 免費版)

由於 Google Colab 免費版有 GPU 時間限制與斷線風險，採用以下策略：

### 階段 1：訓練 (Training)
| 步驟 | 說明 | 預估時間 |
|------|------|----------|
| 1.1 | 安裝環境 + Clone repo | 5 分鐘 |
| 1.2 | 訓練 0 → 200k 步 | ~2 小時 |
| 1.3 | **備份 checkpoint 到 Google Drive** | 1 分鐘 |
| 1.4 | 訓練 200k → 500k 步 | ~2.5 小時 |
| 1.5 | **備份 checkpoint** | 1 分鐘 |
| 1.6 | 訓練 500k → 1M 步 | ~2.5 小時 |
| 1.7 | **最終備份** | 1 分鐘 |

### 階段 2：推論 (Inference)
| 步驟 | 說明 | 預估時間 |
|------|------|----------|
| 2.1 | 載入 checkpoint | 1 分鐘 |
| 2.2 | 執行 inference (50 envs × 3 episodes) | ~10 分鐘 |
| 2.3 | 記錄 normalized score | - |

### 斷線恢復策略
```python
# 每次訓練前先備份到 Google Drive
!cp -r /content/diffuserlite.github.io/results /content/drive/MyDrive/DiffuserLite_backup

# 斷線後恢復
!cp -r /content/drive/MyDrive/DiffuserLite_backup/* /content/diffuserlite.github.io/results/
```

---

## 二、實驗環境參數

### 硬體環境
| 項目 | 規格 |
|------|------|
| Platform | Google Colab |
| GPU | Tesla T4 (16GB) |
| Python | 3.12 |

### 訓練超參數
| 參數 | 值 | 說明 |
|------|-----|------|
| `batch_size` | 256 | 批次大小 |
| `diffusion_gradient_steps` | 1,000,000 | 擴散模型訓練步數 |
| `invdyn_gradient_steps` | 1,000,000 | 逆動力學模型訓練步數 |
| `ema_rate` | 0.9999 | EMA 衰減率 |
| `discount` | 0.997 | 折扣因子 |
| `d_model` | 256 | Transformer 隱藏維度 |
| `n_heads` | 8 | 注意力頭數 |
| `depth` | 2 | Transformer 層數 |
| `log_interval` | 1,000 | 輸出 log 間隔 |
| `save_interval` | 200,000 | 存檔間隔 |

### 推論參數
| 參數 | 值 |
|------|-----|
| `num_envs` | 50 |
| `num_episodes` | 3 |
| `temperature` | 0.5 |
| `use_ema` | True |

---

## 三、訓練進度檢查清單

### HalfCheetah-medium-expert-v2 (R1)

- [ ] **Training**
  - [ ] 0 → 200k 步完成
  - [ ] 200k → 500k 步完成
  - [ ] 500k → 1M 步完成
  - [ ] Checkpoint 已備份到 Google Drive
- [ ] **Inference**
  - [ ] R1 模型推論完成
  - [ ] Normalized score 已記錄

---

## 四、實驗結果

### D4RL Normalized Scores

| Environment | Paper (R1) | My Result | 差異 |
|-------------|------------|-----------|------|
| halfcheetah-medium-expert-v2 | 91.9 | **TBD** | - |

### 訓練曲線

#### Loss Curve
![Loss Curve](./results/figures/loss_curve.png)

#### Inverse Dynamics Loss
![Invdyn Loss](./results/figures/invdyn_loss_curve.png)

---

## 五、關鍵圖表說明

| 圖表名稱 | 檔案路徑 | 說明 |
|----------|----------|------|
| Loss Curve | `results/figures/loss_curve.png` | 三層擴散模型的訓練損失曲線 (loss0, loss1, loss2) |
| Invdyn Loss | `results/figures/invdyn_loss_curve.png` | 逆動力學模型損失曲線 |
| Normalized Score | `results/figures/normalized_score.png` | (Optional) 不同訓練步數的 inference 分數 |

---

## 六、快速開始 (Colab)

### Cell 1: 安裝環境
```python
import os
os.chdir('/content')
!rm -rf /content/diffuserlite.github.io
!git clone https://github.com/JoshuaLee0816/diffuserlite.github.io.git /content/diffuserlite.github.io
!pip install /content/diffuserlite.github.io -q
!pip install git+https://github.com/Farama-Foundation/D4RL.git --ignore-requires-python -q
!pip install "numpy>=1.26.0,<2.0.0" -q
!mkdir -p /root/.mujoco
!wget -q https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O /tmp/mujoco210.tar.gz
!tar -xzf /tmp/mujoco210.tar.gz -C /root/.mujoco/
!apt-get install -qq -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf > /dev/null 2>&1
from google.colab import drive
drive.mount('/content/drive')
print("✅ 安裝完成！")
```

### Cell 2: 執行訓練
```python
# 寫入 run.py (含 mujoco_py shim)
# ... (見 test.ipynb)

!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin && python3 /content/run.py
```

### Cell 3: 備份結果
```python
!cp -r /content/diffuserlite.github.io/results /content/drive/MyDrive/DiffuserLite_results
```

---

## 七、自動化腳本

### 繪製 Loss 曲線 (需自行記錄 loss)
```python
import matplotlib.pyplot as plt
import os

def plot_loss_curves(loss_data, save_dir):
    """
    loss_data: dict with keys 'loss0', 'loss1', 'loss2', 'invdyn_loss'
               each value is a list of (step, loss) tuples
    """
    os.makedirs(save_dir, exist_ok=True)

    # Plot diffusion losses
    fig, ax = plt.subplots(figsize=(10, 6))
    for key in ['loss0', 'loss1', 'loss2']:
        if key in loss_data:
            steps, losses = zip(*loss_data[key])
            ax.plot(steps, losses, label=key)
    ax.set_xlabel('Gradient Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Diffusion Model Training Loss')
    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot invdyn loss
    if 'invdyn_loss' in loss_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        steps, losses = zip(*loss_data['invdyn_loss'])
        ax.plot(steps, losses, color='orange')
        ax.set_xlabel('Gradient Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Inverse Dynamics Model Training Loss')
        ax.grid(True)
        plt.savefig(os.path.join(save_dir, 'invdyn_loss_curve.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✅ 圖表已儲存到 {save_dir}")

# 使用範例：
# loss_data = {
#     'loss0': [(1000, 2.5), (2000, 2.1), ...],
#     'loss1': [(1000, 2.0), (2000, 1.8), ...],
#     'loss2': [(1000, 1.1), (2000, 0.9), ...],
#     'invdyn_loss': [(1000, 0.02), (2000, 0.015), ...]
# }
# plot_loss_curves(loss_data, '/content/diffuserlite.github.io/results/figures')
```

### 移動圖表到指定資料夾
```python
import shutil
import os

def organize_figures():
    """將所有生成的圖表移動到 results/figures/"""
    src_dir = '/content/diffuserlite.github.io/results'
    dst_dir = '/content/diffuserlite.github.io/results/figures'
    os.makedirs(dst_dir, exist_ok=True)

    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith('.png'):
                src = os.path.join(root, f)
                dst = os.path.join(dst_dir, f)
                shutil.move(src, dst)
                print(f"Moved: {f}")

    print(f"✅ 所有圖表已移動到 {dst_dir}")

# organize_figures()
```

---

## 參考資料

- [DiffuserLite Paper](https://arxiv.org/abs/YOUR_PAPER_ID)
- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)
- [D4RL](https://github.com/Farama-Foundation/D4RL)