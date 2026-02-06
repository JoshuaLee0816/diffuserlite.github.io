# DiffuserLite 實驗復現

基於 [DiffuserLite](https://github.com/diffuserlite/diffuserlite.github.io) 論文 "DiffuserLite: Towards Real-time Diffusion Planning" 的實驗復現。

---

## 一、最小可行性工作流 (Colab 免費版)

由於 Google Colab 免費版有 GPU 時間限制與斷線風險，採用以下策略：

### 階段 0：測試階段 (先確認流程正常) ✅ 已完成
| 步驟 | 說明 | 狀態 |
|------|------|------|
| 0.1 | 安裝環境 + Clone repo | ✅ |
| 0.2 | 訓練 500 步（測試用） | ✅ |
| 0.3 | 確認 checkpoint 有正確存檔 | ✅ |
| 0.4 | 確認 checkpoint 可載入 | ✅ |
| 0.5 | 備份到 Hugging Face | ⏳ 待執行 |

**⚠️ 階段 0 通過後，才進入正式訓練。**

### 階段 1：訓練 (Training)
| 步驟 | 說明 | 狀態 |
|------|------|------|
| 1.0 | 從 Hugging Face 下載之前的 checkpoint（斷線恢復） | ⏳ |
| 1.1 | 設定正式訓練參數 | ⏳ |
| 1.2 | 執行訓練（自動從 checkpoint 繼續） | ⏳ |
| 1.3 | **備份 checkpoint 到 Hugging Face** | ⏳ |

### 階段 2：推論 (Inference)
| 步驟 | 說明 |
|------|------|
| 2.1 | 載入 checkpoint |
| 2.2 | 執行 inference (50 envs × 3 episodes) |
| 2.3 | 記錄 normalized score |

### 斷線恢復策略
使用 Hugging Face 作為 checkpoint 儲存庫：

```python
# 斷線後恢復：執行 notebook 中的「階段 1.0」cell
# 會自動從 Hugging Face 下載之前的 checkpoint

# 備份：執行 notebook 中的「階段 1.3」cell
# 會上傳所有 checkpoint 到 Hugging Face
```

Hugging Face Repo: https://huggingface.co/JoshuaLee0816/diffuserlite-results

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

- [x] **階段 0：測試**
  - [x] 500 步訓練完成
  - [x] Checkpoint 存檔正常
  - [x] Checkpoint 可正常載入
  - [ ] Hugging Face 備份完成
- [ ] **階段 1：正式訓練**
  - [ ] 訓練中（目前步數：0）
  - [ ] Checkpoint 已備份到 Hugging Face
- [ ] **階段 2：推論**
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