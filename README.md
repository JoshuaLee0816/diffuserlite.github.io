# DiffuserLite 實驗復現

基於 [DiffuserLite](https://github.com/diffuserlite/diffuserlite.github.io) 論文 "DiffuserLite: Towards Real-time Diffusion Planning" 的實驗復現。

---

## 一、專案架構

```
.
├── 01_DiffuserLite_Test.ipynb  # Colab 訓練 notebook
├── inference_local.py          # 本機端推論 + 學習曲線分析
├── pipelines/                  # 訓練 pipeline
├── configs/                    # Hydra 設定檔
└── cleandiffuser/              # 核心模型程式碼
```

---

## 二、訓練流程 (Colab)

由於 Google Colab 免費版有 GPU 時間限制與斷線風險，採用以下策略：

### Checkpoint 保存策略
- 每 1000 步上傳一次 checkpoint 到 Hugging Face
- 自動清理舊 checkpoint，只保留步數為 `1000, 6000, 11000, 16000...` 的版本
- 這樣可以用較少的儲存空間，同時保留完整的學習曲線資料

### 斷線恢復策略
使用 Hugging Face 作為 checkpoint 儲存庫：
- 斷線後重新執行 notebook，會自動從 Hugging Face 下載最新 checkpoint 並繼續訓練
- Hugging Face Repo: https://huggingface.co/JoshuaLee0816/diffuserlite-results

---

## 三、本機端推論 (Learning Curve Analysis)

使用 `inference_local.py` 可從 Hugging Face 下載所有 checkpoint，並依序執行推論分析學習曲線。

### 功能
- 自動下載所有 checkpoint (1000, 6000, 11000...)
- 對每個 checkpoint 執行推論並記錄 normalized score
- 錄製每個 checkpoint 的 MuJoCo 影片（攝影機會跟隨 agent）
- 合併所有影片成一個「學習過程」影片
- 輸出結果表格

### 使用方式
```bash
# 安裝額外依賴
pip install huggingface_hub imageio[ffmpeg]

# 執行推論
python inference_local.py
```

### 輸出檔案
- `inference_results.json` - 各步數的推論結果
- `inference_video_stepXXXX.mp4` - 各步數的影片
- `learning_progress.mp4` - 合併的學習過程影片

---

## 四、實驗環境參數

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
| `save_interval` | 1,000 | 上傳間隔（實際保留 5000 步間隔）|

### 推論參數
| 參數 | 值 |
|------|-----|
| `num_envs` | 50 |
| `num_episodes` | 3 |
| `temperature` | 0.5 |
| `use_ema` | True |

---

## 五、實驗結果

### D4RL Normalized Scores

| Environment | Paper (R1) | My Result | 差異 |
|-------------|------------|-----------|------|
| halfcheetah-medium-expert-v2 | 91.9 | **TBD** | - |

### 學習曲線
執行 `inference_local.py` 後，可在 `results/figures/` 查看：

| 圖表 | 說明 |
|------|------|
| `learning_progress.mp4` | 各訓練步數的 agent 行為影片 |
| `inference_results.json` | 各步數的 normalized score |

---

## 六、參考資料

- [DiffuserLite 論文](https://arxiv.org/abs/2401.15443)
- [DiffuserLite GitHub](https://github.com/diffuserlite/diffuserlite.github.io)
- [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser)
- [D4RL Dataset](https://github.com/Farama-Foundation/D4RL)