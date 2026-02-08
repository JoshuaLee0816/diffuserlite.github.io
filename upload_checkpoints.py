#!/usr/bin/env python3
"""
上傳本機 checkpoint 到 Hugging Face
帶有 rate limit 處理和重試機制
"""

import os
import time
from pathlib import Path
from huggingface_hub import HfApi, upload_file

# 設定
REPO_ID = "JoshuaLee0816/diffuserlite-results"
LOCAL_DIR = Path(__file__).parent / "checkpoints" / "halfcheetah-medium-expert-v2"
HF_PATH_PREFIX = "diffuserlite_d4rl_mujoco/halfcheetah-medium-expert-v2"

# Rate limit 設定
DELAY_BETWEEN_UPLOADS = 3  # 每次上傳間隔秒數
MAX_RETRIES = 5            # 最大重試次數
RETRY_DELAY = 60           # 遇到 rate limit 時等待秒數


def upload_with_retry(api, local_path, repo_path):
    """上傳單個檔案，帶重試機制"""
    for attempt in range(MAX_RETRIES):
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=repo_path,
                repo_id=REPO_ID,
                repo_type="model"
            )
            return True
        except Exception as e:
            if "rate limit" in str(e).lower() or "429" in str(e):
                wait_time = RETRY_DELAY * (attempt + 1)
                print(f"  Rate limit! 等待 {wait_time} 秒後重試...")
                time.sleep(wait_time)
            else:
                print(f"  錯誤: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"  等待 {RETRY_DELAY} 秒後重試...")
                    time.sleep(RETRY_DELAY)
                else:
                    return False
    return False


def main():
    api = HfApi()

    # 取得所有 .pt 檔案
    pt_files = list(LOCAL_DIR.glob("*.pt"))
    print(f"找到 {len(pt_files)} 個 checkpoint 檔案")

    # 檢查已存在的檔案
    try:
        existing_files = set(api.list_repo_files(REPO_ID, repo_type="model"))
    except:
        existing_files = set()

    # 上傳
    success_count = 0
    skip_count = 0
    fail_count = 0

    for i, local_path in enumerate(sorted(pt_files)):
        filename = local_path.name
        repo_path = f"{HF_PATH_PREFIX}/{filename}"

        # 檢查是否已存在
        if repo_path in existing_files:
            print(f"[{i+1}/{len(pt_files)}] 跳過 (已存在): {filename}")
            skip_count += 1
            continue

        print(f"[{i+1}/{len(pt_files)}] 上傳中: {filename}")

        if upload_with_retry(api, local_path, repo_path):
            print(f"  成功!")
            success_count += 1
        else:
            print(f"  失敗!")
            fail_count += 1

        # 延遲避免 rate limit
        if i < len(pt_files) - 1:
            time.sleep(DELAY_BETWEEN_UPLOADS)

    print(f"\n完成! 成功: {success_count}, 跳過: {skip_count}, 失敗: {fail_count}")


if __name__ == "__main__":
    main()
