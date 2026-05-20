import os

# 1. 必须在 import huggingface_hub 之前设置环境变量
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

my_token = ""
with open("/Users/ningyu/.hf_token") as f:
    my_token = f.readline()

try:
    print("开始从镜像站下载 lerobot/pi0fast-base...")
    snapshot_download(
        repo_id="lerobot/pi0fast-base",
        local_dir="lerobot/pi0fast-base",
        endpoint="https://hf-mirror.com",  # 显式再次指定
        local_dir_use_symlinks=False,
        resume_download=True,
        token=my_token,
    )
    print("下载完成！")
except Exception as e:
    print(f"下载失败: {e}")
