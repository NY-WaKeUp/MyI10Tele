import os

# 解决无法访问 huggingface.co 的问题，防止脚本在读取元数据时卡死
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Use the path you found in your terminal
dataset_root = os.path.expanduser(
    "~/openpi-cache/huggingface/lerobot/lerobot/aloha_sim_transfer_cube_human"
)

print("\n" + "=" * 50)
print("🔍 数据集深度检查工具")
print("=" * 50)

if os.path.exists(dataset_root):
    print(f"📍 目标路径: {dataset_root}")
    print(f"Root contents: {os.listdir(dataset_root)}")
    root_contents = os.listdir(dataset_root)
    print(f"Root contents: {root_contents}")

    if "lerobot" in root_contents:
        print(f"💡 发现子目录 'lerobot'，检查其中是否包含转换后的数据...")
        nested_path = (
            Path(dataset_root) / "lerobot" / "aloha_sim_transfer_cube_human_v30"
        )
        if nested_path.exists():
            print(f"✨ 找到可能的 v3.0 目录: {nested_path}")
            # 取消下面一行的注释可以直接检查该目录
            # dataset_root = str(nested_path)

    # 1. Manual check of info.json
    # In LeRobot, 'meta' is a sibling of 'data'
    info_path = Path(dataset_root) / "meta" / "info.json"
    if info_path.exists():
        print(f"Found metadata at: {info_path}")
        with open(info_path, "r") as f:
            info = json.load(f)
            # Print just a few interesting keys
            version = info.get("codebase_version", "v2.1 or older")
            print(f"📊 代码库版本: {version}")

            if str(version) != "v3.0":
                print("⚠️  Warning: This dataset needs conversion to v3.0 format.")

            print(f"📝 总回合数 (Episodes): {info.get('total_episodes')}")
            print(f"📝 总帧数 (Frames): {info.get('total_frames')}")
    else:
        print(f"❌ info.json not found at {info_path}")

    # 2. 物理文件统计 (检查数据是否真的存在)
    print("\n--- 物理文件统计 ---")
    data_path = Path(dataset_root) / "data"

    if data_path.exists():
        parquet_files = list(data_path.glob("**/*.parquet"))
        print(f"📦 Parquet 数据文件数量: {len(parquet_files)}")
        if len(parquet_files) > 0:
            print(
                f"   样例路径: {parquet_files[0].name} (位于 {parquet_files[0].parent.name})"
            )
    else:
        print("❌ 'data' 文件夹缺失")

    video_path = Path(dataset_root) / "videos"
    if video_path.exists():
        video_files = list(video_path.glob("**/*.mp4"))
        print(f"🎥 MP4 视频文件数量: {len(video_files)}")
        if len(video_files) > 0:
            print(f"   样例视频: {video_files[0].name}")
    else:
        print("⚠️ 'videos' 文件夹缺失 (ACT 策略训练通常需要视频)")

    # 3. 尝试使用元数据 API (仅当版本匹配时)
    print("\n--- LeRobot API 加载测试 ---")
    try:
        # 注意：这里使用 local 模式加载
        metadata = LeRobotDatasetMetadata("aloha_sim", root=dataset_root)
        print(f"✅ Metadata API 加载成功")
        print(f"   特征列表: {list(metadata.features.keys())}")

        ds = LeRobotDataset("aloha_sim", root=dataset_root)
        print(f"✅ Dataset 对象加载成功! 长度: {len(ds)}")

    except Exception as e:
        print(f"❌ API 加载失败 (如果转换成功，这里不应该再报错):")
        print(f"   错误简报: {str(e).splitlines()[0]}")

else:
    print(f"❌ 路径不存在: {dataset_root}")
print("=" * 50 + "\n")
