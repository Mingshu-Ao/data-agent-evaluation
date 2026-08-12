"""下载 KramaBench 数据集（HF 镜像 eugenie-y/kramabench）到本地。

用法:
    python scripts/download_kramabench.py                    # 全量（workload+data+solutions，~1.7GB）
    python scripts/download_kramabench.py --workload-only     # 只下 workload/*.json（快，用于 loader 冒烟测试）
    python scripts/download_kramabench.py --dest D:/data/krama
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO = "eugenie-y/kramabench"


def main() -> None:
    """下载 KramaBench（HF 镜像）到本地目录。"""
    parser = argparse.ArgumentParser(description="下载 KramaBench（HF 镜像）")
    parser.add_argument("--dest", default="data/kramabench", help="目标目录")
    parser.add_argument("--workload-only", action="store_true",
                        help="只下载 workload/*.json（问题清单），跳过 1.7GB 原始数据")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    patterns = ["workload/*"] if args.workload_only else None
    print(f"从 HF 下载 KramaBench 到 {dest}（patterns={patterns or 'all'}）...")
    snapshot_download(
        HF_REPO, repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=patterns,
    )
    print("完成。")
    if args.workload_only:
        print("提示：完整评测还需原始数据，请再运行一次不带 --workload-only 的命令。")


if __name__ == "__main__":
    sys.exit(main())
