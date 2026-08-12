"""下载 LakeQA 数据：任务 JSON（git clone 仓库）+ 数据文件（公共 S3 桶）。

用法:
    python scripts/download_lakeqa.py                            # 只 clone 仓库（任务 JSON，快）
    python scripts/download_lakeqa.py --with-data                # 同时按 datasets_used 下载 S3 数据
    python scripts/download_lakeqa.py --task-filter lakeqa_mini  # 只处理某个 split

说明：
  - S3 桶 lakeqa-yc4103-datalake 公共可读，无需 AWS 账号（等价 aws s3 cp --no-sign-request）。
  - 数据文件下载到 <dest>/data/，保持与 datasets_used 相同的相对路径（loader 约定）。
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO_URL = "https://github.com/lakeagent/datalake-qa"
S3_BUCKET = "lakeqa-yc4103-datalake"
SPLITS = ("lakeqa_mini", "lakeqa-full")


def _download_s3_file(key: str, local_root: Path, bucket: str) -> bool:
    local = local_root / key
    if local.exists() and local.stat().st_size > 0:
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://{bucket}.s3.amazonaws.com/{key}"
    print(f"  下载 {key}")
    try:
        urllib.request.urlretrieve(url, local)
        return True
    except Exception as e:
        print(f"  [warn] 下载失败 {key}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 LakeQA 任务 JSON + 数据")
    parser.add_argument("--dest", default="data/lakeqa", help="目标目录")
    parser.add_argument("--with-data", action="store_true", help="同时下载 datasets_used 的 S3 数据文件")
    parser.add_argument("--task-filter", choices=SPLITS, default=None, help="只处理某个 split")
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    # 1) 任务 JSON：git clone / pull
    repo_dir = dest / "repo"
    if not (repo_dir / ".git").exists():
        import subprocess
        print(f"git clone {REPO_URL} -> {repo_dir}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_dir)], check=True)
    else:
        print(f"仓库已存在，git pull: {repo_dir}")
        import subprocess
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=True)

    splits = SPLITS if args.task_filter is None else (args.task_filter,)
    task_files = []
    for split in splits:
        task_files += sorted((repo_dir / split).rglob("task_*.json"))

    print(f"找到 {len(task_files)} 个任务 JSON：{[str(p.relative_to(repo_dir)) for p in task_files[:5]]}{' ...' if len(task_files) > 5 else ''}")

    # 2) 可选：下载 S3 数据文件
    if args.with_data:
        data_root = dest / "data"
        keys = set()
        for p in task_files:
            try:
                t = json.loads(p.read_text(encoding="utf-8"))
                keys.update(t.get("datasets_used") or [])
            except (OSError, json.JSONDecodeError):
                continue
        print(f"共 {len(keys)} 个数据文件，开始下载到 {data_root} ...")
        for key in sorted(keys):
            _download_s3_file(key, data_root, S3_BUCKET)
        print("数据下载完成。")
    else:
        print("提示：未下载数据文件。评测前请运行带 --with-data 的命令。")


if __name__ == "__main__":
    sys.exit(main())
