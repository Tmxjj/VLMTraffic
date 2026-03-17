'''
Author: yufei Ji
Date: 2026-03-17 21:51:40
LastEditTime: 2026-03-17 22:03:46
Description: this script is used to 
FilePath: /VLMTraffic/scripts/rename_datasets.py
'''
"""
Author: AI Assistant
Description: 将 data/sft_dataset 文件夹下各阶段数据集文件按新命名规范批量重命名：
  dataset.jsonl                                → 01_dataset_raw.jsonl
  dataset_auto_annotated.jsonl                → 02_dataset_auto_annotated.jsonl
  dataset_auto_annotated_final.jsonl          → 03_dataset_reviewed.jsonl
  dataset_auto_annotated_final_annotated.jsonl→ 04_dataset_final.jsonl
"""
import os
import argparse

RENAME_MAP = {
    "dataset.jsonl":                                 "01_dataset_raw.jsonl",
    "dataset_auto_annotated.jsonl":                  "02_dataset_auto_annotated.jsonl",
    "dataset_auto_annotated_final.jsonl":            "03_dataset_reviewed.jsonl",
    "dataset_auto_annotated_final_annotated.jsonl":  "04_dataset_final.jsonl",
}


def rename_in_dir(root_dir: str, dry_run: bool = False):
    renamed, skipped = 0, 0

    for dirpath, _, filenames in os.walk(root_dir):
        for old_name, new_name in RENAME_MAP.items():
            if old_name in filenames:
                src = os.path.join(dirpath, old_name)
                dst = os.path.join(dirpath, new_name)

                if os.path.exists(dst):
                    print(f"[SKIP]  目标文件已存在，跳过: {dst}")
                    skipped += 1
                    continue

                if dry_run:
                    print(f"[DRY]   {src}  →  {dst}")
                else:
                    os.rename(src, dst)
                    print(f"[OK]    {src}  →  {dst}")
                    renamed += 1

    print(f"\n完成：重命名 {renamed} 个文件，跳过 {skipped} 个文件。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量重命名 sft_dataset 下各阶段数据文件")
    parser.add_argument(
        "--root",
        type=str,
        default="data/sft_dataset",
        help="扫描的根目录（默认: data/sft_dataset）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印操作，不实际重命名",
    )
    args = parser.parse_args()
    rename_in_dir(args.root, dry_run=args.dry_run)
