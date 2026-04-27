"""
Author: Codex
Date: 2026-04-27
Description: 对已经落盘的 golden 数据进行后处理过滤，并输出带缩进的可读版本

设计原则：
1. 不侵入生成流程：仅处理已有的原始数据文件，不改动 golden 生成逻辑。
2. 过滤可叠加：复用 sample_filters.py 中的过滤链，后续加规则时无需修改本脚本主体。
3. 兼容历史格式：支持读取当前 `append_response_to_file` 产生的 `-----` 分隔文件。
4. 人类可读：输出使用 `json.dumps(..., indent=2)`，便于人工检查与抽样验收。

python src/dataset/golden_gener/2_filter_golden_dataset.py --input_path data/sft_dataset/JiNan/anon_3_4_jinan_real_debris/01_dataset_raw.jsonl
python src/dataset/golden_gener/2_filter_golden_dataset.py --input_path data/sft_dataset/JiNan/anon_3_4_jinan_real_debris/01_dataset_raw.jsonl data/sft_dataset/JiNan/anon_3_4_jinan_real_emergy/01_dataset_raw.jsonl
"""

import argparse
import json
import os
import sys
from typing import Dict, Generator, List, Tuple


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


from src.dataset.golden_gener.sample_filters import build_default_sample_filter_chain


DEFAULT_INPUT_PATH = os.path.join(
    _PROJECT_ROOT,
    "data",
    "sft_dataset",
    "JiNan",
    "anon_3_4_jinan_real_2000",
    "01_dataset_raw.jsonl",
)

DEFAULT_OUTPUT_PATH = os.path.join(
    _PROJECT_ROOT,
    "data",
    "sft_dataset",
    "JiNan",
    "anon_3_4_jinan_real_2000",
    "02_dataset_filtered_pretty.jsonl",
)

DEFAULT_OUTPUT_FILENAME = os.path.basename(DEFAULT_OUTPUT_PATH)


def iter_json_objects(file_path: str) -> Generator[Dict, None, None]:
    """
    从数据文件中逐个解析 JSON 对象。

    兼容两类输入：
    1. 历史原始文件：单行 JSON + `-----` 分隔符
    2. 可读输出文件：多行缩进 JSON 对象

    实现方式：
    - 先将整个文件读入内存
    - 使用 `json.JSONDecoder().raw_decode` 从任意偏移位置连续解码
    - 遇到分隔符、空白字符等非 JSON 内容时自动跳过
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    decoder = json.JSONDecoder()
    cursor = 0
    content_length = len(content)

    while cursor < content_length:
        while cursor < content_length and content[cursor].isspace():
            cursor += 1

        if cursor >= content_length:
            break

        if content.startswith("-----", cursor):
            cursor += len("-----")
            continue

        try:
            json_obj, next_cursor = decoder.raw_decode(content, cursor)
        except json.JSONDecodeError:
            cursor += 1
            continue

        yield json_obj
        cursor = next_cursor


def filter_samples(input_path: str) -> Tuple[List[Dict], Dict[str, object], List[str]]:
    """
    执行样本过滤并返回统计信息。

    返回：
    - filtered_samples: 保留下来的样本列表
    - stats: 总量/保留量/过滤量统计，以及按原因聚合的过滤统计
    - drop_reasons: 每条被过滤样本的原因文本，便于控制台查看
    """
    filter_chain = build_default_sample_filter_chain()
    filtered_samples: List[Dict] = []
    drop_reasons: List[str] = []

    total_count = 0
    kept_count = 0
    dropped_count = 0

    for index, sample in enumerate(iter_json_objects(input_path), start=1):
        total_count += 1
        passed, decision = filter_chain.apply(sample)
        if passed:
            filtered_samples.append(sample)
            kept_count += 1
            continue

        dropped_count += 1
        junction_id = sample.get("junction_id", "unknown_junction")
        sumo_step = sample.get("sumo_step", "unknown_step")
        reason = decision.reason if decision else "未知过滤原因"
        filter_name = decision.filter_name if decision else "unknown_filter"
        drop_reasons.append(
            f"#{index} | junction_id={junction_id} | sumo_step={sumo_step} | "
            f"{filter_name}: {reason}"
        )

    stats: Dict[str, object] = {
        "total_count": total_count,
        "kept_count": kept_count,
        "dropped_count": dropped_count,
    }
    stats.update(filter_chain.get_stats())
    return filtered_samples, stats, drop_reasons


def write_pretty_jsonl(samples: List[Dict], output_path: str) -> None:
    """
    将过滤后的样本写成“人类可读”的 JSONL 风格文件。

    说明：
    - 每个样本使用 `indent=2` 写成多行 JSON
    - 样本之间用空行分隔，方便人工阅读
    - 这更偏向“pretty JSON blocks”而非严格单行 JSONL，
      但符合当前用户提出的“有缩进保存到新的 jsonl 文件中”的要求
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for index, sample in enumerate(samples):
            if index > 0:
                file.write("\n")
            file.write(json.dumps(sample, ensure_ascii=False, indent=2))
            file.write("\n")


def resolve_output_path(input_path: str) -> str:
    """
    根据输入文件路径自动生成输出文件路径。

    规则：输出到输入文件同目录，文件名固定为 02_dataset_filtered_pretty.jsonl。
    """
    input_dir = os.path.dirname(os.path.abspath(input_path))
    return os.path.join(input_dir, DEFAULT_OUTPUT_FILENAME)


def main():
    parser = argparse.ArgumentParser(description="过滤已落盘的 golden 数据，并输出可读版 JSONL")
    parser.add_argument(
        "--input_path",
        type=str,
        nargs="+",
        default=[DEFAULT_INPUT_PATH],
        help="待过滤的原始数据文件路径，支持一次输入多个路径",
    )
    parser.add_argument(
        "--show_drop_reasons",
        action="store_true",
        help="是否在控制台打印每条被过滤样本的原因",
    )
    args = parser.parse_args()
    total_groups = len(args.input_path)
    for group_index, input_path in enumerate(args.input_path, start=1):
        output_path = resolve_output_path(input_path)

        filtered_samples, stats, drop_reasons = filter_samples(input_path)
        write_pretty_jsonl(filtered_samples, output_path)

        print(f"[{group_index}/{total_groups}] 输入文件: {input_path}")
        print(f"[{group_index}/{total_groups}] 输出文件: {output_path}")
        print(
            f"[{group_index}/{total_groups}] 样本统计: "
            f"total={stats['total_count']} | kept={stats['kept_count']} | "
            f"dropped={stats['dropped_count']}"
        )
        print(f"[{group_index}/{total_groups}] 按过滤原因统计: {stats['dropped_count_by_reason']}")
        print(f"[{group_index}/{total_groups}] 按清洗规则统计: {stats['cleaned_count_by_filter']}")

        if args.show_drop_reasons:
            print(f"[{group_index}/{total_groups}] 过滤详情:")
            for reason in drop_reasons:
                print(reason)

        if group_index < total_groups:
            print("-" * 80)


if __name__ == "__main__":
    main()
