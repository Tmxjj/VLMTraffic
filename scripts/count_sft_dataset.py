#!/usr/bin/env python3
"""
统计 data/sft_dataset 文件夹下所有 03_final_dataset.jsonl 文件中的数据量。
输出 route 和 Event Recognition 类型下的数据量统计。

注意：这些 JSONL 文件可能采用格式化的多行 JSON 格式，并用 ----- 分隔记录。
统计时会跳过 filter_decision == "discard" 的记录。
"""

import json
import re
from collections import Counter
from pathlib import Path


EVENT_RECOGNITION_PATTERN = re.compile(
    r"^\s*-\s*Event Recognition:\s*(?P<value>.+?)\s*$",
    re.MULTILINE,
)
EVENT_TYPE_WITH_CATEGORY_PATTERN = re.compile(
    r"^(?P<event_type>.+?)\s*\([^)]+\)\s+detected\b",
    re.IGNORECASE,
)
EVENT_TYPE_WITHOUT_CATEGORY_PATTERN = re.compile(
    r"^(?P<event_type>.+?)\s+detected\b",
    re.IGNORECASE,
)


def iter_json_records(file_path):
    """逐条读取 JSON 记录，兼容普通 JSONL、多行 JSON 和 ----- 分隔格式。"""
    content = file_path.read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    idx = 0

    while idx < len(content):
        while idx < len(content):
            if content[idx].isspace():
                idx += 1
                continue
            if content.startswith("-----", idx):
                next_line = content.find("\n", idx)
                idx = len(content) if next_line == -1 else next_line + 1
                continue
            break

        if idx >= len(content):
            break

        try:
            record, end_idx = decoder.raw_decode(content, idx)
        except json.JSONDecodeError as exc:
            line_no = content.count("\n", 0, idx) + 1
            raise ValueError(f"JSON parse failed at line {line_no}: {exc}") from exc

        if isinstance(record, dict):
            yield record
        idx = end_idx


def normalize_filter_decision(record):
    filter_decision = record.get("filter_decision")
    if filter_decision is None:
        return "Missing"
    return str(filter_decision).strip()


def extract_event_type(record):
    """从 golden_response 的 Event Recognition 字段提取 Specific Type。"""
    response = record.get("golden_response") or record.get("golden_repsonse") or ""
    if not response:
        return "Missing"

    match = EVENT_RECOGNITION_PATTERN.search(response)
    if not match:
        return "Unparsed"

    event_value = match.group("value").strip().strip('"')
    while event_value.lower().startswith("event recognition:"):
        event_value = event_value.split(":", 1)[1].strip()

    if event_value.lower() == "none":
        return "None"

    event_match = EVENT_TYPE_WITH_CATEGORY_PATTERN.search(event_value)
    if not event_match:
        event_match = EVENT_TYPE_WITHOUT_CATEGORY_PATTERN.search(event_value)
    if event_match:
        return event_match.group("event_type").strip()

    return "Unparsed"


def count_records_and_stats(file_path):
    """统计非 discard 记录数、filter_decision 和事件类型数据量。"""
    filter_decision_stats = Counter()
    event_type_stats = Counter()
    total_count = 0

    try:
        for record in iter_json_records(file_path):
            filter_decision = normalize_filter_decision(record)
            if filter_decision.lower() == "discard":
                continue

            total_count += 1
            filter_decision_stats[filter_decision] += 1
            event_type_stats[extract_event_type(record)] += 1
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, {}, {}

    return total_count, dict(filter_decision_stats), dict(event_type_stats)


def format_counter(counter):
    """将 Counter/字典格式化为紧凑的一行展示文本。"""
    if not counter:
        return "None"
    return ", ".join(f"{key}: {counter[key]}" for key in sorted(counter.keys()))


def main():
    base_path = Path(__file__).parent.parent / "data" / "sft_dataset"

    if not base_path.exists():
        print(f"Error: {base_path} not found!")
        return

    all_routes = []
    global_filter_decisions = Counter()
    global_event_types = Counter()

    for jsonl_file in sorted(base_path.glob("*/*/03_final_dataset.jsonl")):
        route = jsonl_file.parent.name
        scenario = jsonl_file.parent.parent.name

        record_count, filter_decision_stats, event_type_stats = count_records_and_stats(jsonl_file)

        all_routes.append(
            {
                "scenario": scenario,
                "route": route,
                "count": record_count,
                "filter_decisions": filter_decision_stats,
                "event_types": event_type_stats,
            }
        )
        global_filter_decisions.update(filter_decision_stats)
        global_event_types.update(event_type_stats)

    total_count = sum(route_data["count"] for route_data in all_routes)

    print("=" * 100)
    print("SFT Dataset 统计结果（已排除 filter_decision=discard）")
    print("=" * 100)
    print()

    print("📊 Route 级别统计表:")
    print("-" * 160)
    print(f"{'场景':<25} {'Route':<35} {'数据量':>10}  {'Filter Decisions':<30} {'Event Types'}")
    print("-" * 160)
    for route_data in sorted(all_routes, key=lambda x: (x["scenario"], x["route"])):
        print(
            f"{route_data['scenario']:<25} "
            f"{route_data['route']:<35} "
            f"{route_data['count']:>10}  "
            f"{format_counter(route_data['filter_decisions']):<30} "
            f"{format_counter(route_data['event_types'])}"
        )
    print("-" * 160)
    print()

    print("=" * 100)
    print(f"🎯 总数据量: {total_count} samples")
    print(f"📁 总文件数: {len(all_routes)} files")
    print("=" * 100)
    print()

    print("📊 Route 级别排序表 (按数据量降序):")
    print("-" * 80)
    print(f"{'场景':<25} {'Route':<35} {'数据量':>10}")
    print("-" * 80)
    for route_data in sorted(all_routes, key=lambda x: x["count"], reverse=True):
        print(f"{route_data['scenario']:<25} {route_data['route']:<35} {route_data['count']:>10}")
    print("-" * 80)
    print()

    print("🧾 Filter Decision 汇总表（已排除 discard）:")
    print("-" * 80)
    print(f"{'Filter Decision':<50} {'数据量':>15} {'占比':>10}")
    print("-" * 80)
    for filter_decision in sorted(global_filter_decisions.keys()):
        count = global_filter_decisions[filter_decision]
        percentage = f"{count/total_count*100:.1f}%" if total_count > 0 else "0%"
        print(f"{filter_decision:<50} {count:>15} {percentage:>10}")
    print("-" * 80)
    print()

    print("🚦 Event Recognition 类型汇总表:")
    print("-" * 80)
    print(f"{'Event Type':<50} {'数据量':>15} {'占比':>10}")
    print("-" * 80)
    for event_type in sorted(global_event_types.keys()):
        count = global_event_types[event_type]
        percentage = f"{count/total_count*100:.1f}%" if total_count > 0 else "0%"
        print(f"{event_type:<50} {count:>15} {percentage:>10}")
    print("-" * 80)


if __name__ == "__main__":
    main()
