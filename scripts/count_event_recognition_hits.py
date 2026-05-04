#!/usr/bin/env python3
"""
统计 data/eval 下 response.txt / repsonse.txt 中检测到 Event Recognition 的比例，
并输出每一个命中的路径定位信息。

目录约定：
    data/eval/{scenario}/{route}/{method}/{junction_id}/{sumo_step}/response.txt

判定逻辑：
1. 若文件包含 [Model Response] 段，只在该段中解析，避免把 prompt 模板里的
   "Event Recognition" 占位符误判为命中。
2. 提取 "- Event Recognition: ..." 这一行；
3. 当其内容不是 "None" 时，视为检测到事件。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "data" / "eval"
TARGET_FILENAMES = {"response.txt", "repsonse.txt"}

MODEL_RESPONSE_RE = re.compile(
    r"\[Model Response\]\s*(.*?)\s*(?=\[Thinking Process\]|\Z)",
    re.DOTALL,
)
EVENT_RECOGNITION_RE = re.compile(
    r"-\s*Event Recognition:\s*(.*)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count Event Recognition hits from eval response files."
    )
    parser.add_argument(
        "--eval_root",
        type=str,
        default=str(DEFAULT_EVAL_ROOT),
        help="评测结果根目录，默认 data/eval",
    )
    return parser.parse_args()


def iter_response_files(eval_root: Path) -> Iterable[Path]:
    for path in eval_root.rglob("*"):
        if path.is_file() and path.name in TARGET_FILENAMES:
            yield path


def extract_model_response(text: str) -> str:
    match = MODEL_RESPONSE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def extract_event_recognition_value(model_response: str) -> Optional[str]:
    for line in model_response.splitlines():
        match = EVENT_RECOGNITION_RE.search(line.strip())
        if match:
            return match.group(1).strip()
    return None


def is_event_detected(event_value: Optional[str]) -> bool:
    if not event_value:
        return False
    normalized = event_value.strip().strip("`").strip()
    return normalized.lower() != "none"


def format_hit_path(path: Path, eval_root: Path) -> str:
    rel = path.relative_to(eval_root)
    parts = rel.parts
    if len(parts) >= 6:
        scenario, route, method, junction_id, sumo_step = parts[:5]
        return (
            f"{scenario} | {route} | {method} | {junction_id} | {sumo_step}"
        )
    return str(rel)


def main() -> None:
    args = parse_args()
    eval_root = Path(args.eval_root).resolve()
    if not eval_root.exists():
        raise SystemExit(f"评测目录不存在: {eval_root}")

    response_files: List[Path] = sorted(iter_response_files(eval_root))
    total = len(response_files)
    hits: List[str] = []

    for path in response_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        model_response = extract_model_response(text)
        event_value = extract_event_recognition_value(model_response)
        if is_event_detected(event_value):
            hits.append(f"{format_hit_path(path, eval_root)} | {event_value}")

    ratio = (len(hits) / total) if total else 0.0

    print(f"eval_root: {eval_root}")
    print(f"response_files: {total}")
    print(f"event_detected: {len(hits)}")
    print(f"ratio: {ratio:.4%}")
    print("")
    print("Detected Paths:")
    for item in hits:
        print(item)


if __name__ == "__main__":
    main()
