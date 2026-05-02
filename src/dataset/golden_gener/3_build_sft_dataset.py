"""
Author: Codex
Date: 2026-04-30
Description: 汇总 data/sft_dataset 下的 03_final_dataset.jsonl，构造 VLM SFT 训练数据

核心规则：
1. 输入文件：递归扫描 data/sft_dataset/**/03_final_dataset.jsonl。
2. 输入格式：兼容当前 golden 数据的 JSON block + `-----` 分隔符。
3. 输出格式：每条样本包含 id / image / conversations，其中 image 为多图路径列表。
4. Prompt：使用源样本 prompt，并同步新版 prompt 文案与短字段名。
5. Response：只使用 golden_response，按固定字段正则提取并重新排版。
6. 过滤：filter_decision == discard、无 golden_response、正则提取失败的样本不进入 SFT，
   而是写入 failed 输出文件，保留 fail_reason 方便复查。

默认运行：
python src/dataset/golden_gener/3_build_sft_dataset.py
"""

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from collections import deque
from typing import Deque, Dict, Generator, List, Optional, Tuple
import random


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


DEFAULT_INPUT_ROOT = os.path.join(_PROJECT_ROOT, "data", "sft_dataset")
DEFAULT_OUTPUT_PATH = os.path.join(DEFAULT_INPUT_ROOT, "04_sft_train_dataset.jsonl")
DEFAULT_FAILED_OUTPUT_PATH = os.path.join(DEFAULT_INPUT_ROOT, "04_sft_failed_dataset.jsonl")
DEFAULT_STATS_OUTPUT_PATH = os.path.join(DEFAULT_INPUT_ROOT, "04_sft_dataset_stats.json")
DEFAULT_REMOTE_IMAGE_ROOT = "/root/autodl-tmp/sft_dataset"
DEFAULT_LOCAL_IMAGE_ROOT = DEFAULT_INPUT_ROOT


# 事件类型保持原样，Category 用这张表统一纠偏。
EVENT_CATEGORIES = {
    "Ambulance": "Emergency",
    "Police Car": "Emergency",
    "Fire Truck": "Emergency",
    "Public Bus": "Transit",
    "School Bus": "Transit",
    "Traffic Accident": "Crash",
    "Road Debris": "Obstruction",
    "Construction Barrier": "Obstruction",
}

EVENT_TYPE_BY_LOWER = {key.lower(): key for key in EVENT_CATEGORIES}
EVENT_TYPE_PATTERN = "|".join(re.escape(key) for key in EVENT_CATEGORIES)
EVENT_CATEGORY_PATTERN = "|".join(sorted(set(EVENT_CATEGORIES.values())))

DENSITY_LEVELS = {"Empty", "Short", "Medium", "Long", "Critical"}
PRESSURE_LEVELS = {"Empty", "Low", "Medium", "High", "Severe"}
DURATION_OPTIONS = {15, 20, 25, 30, 35, 40}

# 少数历史样本误把 Phase Pressure 写成了队列等级，这里按保守规则纠正。
# Pressure 自身是五级：Empty / Low / Medium / High / Severe。
# 当历史样本给出队列等级时，按需求映射为对应的压力等级。
PRESSURE_FROM_DENSITY = {
    "Empty": "Empty",
    "Short": "Low",
    "Medium": "Medium",
    "Long": "High",
    "Critical": "Severe",
}


# 仅做保守短字段替换，不改事件类型，避免影响后续按 Specific Type 统计。
SHORT_TOKEN_REPLACEMENTS = [
    ("OverallPressure", "Pressure"),
    ("CriticalQueue", "MaxQueue"),
    ("Neighboring Messages", "Neighbor Msgs"),
    ("Condition Assessment", "Condition"),
    ("Duration Reasoning", "Dur Reasoning"),
    ("Lane Analysis (Queue Assessment)", "Lane Analysis"),
]


PROMPT_TEXT_REPLACEMENTS = [
    (
        "- **Capacity Reduction**: A crash or road obstruction effectively removes one or more lanes from service. "
        "Avoid selecting a phase whose movement is blocked. If all phases are partially blocked, minimize green time "
        "on the most-blocked phase.",
        "- **Accident-Aware Capacity Reduction**: If an accident partially blocks a movement, prioritize the affected phase when abnormal congestion or upstream spillback risk exists."
        " If the movement is completely blocked, avoid serving that phase and allocate green time to other phases with longer queues or higher discharge potential.",
    ),
    (
        "- **Pressure**: A holistic synthesis of traffic demand for the phase. "
        "Output: [Low, Medium, High, or Severe]. *(Primary factor for Phase Selection)*",
        "- **Pressure**: A holistic synthesis of traffic demand for the phase. "
        "Output: [Empty, Low, Medium, High, or Severe]. *(Primary factor for Phase Selection)*",
    ),
    (
        "**Visual Localization:**\n"
        "- IF an event is detected: Specify [Specific Type], [Category], [Location: Approach & Lane ID], "
        "and [Directly Affected Phase ID].\n"
        "- IF NO event is present: Strictly output `None`.",
        "**Visual Localization:**\n"
        "- IF one or more events are detected: Specify EACH event's [Specific Type], [Category], "
        "[Location: Approach & Lane ID], and [Directly Affected Phase ID].\n"
        "- Multiple traffic events may be present in the same scene; list all detected events separately.\n"
        "- If an event is located in an unrestricted lane that is not signal-controlled, its Directly Affected "
        "Phase ID must be `None`.\n"
        "- IF NO event is present: Strictly output `None`.",
    ),
    (
        '- Event Recognition: <"None" OR "[Specific Type] ([Category]) detected at [Approach & Lane ID], '
        'affects Phase [ID]">',
        '- Event Recognition: <"None" OR one or more entries in the format '
        '"[Specific Type] ([Category]) detected at [Approach & Lane ID], affects Phase [ID/None]">',
    ),
    (
        '- Broadcast Notice: Format as "[Specific Type] - [Brief impact on upstream/downstream]" '
        'if an event is detected, else "None".',
        '- Broadcast Notice: If `Event Recognition` contains one or more locally detected events, output each notice '
        'as "[Specific Type] - [Brief impact on upstream/downstream]"; otherwise output "None". Received broadcast '
        'information in `Neighbor Msgs` must NOT be re-broadcast.',
    ),
]


TRAFFIC_ACCIDENT_CUE_OLD = (
    "- Traffic Accident: Crashed vehicles with visible structural deformation, positioned abnormally "
    "(e.g., stopped diagonally across lanes)."
)
TRAFFIC_ACCIDENT_CUE_NEW = (
    TRAFFIC_ACCIDENT_CUE_OLD + " A traffic incident occurs on only one lane."
)
ROAD_DEBRIS_CUE_OLD = "- Road Debris: Non-vehicle scattered objects (e.g., logs, fallen cargo) blocking the lane."
ROAD_DEBRIS_CUE_NEW = ROAD_DEBRIS_CUE_OLD + " Road debris occurs on only one lane."


FIELD_LABELS = [
    "Lane Analysis",
    "Phase Mapping",
    "Event Recognition",
    "Neighbor Msgs",
    "Condition",
    "Impact Analysis",
    "Phase Reasoning",
    "Dur Reasoning",
    "Broadcast Notice",
]


class NormalizeError(ValueError):
    """
    单条样本正则化失败时抛出的异常。

    输入：
    - message: 失败原因文本，通常包含 `missing_field:xxx` 或 `invalid_vocab:xxx`。

    输出：
    - 该类本身不产生返回值；异常会在 build_sft_dataset() 中被捕获，
      并写入 failed jsonl 的 fail_reason 字段。
    """


def iter_json_objects(file_path: str) -> Generator[Dict, None, None]:
    """
    从一个源数据文件中逐个解析 JSON 对象。

    输入：
    - file_path: 源文件路径，通常是 data/sft_dataset/**/03_final_dataset.jsonl。

    输出：
    - Generator[Dict, None, None]: 逐条 yield 解析成功的样本字典。

    实现说明：
    - 这里不用按行解析，因为源文件是多行缩进 JSON，并用 `-----` 分隔。
    - json.JSONDecoder().raw_decode 可以从任意位置解码一个完整 JSON 对象，因此能同时兼容：
    - 单行 JSONL；
    - 多行 JSON block；
    - block 之间有 `-----` 或空白字符。
    - 遇到非 JSON 噪声会向后滑动一个字符继续尝试，不会让整个文件解析中断。
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    decoder = json.JSONDecoder()
    cursor = 0
    while cursor < len(content):
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1

        if cursor >= len(content):
            break

        if content.startswith("-----", cursor):
            cursor += len("-----")
            continue

        try:
            json_obj, next_cursor = decoder.raw_decode(content, cursor)
        except json.JSONDecodeError:
            # 遇到非 JSON 噪声时向后滑动一位，保证历史脏文件不会让整个任务中断。
            cursor += 1
            continue

        yield json_obj
        cursor = next_cursor


def find_final_dataset_files(input_root: str) -> List[str]:
    """
    递归查找所有最终 golden 数据文件。

    输入：
    - input_root: 待扫描的数据根目录，例如 data/sft_dataset。

    输出：
    - List[str]: 所有名为 03_final_dataset.jsonl 的文件路径，按字典序排序，
      保证每次运行的样本顺序稳定。
    """
    matched_files: List[str] = []
    for root, _, files in os.walk(input_root):
        if "03_final_dataset.jsonl" in files:
            matched_files.append(os.path.join(root, "03_final_dataset.jsonl"))
    return sorted(matched_files)


def apply_short_token_replacements(text: str) -> str:
    """
    执行保守短字段替换，减少 prompt/response 中重复长字段的 token 开销。

    输入：
    - text: 任意 prompt 或 response 文本。

    输出：
    - str: 替换后的文本。例如 OverallPressure -> Pressure，
      CriticalQueue -> MaxQueue。事件类型名称不会被替换。
    """
    for old, new in SHORT_TOKEN_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def apply_prompt_updates(prompt: str) -> str:
    """
    把历史 prompt 更新到当前约定的新版本。

    输入：
    - prompt: 源样本中的 prompt 字符串。

    输出：
    - str: 已完成新版规则替换、短字段替换和行尾空格清理的 prompt。

    异常：
    - NormalizeError("missing_prompt"): prompt 缺失或不是字符串。

    处理内容：
    - 更新 Crash Clearance Priority、Visual Localization、多事件说明等 prompt 文案。
    - 幂等补充 Traffic Accident / Road Debris “只影响一条车道”的说明。
    - 将 OverallPressure / CriticalQueue 等长字段替换为短字段。
    """
    if not prompt or not isinstance(prompt, str):
        raise NormalizeError("missing_prompt")

    updated = prompt.replace("\r\n", "\n")
    for old, new in PROMPT_TEXT_REPLACEMENTS:
        updated = updated.replace(old, new)

    # 这两条 cue 的旧句子是新版句子的前缀，因此必须幂等处理，避免未来新版 prompt 被重复追加。
    if "A traffic accident affects only one lane." not in updated:
        updated = updated.replace(TRAFFIC_ACCIDENT_CUE_OLD, TRAFFIC_ACCIDENT_CUE_NEW)
    if "Road debris affects only one lane." not in updated:
        updated = updated.replace(ROAD_DEBRIS_CUE_OLD, ROAD_DEBRIS_CUE_NEW)

    updated = apply_short_token_replacements(updated)
    updated = updated.replace(
        "- **Pressure**: A holistic synthesis of traffic demand for the phase. "
        "Output: [Low, Medium, High, or Severe]. *(Primary factor for Phase Selection)*",
        "- **Pressure**: A holistic synthesis of traffic demand for the phase. "
        "Output: [Empty, Low, Medium, High, or Severe]. *(Primary factor for Phase Selection)*",
    )
    updated = updated.replace("`SPECIAL`", "`Special`").replace("`NORMAL`", "`Normal`")
    updated = updated.replace("Condition Assessment", "Condition")
    updated = updated.replace("Neighboring messages", "Neighbor Msgs")
    updated = updated.replace("neighboring messages", "Neighbor Msgs")
    updated = updated.replace('Otherwise, output "N/A".', 'Otherwise, output "None".')
    updated = updated.replace('Tie-Breaker: <"N/A" OR', 'Tie-Breaker: <"None" OR')
    return trim_trailing_spaces(updated)


def trim_trailing_spaces(text: str) -> str:
    """
    清理文本首尾和每行行尾空格。

    输入：
    - text: 待清理的多行文本。

    输出：
    - str: 去除整体首尾空白与每行尾部空格后的文本；原有换行结构会被保留。
    """
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def normalize_response_labels(text: str) -> str:
    """
    统一 golden_response 中的字段名，并修复少数样本把多个字段挤在同一行的问题。

    输入：
    - text: 源样本的 golden_response 原文。

    输出：
    - str: 字段名规范后的 response 文本。

    处理内容：
    - 长字段替换为短字段，例如 Neighboring Messages -> Neighbor Msgs。
    - 修复 Speical / Noraml / SPECIAL / NORMAL 等历史拼写或大小写问题。
    - 将 `affects phase` 统一成 `affects Phase`。
    - 对类似
      `- Event Recognition: ... - Neighboring Messages: Inactive - Condition Assessment: Special`
      的挤在同一行的字段，在下一个 `- 字段名:` 前补换行。
    """
    normalized = text.replace("\r\n", "\n")
    normalized = apply_short_token_replacements(normalized)
    normalized = normalized.replace("Meidum", "Medium").replace("meidum", "Medium")
    normalized = normalized.replace("Speical", "Special").replace("Noraml", "Normal")
    normalized = normalized.replace("SPECIAL", "Special").replace("NORMAL", "Normal")
    normalized = re.sub(r"\baffects\s+phase\b", "affects Phase", normalized, flags=re.IGNORECASE)

    for label in FIELD_LABELS:
        normalized = re.sub(
            r"\s+-\s*" + re.escape(label) + r"\s*:",
            "\n- " + label + ":",
            normalized,
        )
    return normalized


def extract_field(text: str, label: str, required: bool = True) -> Optional[str]:
    """
    从规范化后的 response 中提取指定字段内容。

    输入：
    - text: 已经 normalize_response_labels() 处理过的 response 文本。
    - label: 需要提取的字段名，例如 Lane Analysis / Phase Mapping / Event Recognition。
    - required: 是否为必需字段。True 表示缺失或为空时抛出 NormalizeError；
      False 表示缺失时返回 None。

    输出：
    - Optional[str]: 字段值文本；当 required=False 且字段缺失时返回 None。

    边界说明：
    - 提取范围从 `- Label:` 之后开始，到下一个标准字段、下一个 A/B/C 大段落、
      `]` 或 `Action:` 之前结束。
    - Phase Mapping 内部的 `Tie-Breaker:` 不以 `-` 开头，因此会被保留在
      Phase Mapping block 内。
    """
    next_labels = "|".join(re.escape(item) for item in FIELD_LABELS)
    pattern = re.compile(
        r"(?:^|\n)-\s*"
        + re.escape(label)
        + r"\s*:\s*(.*?)"
        + r"(?=(?:\n-\s*(?:"
        + next_labels
        + r")\s*:)|(?:\n[A-C]\.\s)|(?:\n\])|(?:\nAction\s*:)|\Z)",
        re.DOTALL,
    )
    match = pattern.search(text)
    if not match:
        if required:
            raise NormalizeError("missing_field:" + label)
        return None

    value = trim_trailing_spaces(match.group(1))
    if required and not value:
        raise NormalizeError("empty_field:" + label)
    return value or None


def canonicalize_word(word: str, allowed_words: set) -> str:
    """
    把大小写不稳定的枚举词纠正为标准写法。

    输入：
    - word: 待纠正的词，例如 `medium`、`SPECIAL`。
    - allowed_words: 合法词集合，例如 {"Empty", "Short", ...}。

    输出：
    - str: allowed_words 中的标准写法。

    异常：
    - NormalizeError("invalid_vocab:xxx"): 输入词不在 allowed_words 中。
    """
    lower_to_word = {item.lower(): item for item in allowed_words}
    normalized = lower_to_word.get(word.strip().lower())
    if not normalized:
        raise NormalizeError("invalid_vocab:" + word)
    return normalized


def normalize_lane_analysis(raw_lane: str) -> str:
    """
    规范 Lane Analysis 中的队列密度词汇。

    输入：
    - raw_lane: Lane Analysis 字段的原始多行文本。

    输出：
    - str: 队列密度词统一为 Empty / Short / Medium / Long / Critical 后的文本。

    异常：
    - NormalizeError("lane_analysis_no_density_level"): 没有找到任何合法队列密度词。
    """
    found_levels: List[str] = []

    def replace_level(match: re.Match) -> str:
        """
        将单个正则匹配到的密度词替换为标准写法。

        输入：
        - match: 匹配 `:<Density>` 的正则结果。

        输出：
        - str: 标准化后的 `:<Density>` 片段。
        """
        level = canonicalize_word(match.group(1), DENSITY_LEVELS)
        found_levels.append(level)
        return ":" + level

    lane = re.sub(
        r":\s*(Empty|Short|Medium|Long|Critical)\b",
        replace_level,
        raw_lane,
        flags=re.IGNORECASE,
    )
    if not found_levels:
        raise NormalizeError("lane_analysis_no_density_level")
    return trim_trailing_spaces(lane)


def normalize_phase_mapping(raw_phase_mapping: str) -> str:
    """
    规范 Phase Mapping 的短字段名、压力等级和最大队列等级。

    输入：
    - raw_phase_mapping: Phase Mapping 字段的原始多行文本。

    输出：
    - str: 使用 Pressure / MaxQueue 短字段，并统一等级词后的 Phase Mapping。

    异常：
    - NormalizeError("phase_mapping_no_phase"): 没有发现 Phase 行。
    - NormalizeError("phase_mapping_no_pressure"): 没有提取到 Pressure。
    - NormalizeError("phase_mapping_no_max_queue"): 没有提取到 MaxQueue。

    兼容逻辑：
    - Pressure 标准等级为 Empty/Low/Medium/High/Severe。
    - 少数历史样本误把 Pressure 写成队列等级 Empty/Short/Medium/Long/Critical，
      会通过 PRESSURE_FROM_DENSITY 保守映射成 Empty/Low/Medium/High/Severe。
    """
    phase_mapping = apply_short_token_replacements(raw_phase_mapping)

    pressure_values: List[str] = []
    max_queue_values: List[str] = []

    def replace_pressure(match: re.Match) -> str:
        """
        将单个 Pressure 值替换为标准压力等级。

        输入：
        - match: 匹配 `Pressure: <Level>` 的正则结果。

        输出：
        - str: 标准化后的 `Pressure: <Level>` 片段。
        """
        raw_value = match.group(1)
        try:
            value = canonicalize_word(raw_value, PRESSURE_LEVELS)
        except NormalizeError:
            density_value = canonicalize_word(raw_value, DENSITY_LEVELS)
            value = PRESSURE_FROM_DENSITY[density_value]
        pressure_values.append(value)
        return "Pressure: " + value

    def replace_max_queue(match: re.Match) -> str:
        """
        将单个 MaxQueue 值替换为标准队列等级。

        输入：
        - match: 匹配 `MaxQueue: <Level>` 的正则结果。

        输出：
        - str: 标准化后的 `MaxQueue: <Level>` 片段。
        """
        value = canonicalize_word(match.group(1), DENSITY_LEVELS)
        max_queue_values.append(value)
        return "MaxQueue: " + value

    phase_mapping = re.sub(
        r"Pressure\s*:\s*(Empty|Short|Medium|Long|Critical|Low|High|Severe)\b",
        replace_pressure,
        phase_mapping,
        flags=re.IGNORECASE,
    )
    phase_mapping = re.sub(
        r"MaxQueue\s*:\s*(Empty|Short|Medium|Long|Critical)\b",
        replace_max_queue,
        phase_mapping,
        flags=re.IGNORECASE,
    )
    # 部分历史样本把 Phase 标题和 Pressure/MaxQueue 拆成两行，训练输出统一写在同一行。
    phase_mapping = re.sub(
        r"(Phase\s+\d+\s*\([^)]+\):)[ \t]*\n[ \t]*(?=Pressure\s*:)",
        r"\1 ",
        phase_mapping,
    )
    phase_mapping = re.sub(r"\bTie-Breaker\s*:\s*", "Tie-Breaker: ", phase_mapping)
    # 历史样本中少量 Tie-Breaker 写成 "N/A"，训练格式统一收敛为 None。
    phase_mapping = re.sub(
        r"(Tie-Breaker:\s*)(?:[\"']?[ \t]*N[ \t]*/?[ \t]*A[ \t]*[\"']?)(?=[ \t]*(?:\n|$))",
        r"\1None",
        phase_mapping,
        flags=re.IGNORECASE,
    )

    if "Phase " not in phase_mapping:
        raise NormalizeError("phase_mapping_no_phase")
    if not pressure_values:
        raise NormalizeError("phase_mapping_no_pressure")
    if not max_queue_values:
        raise NormalizeError("phase_mapping_no_max_queue")
    return trim_trailing_spaces(phase_mapping)


def normalize_neighbor_msgs(raw_neighbor: str) -> str:
    """
    规范 Neighbor Msgs 字段。

    输入：
    - raw_neighbor: Neighbor Msgs 字段原始值，可能包含大小写差异或额外文本。

    输出：
    - str: 只返回 `Inactive` 或 `Active`。

    异常：
    - NormalizeError("invalid_neighbor_msgs:xxx"): 无法判断为 Active / Inactive。

    设计原因：
    - 训练输出里该字段只需要表示邻居消息状态，避免夹带多余解释影响格式稳定性。
    """
    value = raw_neighbor.strip().strip('"').strip("'")
    lower = value.lower()
    if lower.startswith("inactive"):
        return "Inactive"
    if lower.startswith("active"):
        return "Active"
    raise NormalizeError("invalid_neighbor_msgs:" + value)


def normalize_condition(raw_condition: str) -> str:
    """
    规范 Condition 字段。

    输入：
    - raw_condition: Condition 字段原始值，可能是 Normal/Special 的大小写变体，
      或历史拼写错误 Speical/Noraml。

    输出：
    - str: 标准写法 `Normal` 或 `Special`。

    异常：
    - NormalizeError("invalid_condition:xxx"): 无法判断为 Normal / Special。
    """
    value = raw_condition.strip().strip('"').strip("'")
    value = value.replace("Speical", "Special").replace("Noraml", "Normal")
    value = value.replace("SPECIAL", "Special").replace("NORMAL", "Normal")
    if value.lower().startswith("normal"):
        return "Normal"
    if value.lower().startswith("special"):
        return "Special"
    raise NormalizeError("invalid_condition:" + value)


def clean_event_location(location: str) -> str:
    """
    轻量清洗 Event Recognition 中的 Location 字段。

    输入：
    - location: 从 `[Specific Type] ([Category]) detected at [Location], affects Phase ...`
      中提取出的原始位置文本。

    输出：
    - str: 轻量清洗后的位置文本。

    当前清洗内容：
    - `lane1` / `Lane 1` -> `L1`。
    - `west approach` -> `West Approach`。
    - 去除多余空格、尾部标点和部分 `&` 分隔符。

    边界说明：
    - 该函数不会强制把所有位置统一成 `<Approach> <LaneID>`。
      例如 `West Approach, L1`、`Approach East L1`、`South L2(S)` 仍可能保留原结构。
    - 如果后续需要严格字段级统计，应再用专门的位置解析正则提取 Approach / LaneID。
    """
    cleaned = location.strip().strip(",.;")
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\blane\s*(\d+)\b", r"L\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bLane\s+(\d+)\b", r"L\1", cleaned)
    cleaned = re.sub(
        r"\b(north|south|east|west)\s+approach\b",
        lambda match: match.group(1).capitalize() + " Approach",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace(" & ", " ")
    cleaned = cleaned.replace(" ,", ",")
    return cleaned.strip()


def normalize_event_recognition(raw_event: str) -> Tuple[str, List[str]]:
    """
    规范 Event Recognition。

    输入：
    - raw_event: Event Recognition 字段原始文本。例如：
      `Traffic Accident (Crash) detected at South L2, affects Phase 1`
      或多个事件用分号分隔。

    输出：
    - Tuple[str, List[str]]:
      1. 第一个元素是规范化后的 Event Recognition 字符串；
      2. 第二个元素是样本中出现的事件类型列表，用于后续统计。
         如果 raw_event 为 None，则返回 ["None"]。

    异常：
    - NormalizeError("event_recognition_parse_failed:xxx"): 非 None 事件无法解析为
      `[Specific Type] ([Category]) detected at [Location], affects Phase [ID/None]`。

    支持：
    - None
    - 单事件
    - 用分号分隔的多事件

    对不满足 `[Specific Type] ([Category]) detected at [Location], affects Phase [ID/None]`
    的样本直接判为失败，避免把脏标注写入训练集。
    """
    event_text = raw_event.strip().strip('"')
    if event_text in {"None", "`None`"}:
        return "None", ["None"]

    pattern = re.compile(
        r"(?P<type>"
        + EVENT_TYPE_PATTERN
        + r")\s*\(\s*(?P<category>"
        + EVENT_CATEGORY_PATTERN
        + r")\s*\)\s*detected\s+at\s+(?P<location>.+?)\s*,\s*affects\s+Phase\s+(?P<phase>None|\d+)\b",
        re.IGNORECASE,
    )

    entries: List[str] = []
    event_types: List[str] = []
    for match in pattern.finditer(event_text):
        event_type = EVENT_TYPE_BY_LOWER[match.group("type").lower()]
        category = EVENT_CATEGORIES[event_type]
        location = clean_event_location(match.group("location"))
        phase = match.group("phase")
        phase = "None" if phase.lower() == "none" else str(int(phase))
        entries.append(
            f"{event_type} ({category}) detected at {location}, affects Phase {phase}"
        )
        event_types.append(event_type)

    if not entries:
        # 历史中存在一条 "Bus detected detected ..." 的脏标注。
        # 该格式缺失 Category，但事件类型可以明确恢复为 Public Bus。
        bus_match = re.search(
            r"\bBus\s+detected\s+detected\s+at\s+(?P<location>.+?)\s*,\s*affects\s+Phase\s+(?P<phase>None|\d+)\b",
            event_text,
            flags=re.IGNORECASE,
        )
        if not bus_match:
            raise NormalizeError("event_recognition_parse_failed:" + event_text[:160])

        location = clean_event_location(bus_match.group("location"))
        phase = bus_match.group("phase")
        phase = "None" if phase.lower() == "none" else str(int(phase))
        entries.append(f"Public Bus (Transit) detected at {location}, affects Phase {phase}")
        event_types.append("Public Bus")

    return "; ".join(entries), event_types


def normalize_free_text(raw_text: Optional[str]) -> Optional[str]:
    """
    清理自由文本推理字段。

    输入：
    - raw_text: Impact Analysis / Phase Reasoning / Dur Reasoning /
      Broadcast Notice 等字段的原始文本；也可能为 None。

    输出：
    - Optional[str]: 清理后的文本；输入为 None 时返回 None。

    处理内容：
    - 短字段替换，例如 CriticalQueue -> MaxQueue。
    - 修复 Speical / Noraml / SPECIAL / NORMAL。
    - 去除每行尾部空格。

    注意：
    - 该函数不重写推理语义，只做格式和词汇层面的轻量清理。
    """
    if raw_text is None:
        return None
    text = apply_short_token_replacements(raw_text)
    text = text.replace("Speical", "Special").replace("Noraml", "Normal")
    text = text.replace("SPECIAL", "Special").replace("NORMAL", "Normal")
    return trim_trailing_spaces(text)


def parse_action(response_text: str) -> Dict[str, int]:
    """
    从 golden_response 中提取并校验最终 Action。

    输入：
    - response_text: 已经过 normalize_response_labels() 处理的完整 response 文本。

    输出：
    - Dict[str, int]: 标准动作字典，格式为 `{"phase": int, "duration": int}`。

    异常：
    - NormalizeError("missing_action"): 未找到 Action 字段。
    - NormalizeError("action_json_parse_failed:xxx"): Action 无法解析且兜底正则也失败。
    - NormalizeError("action_phase_duration_not_int"): phase 或 duration 不是整数。
    - NormalizeError("action_invalid_duration:xxx"): duration 不在 DURATION_OPTIONS 中。

    兼容逻辑：
    - 个别历史样本写成 `"duration": "40}`，语义明确但 JSON 不合法。
      JSON 解析失败后，会用严格数字正则兜底恢复 phase/duration。
    """
    match = re.search(r"Action\s*:\s*(\{.*?\})", response_text, re.DOTALL)
    if not match:
        raise NormalizeError("missing_action")

    action_text = match.group(1)
    try:
        action = json.loads(action_text)
    except json.JSONDecodeError as exc:
        # 个别历史样本写成 `"duration": "40}`，语义清楚但 JSON 不合法。
        # 为了不丢弃可恢复样本，失败后用严格数字正则兜底。
        phase_match = re.search(r'"phase"\s*:\s*(\d+)', action_text)
        duration_match = re.search(r'"duration"\s*:\s*"?(\d+)', action_text)
        if not phase_match or not duration_match:
            raise NormalizeError("action_json_parse_failed:" + str(exc))
        action = {
            "phase": int(phase_match.group(1)),
            "duration": int(duration_match.group(1)),
        }

    phase = action.get("phase")
    duration = action.get("duration")
    if not isinstance(phase, int) or not isinstance(duration, int):
        raise NormalizeError("action_phase_duration_not_int")
    if duration not in DURATION_OPTIONS:
        raise NormalizeError("action_invalid_duration:" + str(duration))
    return {"phase": phase, "duration": duration}


def normalize_golden_response(golden_response: str) -> Tuple[str, List[str], Dict[str, int]]:
    """
    从 golden_response 中抽取目标字段，并重新组织为稳定的 SFT 输出。

    输入：
    - golden_response: 源样本中的 golden_response 字符串，作为 SFT 的 gpt.value 来源。

    输出：
    - Tuple[str, List[str], Dict[str, int]]:
      1. str: 重新排版后的标准 response 文本；
      2. List[str]: 该样本出现的事件类型列表，用于统计；
      3. Dict[str, int]: 解析后的 Action 字典。

    必需字段：
    - Lane Analysis
    - Phase Mapping
    - Event Recognition
    - Neighbor Msgs
    - Condition
    - Phase Reasoning
    - Dur Reasoning
    - Action

    可选字段：
    - Impact Analysis
    - Broadcast Notice

    异常：
    - NormalizeError: 任一必需字段缺失、词汇非法、事件格式异常或 Action 异常时抛出。
    """
    if not golden_response or not isinstance(golden_response, str):
        raise NormalizeError("missing_golden_response")

    response = normalize_response_labels(golden_response)

    lane_analysis = normalize_lane_analysis(extract_field(response, "Lane Analysis"))
    phase_mapping = normalize_phase_mapping(extract_field(response, "Phase Mapping"))
    event_recognition, event_types = normalize_event_recognition(
        extract_field(response, "Event Recognition")
    )
    neighbor_msgs = normalize_neighbor_msgs(extract_field(response, "Neighbor Msgs"))
    condition = normalize_condition(extract_field(response, "Condition"))
    impact_analysis = normalize_free_text(extract_field(response, "Impact Analysis", required=False))
    phase_reasoning = normalize_free_text(extract_field(response, "Phase Reasoning"))
    dur_reasoning = normalize_free_text(extract_field(response, "Dur Reasoning"))
    broadcast_notice = normalize_free_text(extract_field(response, "Broadcast Notice", required=False))
    action = parse_action(response)

    # Broadcast Notice 只允许广播本路口本帧检测到的事件。
    # 当 Event Recognition 为 None 时，即使 Neighbor Msgs 为 Active，也不能把收到的广播再次输出。
    if event_recognition == "None" and (neighbor_msgs == "Active" or broadcast_notice):
        broadcast_notice = "None"

    lines: List[str] = [
        "Thought: [",
        "A. Scene Understanding:",
        "- Lane Analysis:",
        lane_analysis,
        "- Phase Mapping:",
        phase_mapping,
        "",
        "B. Scene Analysis:",
        "- Event Recognition: " + event_recognition,
        "- Neighbor Msgs: " + neighbor_msgs,
        "- Condition: " + condition,
        "",
        "C. Adaptive Reasoning:",
    ]
    if impact_analysis:
        lines.append("- Impact Analysis: " + impact_analysis)
    lines.extend(
        [
            "- Phase Reasoning: " + phase_reasoning,
            "- Dur Reasoning: " + dur_reasoning,
        ]
    )
    if broadcast_notice:
        lines.append("- Broadcast Notice: " + broadcast_notice)

    lines.extend(
        [
            "]",
            "Action: " + json.dumps(action, ensure_ascii=False),
        ]
    )
    return "\n".join(lines), event_types, action


def build_sample_id(sample: Dict, source_file: str) -> Tuple[str, str, str]:
    """
    构造 SFT 样本 id。

    输入：
    - sample: 源 JSON 样本字典，需要包含 scenario / sumo_step / junction_id。
    - source_file: 当前样本来源文件路径，用于推断 route 目录名和兜底 scenario。

    输出：
    - Tuple[str, str, str]:
      1. sample_id: `场景名-route目录名-step_{sumo_step}-{junction_id}`；
      2. scenario: 场景名；
      3. route_name: route 目录名。

    异常：
    - NormalizeError("missing_id_component"): 缺少 scenario / sumo_step / junction_id 等关键字段。
    """
    scenario = sample.get("scenario") or os.path.basename(os.path.dirname(os.path.dirname(source_file)))
    route_name = os.path.basename(os.path.dirname(source_file))
    sumo_step = sample.get("sumo_step")
    junction_id = sample.get("junction_id")

    if scenario is None or sumo_step is None or junction_id is None:
        raise NormalizeError("missing_id_component")

    sample_id = f"{scenario}-{route_name}-step_{sumo_step}-{junction_id}"
    return sample_id, str(scenario), route_name


def convert_image_paths(
    image_paths: object,
    local_image_root: str,
    remote_image_root: str,
) -> List[str]:
    """
    将本地 data/sft_dataset 路径替换为 /root/autodl-tmp/sft_dataset。

    输入：
    - image_paths: 源样本中的 image_paths 字段，要求为非空 list[str]。
    - local_image_root: 本地 data/sft_dataset 根目录。
    - remote_image_root: 训练环境中图片对应的远端根目录。

    输出：
    - List[str]: 替换为 remote_image_root 前缀后的多图路径列表。

    异常：
    - NormalizeError("missing_image_paths"): image_paths 缺失、不是列表或为空。
    - NormalizeError("invalid_image_path"): 列表中存在非字符串或空路径。
    - NormalizeError("image_path_outside_sft_root:xxx"): 图片路径无法映射到 sft_dataset 根目录。

    设计说明：
    输出保留多图列表，满足多视角 VLM SFT 训练输入。
    """
    if not isinstance(image_paths, list) or not image_paths:
        raise NormalizeError("missing_image_paths")

    converted_paths: List[str] = []
    local_root = os.path.abspath(local_image_root)
    remote_root = remote_image_root.rstrip("/")

    for image_path in image_paths:
        if not isinstance(image_path, str) or not image_path:
            raise NormalizeError("invalid_image_path")

        normalized_path = os.path.abspath(image_path) if image_path.startswith("/") else image_path
        if normalized_path.startswith(local_root + os.sep):
            relative_path = os.path.relpath(normalized_path, local_root)
        elif "/data/sft_dataset/" in image_path:
            relative_path = image_path.split("/data/sft_dataset/", 1)[1]
        elif image_path.startswith("data/sft_dataset/"):
            relative_path = image_path[len("data/sft_dataset/") :]
        elif image_path.startswith(remote_root + "/"):
            converted_paths.append(image_path)
            continue
        else:
            raise NormalizeError("image_path_outside_sft_root:" + image_path)

        converted_paths.append(remote_root + "/" + relative_path.replace(os.sep, "/"))

    return converted_paths


def write_jsonl(records: List[Dict], output_path: str) -> None:
    """
    将记录列表写成标准单行 JSONL 文件。

    输入：
    - records: 待写出的字典列表，例如正式 SFT 样本或 failed 样本。
    - output_path: 输出文件路径。

    输出：
    - None。

    副作用：
    - 自动创建输出目录。
    - 覆盖写入 output_path，每条记录占一行，便于训练脚本流式读取。
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


def write_json(data: Dict, output_path: str) -> None:
    """
    将统计字典写成可读 JSON 文件。

    输入：
    - data: 待写出的统计信息字典。
    - output_path: 输出文件路径。

    输出：
    - None。

    副作用：
    - 自动创建输出目录。
    - 覆盖写入 output_path，使用 indent=2 方便人工检查。
    """
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def build_failed_record(
    sample: Dict,
    source_file: str,
    source_index: int,
    fail_reason: str,
    sample_id: Optional[str] = None,
) -> Dict:
    """
    构造未进入正式 SFT 数据集的失败样本记录。

    输入：
    - sample: 原始源样本字典，会完整保存在 failed 记录的 sample 字段中。
    - source_file: 该样本来源的 03_final_dataset.jsonl 文件路径。
    - source_index: 样本在来源文件中的序号，从 1 开始。
    - fail_reason: 失败原因，例如 filter_decision_discard / missing_field:xxx。
    - sample_id: 已经构造出的样本 id；如果 id 构造阶段失败，则可能为 None。

    输出：
    - Dict: failed jsonl 中的一条记录，包含来源信息、失败原因、是否有 golden_response
      以及完整原始 sample。
    """
    return {
        "source_file": os.path.relpath(source_file, _PROJECT_ROOT),
        "source_index": source_index,
        "id": sample_id,
        "scenario": sample.get("scenario"),
        "route": os.path.basename(os.path.dirname(source_file)),
        "junction_id": sample.get("junction_id"),
        "sumo_step": sample.get("sumo_step"),
        "filter_decision": sample.get("filter_decision"),
        "has_golden_response": bool(sample.get("golden_response")),
        "fail_reason": fail_reason,
        "sample": sample,
    }


def counter_to_dict(counter: Counter) -> Dict[str, int]:
    """
    将 Counter 转为按 key 排序的普通 dict。

    输入：
    - counter: collections.Counter 计数器。

    输出：
    - Dict[str, int]: 普通字典，key 按字典序排序，保证统计文件每次输出顺序稳定。
    """
    return {key: counter[key] for key in sorted(counter)}


def build_shuffle_key(scenario: str, event_types: List[str]) -> str:
    """
    构造用于均匀打乱的分桶键。

    输入：
    - scenario: 场景名，例如 JiNan / France_Massy / Hongkong_YMT。
    - event_types: 当前样本解析出的事件类型列表。

    输出：
    - str: 分桶键，格式为 `scenario|event_type`；如果存在多个事件类型，则按字母序拼接为
      `scenario|type_a+type_b`；如果没有事件，则为 `scenario|None`。

    设计目的：
    - 让不同场景、不同事件类型的样本分布尽量均匀，避免输出顺序集中在同一类样本上。
    """
    if not event_types:
        return f"{scenario}|None"
    unique_types = sorted(set(event_types))
    return f"{scenario}|{'+'.join(unique_types)}"


def stratified_round_robin_shuffle(
    records: List[Dict],
    shuffle_keys: List[str],
    seed: int = 42,
) -> List[Dict]:
    """
    对正式 SFT 样本做分层轮转打乱。

    输入：
    - records: 待打乱的样本列表。
    - shuffle_keys: 与 records 等长的分桶键列表，每个键通常是 `scenario|event_type`。
    - seed: 随机种子，用于保证打乱结果可复现。

    输出：
    - List[Dict]: 打乱后的样本列表。

    打乱策略：
    - 先按分桶键分组；
    - 每个桶内部做一次随机打散；
    - 再按“轮转抽取”的方式从各桶中每次取 1 条，直到所有桶为空；
    - 每一轮会对活跃桶顺序做一次随机扰动，减少固定顺序带来的偏置。

    说明：
    - 该逻辑只作用于正式 SFT 数据，不改变样本内容和统计信息。
    """
    if len(records) != len(shuffle_keys):
        raise ValueError("records and shuffle_keys must have the same length")
    if not records:
        return []

    rng = random.Random(seed)
    buckets: Dict[str, Deque[Dict]] = defaultdict(deque)
    for record, key in zip(records, shuffle_keys):
        buckets[key].append(record)

    for key in list(buckets.keys()):
        bucket_items = list(buckets[key])
        rng.shuffle(bucket_items)
        buckets[key] = deque(bucket_items)

    active_keys = list(buckets.keys())
    rng.shuffle(active_keys)

    shuffled: List[Dict] = []
    while active_keys:
        rng.shuffle(active_keys)
        next_active_keys: List[str] = []
        for key in active_keys:
            bucket = buckets[key]
            if not bucket:
                continue
            shuffled.append(bucket.popleft())
            if bucket:
                next_active_keys.append(key)
        active_keys = next_active_keys
    return shuffled


def build_sft_dataset(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict], Dict]:
    """
    构建正式 SFT 数据、失败样本数据和统计报告。

    输入：
    - args: 命令行参数对象，至少包含：
      input_root / output_path / failed_output_path / stats_output_path /
      local_image_root / remote_image_root。

    输出：
    - Tuple[List[Dict], List[Dict], Dict]:
      1. sft_records: 正式 SFT 样本列表；
      2. failed_records: 未进入正式 SFT 的样本列表；
      3. stats_json: 全局统计和 route 级统计字典。

    主流程：
    - 扫描所有 03_final_dataset.jsonl。
    - 逐条解析 JSON block。
    - 跳过 filter_decision == discard。
    - 更新 prompt，正则化 golden_response，转换图片路径，构造 SFT 样本。
    - 捕获 NormalizeError，将失败样本写入 failed_records。
    - 汇总全局事件统计、失败原因统计和 route 级统计。

    异常：
    - RuntimeError: input_root 下没有找到任何 03_final_dataset.jsonl。
    """
    input_files = find_final_dataset_files(args.input_root)
    if not input_files:
        raise RuntimeError("未找到任何 03_final_dataset.jsonl: " + args.input_root)

    sft_records: List[Dict] = []
    sft_shuffle_keys: List[str] = []
    failed_records: List[Dict] = []
    seen_ids = set()

    global_event_sample_counts = Counter()
    global_event_occurrence_counts = Counter()
    failed_reason_counts = Counter()
    route_stats = defaultdict(
        lambda: {
            "source_count": 0,
            "sft_count": 0,
            "failed_count": 0,
            "discard_count": 0,
            "missing_golden_count": 0,
            "regex_failed_count": 0,
            "event_type_counts": Counter(),
            "event_occurrence_counts": Counter(),
        }
    )

    total_source_count = 0

    for source_file in input_files:
        for source_index, sample in enumerate(iter_json_objects(source_file), start=1):
            total_source_count += 1
            route_key = os.path.join(
                sample.get("scenario") or os.path.basename(os.path.dirname(os.path.dirname(source_file))),
                os.path.basename(os.path.dirname(source_file)),
            )
            route_stats[route_key]["source_count"] += 1

            sample_id = None
            try:
                sample_id, scenario_name, _ = build_sample_id(sample, source_file)

                if sample.get("filter_decision") == "discard":
                    raise NormalizeError("filter_decision_discard")
                if sample_id in seen_ids:
                    raise NormalizeError("duplicate_id:" + sample_id)

                prompt = apply_prompt_updates(sample.get("prompt"))
                response, event_types, _ = normalize_golden_response(sample.get("golden_response"))
                images = convert_image_paths(
                    sample.get("image_paths"),
                    args.local_image_root,
                    args.remote_image_root,
                )

                sft_record = {
                    "id": sample_id,
                    "image": images,
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt", "value": response},
                    ],
                }
                sft_records.append(sft_record)
                sft_shuffle_keys.append(build_shuffle_key(scenario_name, event_types))
                seen_ids.add(sample_id)

                route_stats[route_key]["sft_count"] += 1
                unique_event_types = sorted(set(event_types))
                for event_type in unique_event_types:
                    route_stats[route_key]["event_type_counts"][event_type] += 1
                    global_event_sample_counts[event_type] += 1
                for event_type in event_types:
                    route_stats[route_key]["event_occurrence_counts"][event_type] += 1
                    global_event_occurrence_counts[event_type] += 1

            except NormalizeError as exc:
                fail_reason = str(exc)
                failed_reason_counts[fail_reason] += 1
                route_stats[route_key]["failed_count"] += 1
                if fail_reason == "filter_decision_discard":
                    route_stats[route_key]["discard_count"] += 1
                if not sample.get("golden_response"):
                    route_stats[route_key]["missing_golden_count"] += 1
                if fail_reason not in {"filter_decision_discard", "missing_golden_response"}:
                    route_stats[route_key]["regex_failed_count"] += 1

                failed_records.append(
                    build_failed_record(sample, source_file, source_index, fail_reason, sample_id)
                )

    route_stats_json = {}
    for route_key in sorted(route_stats):
        stats = route_stats[route_key]
        route_stats_json[route_key] = {
            "source_count": stats["source_count"],
            "sft_count": stats["sft_count"],
            "failed_count": stats["failed_count"],
            "discard_count": stats["discard_count"],
            "missing_golden_count": stats["missing_golden_count"],
            "regex_failed_count": stats["regex_failed_count"],
            "event_type_counts": counter_to_dict(stats["event_type_counts"]),
            "event_occurrence_counts": counter_to_dict(stats["event_occurrence_counts"]),
        }

    # 只打乱正式 SFT 数据，failed 样本保留原始顺序，方便人工排查和回溯。
    sft_records = stratified_round_robin_shuffle(
        sft_records,
        sft_shuffle_keys,
        seed=args.shuffle_seed,
    )

    # 汇总全局统计，便于快速核对输入规模、输出规模、失败原因和事件分布。
    stats_json = {
        # 本次递归扫描的数据根目录，相对项目根目录保存，避免不同机器上的绝对路径不一致。
        "input_root": os.path.relpath(args.input_root, _PROJECT_ROOT),
        # 实际扫描到的 03_final_dataset.jsonl 文件数量。
        "input_file_count": len(input_files),
        # 源数据总样本数，包含可训练样本、discard 样本和其他失败样本。
        "total_source_count": total_source_count,
        # 最终写入正式 SFT 数据集的样本数。
        "sft_count": len(sft_records),
        # 未进入正式 SFT 数据集的样本数，都会写入 failed_output_path。
        "failed_count": len(failed_records),
        # 因 filter_decision == discard 被过滤的样本数。
        "discard_count": sum(stats["discard_count"] for stats in route_stats.values()),
        # 缺失 golden_response 的样本数；该字段独立统计，不一定等同于 failed_reason。
        "missing_golden_count": sum(stats["missing_golden_count"] for stats in route_stats.values()),
        # 非 discard 且非缺失 golden_response，但正则化/格式校验失败的样本数。
        "regex_failed_count": sum(stats["regex_failed_count"] for stats in route_stats.values()),
        # 正式 SFT 数据、失败样本和统计报告的输出路径。
        "output_path": os.path.relpath(args.output_path, _PROJECT_ROOT),
        "failed_output_path": os.path.relpath(args.failed_output_path, _PROJECT_ROOT),
        "stats_output_path": os.path.relpath(args.stats_output_path, _PROJECT_ROOT),
        # 图片路径替换后的远端根目录，用于确认训练环境路径是否正确。
        "remote_image_root": args.remote_image_root,
        # 按事件类型统计样本数；多事件样本中同一事件类型只计一次。
        "event_type_counts": counter_to_dict(global_event_sample_counts),
        # 按事件类型统计事件出现次数；多事件样本中的每个事件都会计数。
        "event_occurrence_counts": counter_to_dict(global_event_occurrence_counts),
        # 按失败原因聚合计数，便于定位数据清洗问题。
        "failed_reason_counts": counter_to_dict(failed_reason_counts),
        # route 级统计，包含每个 route 的样本量、失败量和事件类型分布。
        "route_stats": route_stats_json,
        # 正式 SFT 输出的打乱策略与随机种子，便于复现。
        "shuffle_strategy": "stratified_round_robin_by_scenario_and_event_type",
        "shuffle_seed": args.shuffle_seed,
    }
    return sft_records, failed_records, stats_json


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    输入：
    - 无显式参数；从 sys.argv 读取命令行参数。

    输出：
    - argparse.Namespace: 包含输入根目录、输出路径、图片路径替换根目录等配置。
    """
    parser = argparse.ArgumentParser(description="汇总 03_final_dataset.jsonl 为 SFT 训练数据")
    parser.add_argument(
        "--input_root",
        type=str,
        default=DEFAULT_INPUT_ROOT,
        help="递归扫描的 sft_dataset 根目录",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="正式 SFT 数据输出路径",
    )
    parser.add_argument(
        "--failed_output_path",
        type=str,
        default=DEFAULT_FAILED_OUTPUT_PATH,
        help="未进入 SFT 的样本输出路径",
    )
    parser.add_argument(
        "--stats_output_path",
        type=str,
        default=DEFAULT_STATS_OUTPUT_PATH,
        help="统计报告输出路径",
    )
    parser.add_argument(
        "--local_image_root",
        type=str,
        default=DEFAULT_LOCAL_IMAGE_ROOT,
        help="本地 image_paths 中 data/sft_dataset 对应的根目录",
    )
    parser.add_argument(
        "--remote_image_root",
        type=str,
        default=DEFAULT_REMOTE_IMAGE_ROOT,
        help="训练环境中替换后的图片根目录",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="正式 SFT 数据分层轮转打乱的随机种子",
    )
    return parser.parse_args()


def main() -> None:
    """
    脚本入口函数。

    输入：
    - 无显式参数；通过 parse_args() 读取命令行配置。

    输出：
    - None。

    副作用：
    - 生成正式 SFT JSONL。
    - 生成 failed JSONL。
    - 生成 stats JSON。
    - 在控制台打印输入文件数、源样本数、SFT 样本数和输出路径。
    """
    args = parse_args()
    sft_records, failed_records, stats_json = build_sft_dataset(args)

    write_jsonl(sft_records, args.output_path)
    write_jsonl(failed_records, args.failed_output_path)
    write_json(stats_json, args.stats_output_path)

    print("输入文件数:", stats_json["input_file_count"])
    print("源样本数:", stats_json["total_source_count"])
    print("SFT样本数:", stats_json["sft_count"])
    print("失败/过滤样本数:", stats_json["failed_count"])
    print("SFT输出:", args.output_path)
    print("失败输出:", args.failed_output_path)
    print("统计输出:", args.stats_output_path)


if __name__ == "__main__":
    main()
