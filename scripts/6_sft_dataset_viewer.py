"""
6_sft_dataset_viewer.py - SFT / Failed 数据人工查看前端

用法：
    python scripts/6_sft_dataset_viewer.py

可选参数：
    python scripts/6_sft_dataset_viewer.py \
        --sft_path data/sft_dataset/04_sft_train_dataset.jsonl \
        --failed_path data/sft_dataset/04_sft_failed_dataset.jsonl \
        --host 127.0.0.1 \
        --port 8502

说明：
1. 这是一个只读查看器，不会修改 SFT 或 failed 数据文件。
2. 参考 scripts/5_data_filter.py 的查看逻辑，但不依赖 streamlit。
3. 仅使用 Python 标准库启动本地 HTTP 服务，前端 HTML/JS/CSS 内嵌在本文件中。
4. 支持筛选、翻页、多图查看、prompt/response/raw JSON 查看，以及 Event Recognition 结构化解析。
"""

from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SFT_PATH = PROJECT_ROOT / "data" / "sft_dataset" / "04_sft_train_dataset.jsonl"
DEFAULT_FAILED_PATH = PROJECT_ROOT / "data" / "sft_dataset" / "04_sft_failed_dataset.jsonl"
DEFAULT_REMOTE_IMAGE_PREFIX = "/root/autodl-tmp/sft_dataset"
DEFAULT_LOCAL_IMAGE_PREFIX = str(PROJECT_ROOT / "data" / "sft_dataset")

EVENT_ENTRY_RE = re.compile(
    r"^(?P<specific_type>Ambulance|Police Car|Fire Truck|Public Bus|School Bus|Traffic Accident|Road Debris|Construction Barrier) "
    r"\((?P<category>Emergency|Transit|Crash|Obstruction)\) detected at (?P<location>.+), affects Phase (?P<phase>None|\d+)$"
)

LOCATION_PAIR_RE = re.compile(
    r"(?:(?P<prefix_approach>North|South|East|West)\s+(?:Approach\s*,?\s*)?|Approach\s+(?P<suffix_approach>North|South|East|West)\s+)"
    r"(?P<lane_id>L\d+)(?:\((?P<lane_func>[LSR])\))?",
    re.IGNORECASE,
)


class ViewerData:
    """
    保存查看器运行期间需要复用的数据。

    输入：
    - sft_records: 正式 SFT 样本列表。
    - failed_records: failed 样本列表。
    - sft_index: SFT 样本筛选索引。
    - failed_index: failed 样本筛选索引。
    - remote_image_prefix: 数据中图片路径的远端前缀。
    - local_image_prefix: 本机可读取的图片根目录。

    输出：
    - dataclass 实例；供 HTTP handler 查询和渲染接口使用。
    """

    def __init__(
        self,
        sft_records: List[Dict[str, Any]],
        failed_records: List[Dict[str, Any]],
        sft_index: List[Dict[str, Any]],
        failed_index: List[Dict[str, Any]],
        remote_image_prefix: str,
        local_image_prefix: str,
    ) -> None:
        """
        初始化查看器数据容器。

        输入：
        - sft_records: 正式 SFT 样本列表。
        - failed_records: failed 样本列表。
        - sft_index: SFT 样本筛选索引。
        - failed_index: failed 样本筛选索引。
        - remote_image_prefix: 数据中图片路径的远端前缀。
        - local_image_prefix: 本机可读取的图片根目录。

        输出：
        - None；属性写入当前实例。
        """
        self.sft_records = sft_records
        self.failed_records = failed_records
        self.sft_index = sft_index
        self.failed_index = failed_index
        self.remote_image_prefix = remote_image_prefix
        self.local_image_prefix = local_image_prefix


VIEWER_DATA: Optional[ViewerData] = None


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。

    输入：
    - 无显式入参；从 sys.argv 读取。

    输出：
    - argparse.Namespace: 包含数据路径、图片路径映射和 HTTP 服务监听地址。
    """
    parser = argparse.ArgumentParser(description="SFT / Failed dataset local web viewer")
    parser.add_argument("--sft_path", type=str, default=str(DEFAULT_SFT_PATH))
    parser.add_argument("--failed_path", type=str, default=str(DEFAULT_FAILED_PATH))
    parser.add_argument("--remote_image_prefix", type=str, default=DEFAULT_REMOTE_IMAGE_PREFIX)
    parser.add_argument("--local_image_prefix", type=str, default=DEFAULT_LOCAL_IMAGE_PREFIX)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8502, help="监听端口；设为 0 时由系统自动分配空闲端口")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    """
    将命令行传入路径解析为绝对路径。

    输入：
    - path_text: 绝对路径或相对项目根目录的路径。

    输出：
    - Path: 绝对路径。
    """
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def read_json_records(file_path: Path) -> List[Dict[str, Any]]:
    """
    读取 JSONL / JSON block 数据文件。

    输入：
    - file_path: 数据文件绝对路径。

    输出：
    - List[Dict[str, Any]]: 解析成功的 JSON 对象列表。

    兼容格式：
    - 标准单行 JSONL。
    - 多行 JSON block。
    - block 之间使用 `-----` 分隔。
    """
    if not file_path.exists():
        return []

    content = file_path.read_text(encoding="utf-8")
    records: List[Dict[str, Any]] = []
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
            obj, next_cursor = decoder.raw_decode(content, cursor)
        except json.JSONDecodeError:
            cursor += 1
            continue
        if isinstance(obj, dict):
            records.append(obj)
        cursor = next_cursor
    return records


def get_conversation_value(record: Dict[str, Any], role: str) -> str:
    """
    从 SFT conversations 中获取指定角色文本。

    输入：
    - record: SFT 样本字典。
    - role: 角色名，例如 human 或 gpt。

    输出：
    - str: 角色对应的 value；不存在时返回空字符串。
    """
    for item in record.get("conversations") or []:
        if isinstance(item, dict) and item.get("from") == role:
            return item.get("value") or ""
    return ""


def extract_event_line(text: str) -> str:
    """
    从 response 文本中提取 Event Recognition 字段。

    输入：
    - text: gpt.value、golden_response 或 vlm_response。

    输出：
    - str: Event Recognition 后面的内容；缺失时返回空字符串。
    """
    if not text:
        return ""
    match = re.search(r"^- Event Recognition:\s*(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def parse_sft_id(sample_id: str) -> Dict[str, str]:
    """
    从 SFT id 中解析 step 和 junction_id。

    输入：
    - sample_id: 形如 `场景-route-step_{sumo_step}-{junction_id}` 的字符串。

    输出：
    - Dict[str, str]: 包含 sumo_step 和 junction_id；解析失败时为空字符串。
    """
    match = re.search(r"-step_(?P<step>[^-]+)-(?P<junction>.+)$", sample_id or "")
    if not match:
        return {"sumo_step": "", "junction_id": ""}
    return {"sumo_step": match.group("step"), "junction_id": match.group("junction")}


def extract_route_from_images(image_paths: List[str]) -> Tuple[str, str]:
    """
    从图片路径推断 scenario 和 route。

    输入：
    - image_paths: SFT 的 image 列表或源样本 image_paths 列表。

    输出：
    - Tuple[str, str]: (scenario, route)。无法推断时返回空字符串。
    """
    if not image_paths:
        return "", ""
    normalized = image_paths[0].replace("\\", "/")
    if "/sft_dataset/" in normalized:
        relative_path = normalized.split("/sft_dataset/", 1)[1]
    elif normalized.startswith("data/sft_dataset/"):
        relative_path = normalized[len("data/sft_dataset/") :]
    else:
        return "", ""
    parts = relative_path.split("/")
    if len(parts) < 2:
        return "", ""
    return parts[0], parts[1]


def get_prompt_response(record: Dict[str, Any], mode: str) -> Tuple[str, str, str]:
    """
    获取样本中的 prompt、规范 response 和原始 response。

    输入：
    - record: SFT 或 failed 样本。
    - mode: `sft` 或 `failed`。

    输出：
    - Tuple[str, str, str]: (prompt, normalized_response, original_response)。
    """
    if mode == "sft":
        return get_conversation_value(record, "human"), get_conversation_value(record, "gpt"), ""

    sample = record.get("sample") or {}
    return (
        sample.get("prompt") or "",
        sample.get("golden_response") or "",
        sample.get("vlm_response") or sample.get("vlm_thought") or "",
    )


def get_image_paths(record: Dict[str, Any], mode: str) -> List[str]:
    """
    获取样本图片路径列表。

    输入：
    - record: SFT 或 failed 样本。
    - mode: `sft` 或 `failed`。

    输出：
    - List[str]: 图片路径列表。
    """
    if mode == "sft":
        return record.get("image") or []
    sample = record.get("sample") or {}
    return sample.get("image_paths") or []


def get_record_metadata(record: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """
    统一提取 SFT / failed 样本元信息。

    输入：
    - record: 样本字典。
    - mode: `sft` 或 `failed`。

    输出：
    - Dict[str, Any]: 包含 id / scenario / route / junction_id / sumo_step /
      fail_reason / filter_decision / has_golden_response。
    """
    if mode == "failed":
        return {
            "id": record.get("id") or "",
            "scenario": record.get("scenario") or "",
            "route": record.get("route") or "",
            "junction_id": record.get("junction_id") or "",
            "sumo_step": record.get("sumo_step") or "",
            "fail_reason": record.get("fail_reason") or "",
            "filter_decision": record.get("filter_decision") or "",
            "has_golden_response": bool(record.get("has_golden_response")),
        }

    image_paths = get_image_paths(record, mode)
    scenario, route = extract_route_from_images(image_paths)
    id_info = parse_sft_id(record.get("id") or "")
    return {
        "id": record.get("id") or "",
        "scenario": scenario,
        "route": route,
        "junction_id": id_info["junction_id"],
        "sumo_step": id_info["sumo_step"],
        "fail_reason": "",
        "filter_decision": "",
        "has_golden_response": True,
    }


def parse_location_pairs(location: str) -> List[Dict[str, str]]:
    """
    从 Location 文本中提取 Approach / LaneID。

    输入：
    - location: Event Recognition 中 `detected at` 后面的文本。

    输出：
    - List[Dict[str, str]]: 每个位置包含 approach / lane_id / lane_func。

    支持变体：
    - West L2
    - West Approach L2
    - West Approach, L2
    - Approach East L1
    - South L2(S)
    - North Approach L2 and South Approach L2
    """
    pairs: List[Dict[str, str]] = []
    for match in LOCATION_PAIR_RE.finditer(location or ""):
        approach = match.group("prefix_approach") or match.group("suffix_approach") or ""
        pairs.append(
            {
                "approach": approach.capitalize(),
                "lane_id": (match.group("lane_id") or "").upper(),
                "lane_func": match.group("lane_func") or "",
            }
        )
    return pairs


def parse_event_entries(event_line: str) -> List[Dict[str, Any]]:
    """
    将 Event Recognition 字段解析为结构化事件列表。

    输入：
    - event_line: Event Recognition 后面的内容，可以是 None 或多个分号分隔事件。

    输出：
    - List[Dict[str, Any]]: 每个事件位置一条记录，包含 specific_type / category /
      location / approach / lane_id / lane_func / phase / parse_status / raw。
    """
    if not event_line or event_line == "None":
        return []

    rows: List[Dict[str, Any]] = []
    for part in [item.strip() for item in event_line.split(";") if item.strip()]:
        match = EVENT_ENTRY_RE.match(part)
        if not match:
            rows.append(
                {
                    "specific_type": "",
                    "category": "",
                    "location": "",
                    "approach": "",
                    "lane_id": "",
                    "lane_func": "",
                    "phase": "",
                    "parse_status": "event_format_failed",
                    "raw": part,
                }
            )
            continue

        location = match.group("location")
        pairs = parse_location_pairs(location)
        if not pairs:
            rows.append(
                {
                    "specific_type": match.group("specific_type"),
                    "category": match.group("category"),
                    "location": location,
                    "approach": "",
                    "lane_id": "",
                    "lane_func": "",
                    "phase": match.group("phase"),
                    "parse_status": "location_failed",
                    "raw": part,
                }
            )
            continue

        for pair in pairs:
            rows.append(
                {
                    "specific_type": match.group("specific_type"),
                    "category": match.group("category"),
                    "location": location,
                    "approach": pair["approach"],
                    "lane_id": pair["lane_id"],
                    "lane_func": pair["lane_func"],
                    "phase": match.group("phase"),
                    "parse_status": "ok",
                    "raw": part,
                }
            )
    return rows


def build_index(records: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """
    构建前端筛选索引。

    输入：
    - records: 样本列表。
    - mode: `sft` 或 `failed`。

    输出：
    - List[Dict[str, Any]]: 每条记录包含基础元信息、事件类型、解析状态和搜索文本。
    """
    rows: List[Dict[str, Any]] = []
    for idx, record in enumerate(records):
        prompt, response, original_response = get_prompt_response(record, mode)
        event_line = extract_event_line(response or original_response)
        event_rows = parse_event_entries(event_line)
        event_types = sorted({row["specific_type"] for row in event_rows if row["specific_type"]})
        parse_statuses = sorted({row["parse_status"] for row in event_rows}) or ["none"]
        meta = get_record_metadata(record, mode)
        rows.append(
            {
                "idx": idx,
                "id": meta["id"],
                "scenario": meta["scenario"],
                "route": meta["route"],
                "junction_id": meta["junction_id"],
                "sumo_step": meta["sumo_step"],
                "fail_reason": meta["fail_reason"],
                "filter_decision": meta["filter_decision"],
                "has_golden_response": meta["has_golden_response"],
                "event_line": event_line,
                "event_types": event_types,
                "event_type_text": ", ".join(event_types) if event_types else "None",
                "event_parse_status": ", ".join(parse_statuses),
                "search_text": " ".join(
                    [
                        str(meta["id"]),
                        str(meta["scenario"]),
                        str(meta["route"]),
                        str(meta["junction_id"]),
                        str(meta["sumo_step"]),
                        str(meta["fail_reason"]),
                        event_line,
                        prompt[:1000],
                        response[:1000],
                        original_response[:1000],
                    ]
                ).lower(),
            }
        )
    return rows


def replace_image_prefix(path: str, remote_prefix: str, local_prefix: str) -> str:
    """
    将数据中的图片路径映射成本机文件路径。

    输入：
    - path: 数据中的图片路径。
    - remote_prefix: 远端路径前缀，例如 /root/autodl-tmp/sft_dataset。
    - local_prefix: 本机路径前缀，例如 /path/to/project/data/sft_dataset。

    输出：
    - str: 本机候选路径；如果无法映射则返回原路径。
    """
    if not path:
        return ""

    if remote_prefix and path.startswith(remote_prefix):
        suffix = path[len(remote_prefix) :].lstrip("/")
        return str(Path(local_prefix) / suffix)

    if os.path.exists(path):
        return path

    marker = "/data/sft_dataset/"
    if marker in path:
        suffix = path.split(marker, 1)[1]
        return str(PROJECT_ROOT / "data" / "sft_dataset" / suffix)

    if path.startswith("data/sft_dataset/"):
        return str(PROJECT_ROOT / path)

    return path


def extract_dir_from_path(path: str) -> Optional[str]:
    """
    从图片文件名中识别方向。

    输入：
    - path: 图片路径。

    输出：
    - Optional[str]: N / E / S / W；无法识别时返回 None。
    """
    stem = os.path.splitext(os.path.basename(path or ""))[0]
    match = re.search(r"_([NESW])(?:_no_watermark)?$", stem, re.IGNORECASE)
    return match.group(1).upper() if match else None


def build_dir_image_map(image_paths: List[str]) -> Dict[str, Optional[str]]:
    """
    将图片列表映射到 N/E/S/W 四个方向。

    输入：
    - image_paths: 图片路径列表。

    输出：
    - Dict[str, Optional[str]]: 方向到图片路径的映射；缺失方向为 None。
    """
    dir_map: Dict[str, Optional[str]] = {"N": None, "E": None, "S": None, "W": None}
    unresolved: List[Tuple[int, str]] = []
    for idx, path in enumerate(image_paths or []):
        direction = extract_dir_from_path(path)
        if direction in dir_map and dir_map[direction] is None:
            dir_map[direction] = path
        else:
            unresolved.append((idx, path))

    legacy_order = ["N", "E", "S", "W"]
    for idx, path in unresolved:
        if idx < len(legacy_order) and dir_map[legacy_order[idx]] is None:
            dir_map[legacy_order[idx]] = path
    return dir_map


def json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    """
    向前端返回 JSON 响应。

    输入：
    - handler: 当前 HTTP request handler。
    - payload: 可 JSON 序列化的数据。
    - status: HTTP 状态码。

    输出：
    - None；直接写入 HTTP 响应。
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def text_response(handler: BaseHTTPRequestHandler, text: str, content_type: str, status: int = 200) -> None:
    """
    向前端返回文本响应。

    输入：
    - handler: 当前 HTTP request handler。
    - text: 响应正文。
    - content_type: Content-Type 值。
    - status: HTTP 状态码。

    输出：
    - None；直接写入 HTTP 响应。
    """
    body = text.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def get_records_and_index(mode: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    根据 mode 获取原始记录和筛选索引。

    输入：
    - mode: `sft` 或 `failed`。

    输出：
    - Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: (records, index_rows)。
    """
    if VIEWER_DATA is None:
        return [], []
    if mode == "failed":
        return VIEWER_DATA.failed_records, VIEWER_DATA.failed_index
    return VIEWER_DATA.sft_records, VIEWER_DATA.sft_index


def build_record_payload(mode: str, idx: int) -> Dict[str, Any]:
    """
    构造单条样本详情接口返回值。

    输入：
    - mode: `sft` 或 `failed`。
    - idx: 原始 records 中的样本索引。

    输出：
    - Dict[str, Any]: 包含 record、metadata、prompt、response、image_paths 和事件解析结果。
    """
    records, index_rows = get_records_and_index(mode)
    if idx < 0 or idx >= len(records):
        raise IndexError("record index out of range")
    record = records[idx]
    prompt, response, original_response = get_prompt_response(record, mode)
    event_line = extract_event_line(response or original_response)
    return {
        "record": record,
        "metadata": get_record_metadata(record, mode),
        "index": next((row for row in index_rows if row["idx"] == idx), {}),
        "prompt": prompt,
        "response": response,
        "original_response": original_response,
        "event_line": event_line,
        "event_rows": parse_event_entries(event_line),
        "image_paths": get_image_paths(record, mode),
    }


def serve_image(handler: BaseHTTPRequestHandler, query: Dict[str, List[str]]) -> None:
    """
    根据图片路径查询参数读取并返回图片文件。

    输入：
    - handler: 当前 HTTP request handler。
    - query: URL query 参数，必须包含 path。

    输出：
    - None；成功时返回图片二进制，失败时返回 404 JSON。
    """
    if VIEWER_DATA is None:
        json_response(handler, {"error": "viewer data not loaded"}, HTTPStatus.INTERNAL_SERVER_ERROR)
        return

    raw_path = unquote(query.get("path", [""])[0])
    local_path = replace_image_prefix(
        raw_path,
        VIEWER_DATA.remote_image_prefix,
        VIEWER_DATA.local_image_prefix,
    )
    if not local_path or not os.path.exists(local_path):
        json_response(handler, {"error": "image not found", "path": local_path}, HTTPStatus.NOT_FOUND)
        return

    content_type = mimetypes.guess_type(local_path)[0] or "application/octet-stream"
    with open(local_path, "rb") as file:
        body = file.read()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class ViewerHandler(BaseHTTPRequestHandler):
    """
    本地查看器 HTTP handler。

    输入：
    - 浏览器发起的 HTTP GET 请求。

    输出：
    - `/`: HTML 前端。
    - `/api/index?mode=sft|failed`: 筛选索引。
    - `/api/record?mode=sft|failed&idx=N`: 单条样本详情。
    - `/image?path=...`: 图片二进制。
    """

    def log_message(self, format: str, *args: Any) -> None:
        """
        控制 HTTP 访问日志输出。

        输入：
        - format: BaseHTTPRequestHandler 传入的日志格式。
        - args: 日志参数。

        输出：
        - None。这里保持默认简洁输出，方便发现前端请求错误。
        """
        super().log_message(format, *args)

    def do_GET(self) -> None:
        """
        处理 GET 请求。

        输入：
        - 无显式入参；从 self.path 读取请求路径和 query。

        输出：
        - None；根据路径写入 HTTP 响应。
        """
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)

        if parsed.path == "/":
            text_response(self, HTML_APP, "text/html; charset=utf-8")
            return

        if parsed.path == "/api/index":
            mode = query.get("mode", ["sft"])[0]
            _, index_rows = get_records_and_index(mode)
            json_response(self, {"rows": index_rows})
            return

        if parsed.path == "/api/record":
            mode = query.get("mode", ["sft"])[0]
            idx = int(query.get("idx", ["0"])[0])
            try:
                json_response(self, build_record_payload(mode, idx))
            except Exception as exc:
                json_response(self, {"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        if parsed.path == "/image":
            serve_image(self, query)
            return

        json_response(self, {"error": "not found"}, HTTPStatus.NOT_FOUND)


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """
    允许快速复用地址的 HTTP Server。

    输入：
    - 初始化参数与 ThreadingHTTPServer 一致。

    输出：
    - HTTP Server 实例。

    作用：
    - allow_reuse_address=True 可以减少程序刚退出后端口仍处于 TIME_WAIT 导致的绑定失败。
    - 如果端口被其他正在运行的进程占用，仍会抛出 Address already in use，
      该情况由 create_http_server() 继续尝试后续端口。
    """

    allow_reuse_address = True


def create_http_server(host: str, port: int) -> ReusableThreadingHTTPServer:
    """
    创建 HTTP Server，并在端口占用时自动回退到其他端口。

    输入：
    - host: 监听地址，例如 127.0.0.1。
    - port: 首选监听端口。传入 0 时由系统直接分配空闲端口。

    输出：
    - ReusableThreadingHTTPServer: 已成功绑定端口的 server 实例。

    端口选择策略：
    - port == 0: 直接交给系统分配空闲端口。
    - port > 0: 先尝试指定端口；若被占用，则依次尝试 port+1 到 port+50。
    - 如果连续 51 个端口都被占用，最后退回 port=0 让系统分配。
    """
    if port == 0:
        return ReusableThreadingHTTPServer((host, 0), ViewerHandler)

    last_error: Optional[OSError] = None
    for candidate_port in range(port, port + 51):
        try:
            return ReusableThreadingHTTPServer((host, candidate_port), ViewerHandler)
        except OSError as exc:
            if exc.errno not in (48, 98):
                raise
            last_error = exc
            if candidate_port == port:
                print(f"Port {port} is already in use; trying the next available port...")

    if last_error is not None:
        print(f"Ports {port}-{port + 50} are unavailable; falling back to an OS-assigned port.")
    return ReusableThreadingHTTPServer((host, 0), ViewerHandler)


HTML_APP = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SFT Dataset Viewer</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --border: #d7dce2;
      --accent: #2364aa;
      --accent-soft: #e8f1fb;
      --bad: #b42318;
      --ok: #067647;
      --warn: #b54708;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    .layout { display: grid; grid-template-columns: 340px 1fr; min-height: 100vh; }
    aside {
      background: var(--panel);
      border-right: 1px solid var(--border);
      padding: 18px;
      overflow: auto;
      position: sticky;
      top: 0;
      height: 100vh;
    }
    main { padding: 20px; overflow: auto; }
    h1 { font-size: 21px; margin: 0 0 18px; }
    h2 { font-size: 18px; margin: 0 0 12px; }
    h3 { font-size: 15px; margin: 0 0 10px; }
    label { display: block; font-size: 12px; color: var(--muted); margin: 12px 0 5px; }
    input, select, button, textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    input, select { height: 34px; padding: 6px 8px; }
    button {
      height: 34px;
      cursor: pointer;
      background: #fff;
    }
    button.primary {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    textarea {
      min-height: 360px;
      padding: 10px;
      resize: vertical;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
      line-height: 1.45;
    }
    pre {
      margin: 0;
      padding: 12px;
      overflow: auto;
      background: #111827;
      color: #f9fafb;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.45;
      max-height: 520px;
    }
    .muted { color: var(--muted); }
    .divider { height: 1px; background: var(--border); margin: 16px 0; }
    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    .summary { display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin-bottom: 16px; }
    .metric {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px;
      min-width: 0;
    }
    .metric .name { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .metric .value { font-weight: 650; overflow-wrap: anywhere; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 14px;
      margin-bottom: 16px;
    }
    .nav { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 10px 0; }
    .image-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .image-card {
      border: 1px solid var(--border);
      border-radius: 6px;
      min-height: 170px;
      background: #fafafa;
      overflow: hidden;
    }
    .image-card .caption {
      padding: 6px 8px;
      font-size: 12px;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      overflow-wrap: anywhere;
    }
    .image-card img {
      width: 100%;
      height: 280px;
      object-fit: contain;
      display: block;
      background: #f3f4f6;
    }
    .missing { padding: 18px; color: var(--bad); font-size: 13px; }
    .tabs { display: flex; gap: 8px; margin-bottom: 10px; }
    .tabs button { width: auto; padding: 0 12px; }
    .tabs button.active { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border-bottom: 1px solid var(--border);
      padding: 7px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th { color: var(--muted); font-weight: 600; background: #fafafa; }
    .status-ok { color: var(--ok); font-weight: 600; }
    .status-bad { color: var(--bad); font-weight: 600; }
    .notice {
      padding: 10px;
      border-radius: 6px;
      background: #fff7ed;
      border: 1px solid #fed7aa;
      color: var(--warn);
      margin-bottom: 12px;
      overflow-wrap: anywhere;
    }
    .record-list {
      max-height: 270px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #fff;
    }
    .record-item {
      padding: 8px;
      border-bottom: 1px solid var(--border);
      cursor: pointer;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .record-item.active { background: var(--accent-soft); }
    @media (max-width: 1100px) {
      .layout { grid-template-columns: 1fr; }
      aside { position: static; height: auto; }
      .row, .summary { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <h1>SFT Dataset Viewer</h1>
      <label>Dataset</label>
      <select id="mode">
        <option value="sft">SFT</option>
        <option value="failed">Failed</option>
      </select>

      <div class="divider"></div>
      <h3>Filters</h3>
      <label>Search</label>
      <input id="search" placeholder="id / route / junction / event">
      <label>Scenario</label>
      <select id="scenario"></select>
      <label>Route</label>
      <select id="route"></select>
      <label>Event Type</label>
      <select id="eventType"></select>
      <label>Event Parse Status</label>
      <select id="parseStatus"></select>
      <label>Fail Reason</label>
      <select id="failReason"></select>

      <div class="divider"></div>
      <h3>Navigation</h3>
      <div class="nav">
        <button id="prev">Prev</button>
        <button id="next" class="primary">Next</button>
      </div>
      <label>Jump to filtered index</label>
      <input id="jump" type="number" min="1" value="1">
      <p class="muted" id="countText"></p>

      <div class="divider"></div>
      <h3>Results</h3>
      <div class="record-list" id="recordList"></div>
    </aside>

    <main>
      <div id="failedNotice"></div>
      <h2 id="title">Loading...</h2>
      <div class="summary" id="summary"></div>
      <div class="row">
        <section class="panel">
          <h3>Images</h3>
          <div class="image-grid" id="images"></div>
        </section>
        <section class="panel">
          <h3>Event Recognition</h3>
          <pre id="eventLine"></pre>
          <div id="eventTable"></div>
        </section>
      </div>

      <section class="panel">
        <div class="tabs">
          <button data-tab="prompt" class="active">Human Prompt</button>
          <button data-tab="response">GPT / Response</button>
          <button data-tab="raw">Raw JSON</button>
        </div>
        <div id="tabPrompt"><textarea id="promptText" readonly></textarea></div>
        <div id="tabResponse" style="display:none"><textarea id="responseText" readonly></textarea></div>
        <div id="tabRaw" style="display:none"><pre id="rawJson"></pre></div>
      </section>
    </main>
  </div>

<script>
const state = {
  mode: 'sft',
  rows: [],
  filtered: [],
  cursor: 0,
  current: null,
};

const els = {
  mode: document.getElementById('mode'),
  search: document.getElementById('search'),
  scenario: document.getElementById('scenario'),
  route: document.getElementById('route'),
  eventType: document.getElementById('eventType'),
  parseStatus: document.getElementById('parseStatus'),
  failReason: document.getElementById('failReason'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
  jump: document.getElementById('jump'),
  countText: document.getElementById('countText'),
  recordList: document.getElementById('recordList'),
  failedNotice: document.getElementById('failedNotice'),
  title: document.getElementById('title'),
  summary: document.getElementById('summary'),
  images: document.getElementById('images'),
  eventLine: document.getElementById('eventLine'),
  eventTable: document.getElementById('eventTable'),
  promptText: document.getElementById('promptText'),
  responseText: document.getElementById('responseText'),
  rawJson: document.getElementById('rawJson'),
};

function uniq(values) {
  return [...new Set(values.filter(v => v !== undefined && v !== null && String(v).length > 0))].sort();
}

function optionHtml(values, includeAll=true) {
  const opts = includeAll ? ['All', ...values] : values;
  return opts.map(v => `<option value="${escapeHtml(v)}">${escapeHtml(v)}</option>`).join('');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
  }[ch]));
}

function setSelectOptions(select, values, current='All') {
  select.innerHTML = optionHtml(values);
  if ([...select.options].some(opt => opt.value === current)) {
    select.value = current;
  }
}

async function loadIndex() {
  const res = await fetch(`/api/index?mode=${state.mode}`);
  const data = await res.json();
  state.rows = data.rows || [];
  fillFilters();
  applyFilters();
}

function fillFilters() {
  setSelectOptions(els.scenario, uniq(state.rows.map(r => r.scenario)), els.scenario.value || 'All');
  setSelectOptions(els.route, uniq(state.rows.map(r => r.route)), els.route.value || 'All');
  const events = uniq(state.rows.flatMap(r => r.event_types || []));
  els.eventType.innerHTML = optionHtml(['None', ...events]);
  if (![...els.eventType.options].some(opt => opt.value === els.eventType.value)) els.eventType.value = 'All';
  setSelectOptions(els.parseStatus, ['ok', 'location_failed', 'event_format_failed', 'none'], els.parseStatus.value || 'All');
  setSelectOptions(els.failReason, uniq(state.rows.map(r => r.fail_reason)), els.failReason.value || 'All');
  els.failReason.disabled = state.mode !== 'failed';
}

function applyFilters() {
  const q = els.search.value.trim().toLowerCase();
  state.filtered = state.rows.filter(row => {
    if (q && !(row.search_text || '').includes(q)) return false;
    if (els.scenario.value !== 'All' && row.scenario !== els.scenario.value) return false;
    if (els.route.value !== 'All' && row.route !== els.route.value) return false;
    if (els.eventType.value === 'None' && (row.event_types || []).length > 0) return false;
    if (els.eventType.value !== 'All' && els.eventType.value !== 'None' && !(row.event_types || []).includes(els.eventType.value)) return false;
    if (els.parseStatus.value !== 'All' && !(row.event_parse_status || '').split(', ').includes(els.parseStatus.value)) return false;
    if (state.mode === 'failed' && els.failReason.value !== 'All' && row.fail_reason !== els.failReason.value) return false;
    return true;
  });
  state.cursor = Math.min(state.cursor, Math.max(0, state.filtered.length - 1));
  renderList();
  loadCurrent();
}

function renderList() {
  els.countText.textContent = `${state.filtered.length} filtered / ${state.rows.length} total`;
  els.jump.max = Math.max(1, state.filtered.length);
  els.jump.value = state.filtered.length ? state.cursor + 1 : 1;
  const items = state.filtered.slice(0, 300).map((row, i) => {
    const active = i === state.cursor ? ' active' : '';
    return `<div class="record-item${active}" data-pos="${i}">
      <strong>${i + 1}.</strong> ${escapeHtml(row.id || '(no id)')}<br>
      <span class="muted">${escapeHtml(row.route)} | ${escapeHtml(row.event_type_text)} | ${escapeHtml(row.event_parse_status)}</span>
    </div>`;
  }).join('');
  els.recordList.innerHTML = items || '<div class="record-item">No records</div>';
  els.recordList.querySelectorAll('.record-item[data-pos]').forEach(el => {
    el.addEventListener('click', () => {
      state.cursor = Number(el.dataset.pos);
      renderList();
      loadCurrent();
    });
  });
}

async function loadCurrent() {
  if (!state.filtered.length) {
    els.title.textContent = 'No matching records';
    return;
  }
  const row = state.filtered[state.cursor];
  const res = await fetch(`/api/record?mode=${state.mode}&idx=${row.idx}`);
  state.current = await res.json();
  renderCurrent();
}

function metric(name, value) {
  return `<div class="metric"><div class="name">${escapeHtml(name)}</div><div class="value">${escapeHtml(value || '-')}</div></div>`;
}

function renderCurrent() {
  const payload = state.current;
  const meta = payload.metadata || {};
  const index = payload.index || {};
  els.title.textContent = meta.id || '(no id)';
  els.failedNotice.innerHTML = state.mode === 'failed'
    ? `<div class="notice">Fail reason: ${escapeHtml(meta.fail_reason || '-')} | filter_decision: ${escapeHtml(meta.filter_decision || '-')} | has_golden_response: ${escapeHtml(meta.has_golden_response)}</div>`
    : '';
  els.summary.innerHTML = [
    metric('Scenario', meta.scenario),
    metric('Route', meta.route),
    metric('Junction', meta.junction_id),
    metric('Step', meta.sumo_step),
    metric('Event', index.event_type_text),
    metric('Parse', index.event_parse_status),
  ].join('');
  renderImages(payload.image_paths || []);
  els.eventLine.textContent = payload.event_line || 'None';
  renderEventTable(payload.event_rows || []);
  els.promptText.value = payload.prompt || '';
  els.responseText.value = payload.response || payload.original_response || '';
  els.rawJson.textContent = JSON.stringify(payload.record, null, 2);
}

function dirFromPath(path) {
  const name = String(path || '').split('/').pop().replace(/\.[^.]+$/, '');
  const m = name.match(/_([NESW])(?:_no_watermark)?$/i);
  return m ? m[1].toUpperCase() : null;
}

function renderImages(paths) {
  const order = ['N', 'E', 'S', 'W'];
  const labels = {N: 'North', E: 'East', S: 'South', W: 'West'};
  const map = {N: null, E: null, S: null, W: null};
  const unresolved = [];
  paths.forEach((p, idx) => {
    const d = dirFromPath(p);
    if (d && !map[d]) map[d] = p;
    else unresolved.push([idx, p]);
  });
  unresolved.forEach(([idx, p]) => {
    const d = order[idx];
    if (d && !map[d]) map[d] = p;
  });
  els.images.innerHTML = order.map(d => {
    const p = map[d];
    if (!p) return `<div class="image-card"><div class="caption">${labels[d]}</div><div class="missing">No image</div></div>`;
    const url = `/image?path=${encodeURIComponent(p)}`;
    return `<div class="image-card"><div class="caption">${labels[d]} | ${escapeHtml(p)}</div><img src="${url}" alt="${labels[d]}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'missing',textContent:'Image not found'}))"></div>`;
  }).join('');
}

function renderEventTable(rows) {
  if (!rows.length) {
    els.eventTable.innerHTML = '<p class="muted">No event rows.</p>';
    return;
  }
  const body = rows.map(row => {
    const cls = row.parse_status === 'ok' ? 'status-ok' : 'status-bad';
    return `<tr>
      <td>${escapeHtml(row.specific_type)}</td>
      <td>${escapeHtml(row.category)}</td>
      <td>${escapeHtml(row.approach)}</td>
      <td>${escapeHtml(row.lane_id)}</td>
      <td>${escapeHtml(row.lane_func)}</td>
      <td>${escapeHtml(row.phase)}</td>
      <td class="${cls}">${escapeHtml(row.parse_status)}</td>
      <td>${escapeHtml(row.raw)}</td>
    </tr>`;
  }).join('');
  els.eventTable.innerHTML = `<table>
    <thead><tr><th>Specific Type</th><th>Category</th><th>Approach</th><th>LaneID</th><th>Func</th><th>Phase</th><th>Status</th><th>Raw</th></tr></thead>
    <tbody>${body}</tbody>
  </table>`;
}

function switchTab(name) {
  document.querySelectorAll('.tabs button').forEach(btn => btn.classList.toggle('active', btn.dataset.tab === name));
  document.getElementById('tabPrompt').style.display = name === 'prompt' ? '' : 'none';
  document.getElementById('tabResponse').style.display = name === 'response' ? '' : 'none';
  document.getElementById('tabRaw').style.display = name === 'raw' ? '' : 'none';
}

els.mode.addEventListener('change', () => {
  state.mode = els.mode.value;
  state.cursor = 0;
  loadIndex();
});
[els.search, els.scenario, els.route, els.eventType, els.parseStatus, els.failReason].forEach(el => {
  el.addEventListener('input', () => { state.cursor = 0; applyFilters(); });
  el.addEventListener('change', () => { state.cursor = 0; applyFilters(); });
});
els.prev.addEventListener('click', () => {
  state.cursor = Math.max(0, state.cursor - 1);
  renderList();
  loadCurrent();
});
els.next.addEventListener('click', () => {
  state.cursor = Math.min(Math.max(0, state.filtered.length - 1), state.cursor + 1);
  renderList();
  loadCurrent();
});
els.jump.addEventListener('change', () => {
  const target = Math.max(1, Math.min(Number(els.jump.value || 1), state.filtered.length));
  state.cursor = target - 1;
  renderList();
  loadCurrent();
});
document.querySelectorAll('.tabs button').forEach(btn => btn.addEventListener('click', () => switchTab(btn.dataset.tab)));

loadIndex();
</script>
</body>
</html>
"""


def load_viewer_data(args: argparse.Namespace) -> ViewerData:
    """
    加载数据文件并构建全局查看器数据。

    输入：
    - args: 命令行参数对象。

    输出：
    - ViewerData: 包含 SFT/failed 原始样本和筛选索引。
    """
    sft_records = read_json_records(resolve_path(args.sft_path))
    failed_records = read_json_records(resolve_path(args.failed_path))
    local_prefix = str(resolve_path(args.local_image_prefix))
    return ViewerData(
        sft_records=sft_records,
        failed_records=failed_records,
        sft_index=build_index(sft_records, "sft"),
        failed_index=build_index(failed_records, "failed"),
        remote_image_prefix=args.remote_image_prefix.rstrip("/"),
        local_image_prefix=local_prefix,
    )


def main() -> None:
    """
    脚本入口。

    输入：
    - 无显式入参；从命令行读取配置。

    输出：
    - None；启动本地 HTTP 服务并阻塞运行。
    """
    global VIEWER_DATA

    args = parse_args()
    VIEWER_DATA = load_viewer_data(args)

    server = create_http_server(args.host, args.port)
    actual_host, actual_port = server.server_address[:2]
    display_host = args.host if args.host not in ("0.0.0.0", "") else actual_host
    url = f"http://{display_host}:{actual_port}"
    print(f"SFT records: {len(VIEWER_DATA.sft_records)}")
    print(f"Failed records: {len(VIEWER_DATA.failed_records)}")
    print(f"Viewer URL: {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
