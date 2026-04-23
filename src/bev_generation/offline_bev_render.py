'''
Author: yufei Ji
Description: 离线 BEV 渲染脚本。
             从 run_eval.py 保存的 render_info/render_{t}.json 读取仿真状态快照，
             调用 TSHubRenderer 离线重渲染并将图像写回同级目录下：
               render_info/{sumo_t}/  ← 渲染结果子目录
                 {sensor_id}.png

             支持两种运行模式：
               1. CLI 运行（推荐）：
                  python src/bev_generation/offline_bev_render.py \
                      --scenario_key JiNan \
                      --json_files data/eval/JiNan/anon_3_4_jinan_real/qwen3-vl-4b/render_info/render_28.json

               2. 本地 Debug（直接 python offline_bev_render.py，修改下方 DEBUG_* 变量）

             render_info/render_{t}.json 由 run_eval.py 在每次 VLM 决策前保存，
             包含 vehicle / tls / aircraft / person 四个顶层键。
'''

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET

import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Debug 模式配置（仅在直接 python offline_bev_render.py 时生效）
# 每个任务都需要一一对应：
# - scenario_key
# - json_files 或 json_glob（二选一）
# - route_xml（可选，不填则自动推断）
# ──────────────────────────────────────────────────────────────────────────────
DEBUG_RENDER_TASKS = [
    # {
    #     "scenario_key": "France_Massy",
    #     "json_files": [
    #         "data/test/prompt_test/France_Massy/render_540.json",
    #     ],
    #     "json_glob": None,
    #     "route_xml": "data/raw/France_Massy/env/massy_accident.rou.xml",
    # },
    # {
    #     "scenario_key": "JiNan",
    #     "json_files": [
    #         "data/test/prompt_test/JiNan/render_1000.json",
    #     ],
    #     "json_glob": None,
    #     "route_xml": "data/raw/JiNan/env/anon_3_4_jinan_real_accident.rou.xml",
    # },
    # {
    #     "scenario_key": "SouthKorea_Songdo",
    #     "json_files": [
    #         "data/test/prompt_test/SouthKorea_Songdo/render_540.json",
    #     ],
    #     "json_glob": None,
    #     "route_xml": "data/raw/SouthKorea_Songdo/env/songdo_accident.rou.xml",
    # },
        {
        "scenario_key": "Hongkong_YMT",
        "json_files": [
            "data/test/prompt_test/Hongkong_YMT/render_390.json",
        ],
        "json_glob": None,
        "route_xml": "data/raw/Hongkong_YMT/env/YMT_debris.rou.xml",
    },
    #     {
    #     "scenario_key": "Hangzhou",
    #     "json_files": [
    #         "data/test/prompt_test/Hangzhou/render_900.json",
    #     ],
    #     "json_glob": None,
    #     "route_xml": "data/raw/Hangzhou/env/anon_4_4_hangzhou_real_emergy.rou.xml",
    # },
    #     {
    #     "scenario_key": "Hangzhou",
    #     "json_files": [
    #         "data/test/prompt_test/Hangzhou/render_1200.json",
    #     ],
    #     "json_glob": None,
    #     "route_xml": "data/raw/Hangzhou/env/anon_4_4_hangzhou_real_bus.rou.xml",
    # },
]


# ──────────────────────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────────────────────

def _extract_step(filepath: str) -> int:
    """从 render_{t}.json 文件名中提取仿真时间步。"""
    base = os.path.basename(filepath)          # render_28.json
    stem = base.replace("render_", "").replace(".json", "")   # 28
    return int(stem)


def _save_folder_for(json_path: str, step: int) -> str:
    """
    计算图像输出目录。
    JSON 路径: .../render_info/render_{t}.json
    输出目录:  .../render_info/{t}/
    """
    render_info_dir = os.path.dirname(os.path.abspath(json_path))
    return os.path.join(render_info_dir, str(step))


def _get_project_root() -> str:
    return os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )


def _normalize_render_task(
    scenario_key: str,
    json_files: list[str] | None = None,
    json_glob: str | None = None,
    route_xml_path: str | None = None,
) -> dict:
    """统一渲染任务结构，并解析 json_glob。"""
    if json_files and json_glob:
        raise ValueError("json_files 与 json_glob 只能二选一。")
    if not json_files and not json_glob:
        raise ValueError("必须提供 json_files 或 json_glob 之一。")

    if json_glob:
        resolved_json_files = sorted(glob.glob(json_glob))
        if not resolved_json_files:
            raise FileNotFoundError(f"json_glob 无匹配文件: {json_glob}")
    else:
        resolved_json_files = json_files or []

    return {
        "scenario_key": scenario_key,
        "json_files": resolved_json_files,
        "route_xml_path": route_xml_path,
    }


def _collect_debug_render_tasks() -> list[dict]:
    """收集并校验 Debug 模式下的批量渲染任务。"""
    if not DEBUG_RENDER_TASKS:
        raise ValueError("DEBUG_RENDER_TASKS 为空，请至少配置一个调试渲染任务。")

    normalized_tasks = []
    for idx, task in enumerate(DEBUG_RENDER_TASKS, start=1):
        if "scenario_key" not in task:
            raise KeyError(f"DEBUG_RENDER_TASKS 第 {idx} 项缺少 scenario_key")

        normalized_tasks.append(
            _normalize_render_task(
                scenario_key=task["scenario_key"],
                json_files=task.get("json_files"),
                json_glob=task.get("json_glob"),
                route_xml_path=task.get("route_xml"),
            )
        )

    return normalized_tasks


def _read_route_file_from_sumocfg(sumocfg_path: str) -> str | None:
    """从 sumocfg 的 <route-files> 节点解析首个路由文件绝对路径。"""
    try:
        tree = ET.parse(sumocfg_path)
        root = tree.getroot()
        node = root.find(".//route-files")
        if node is None:
            return None

        rel_route = node.get("value", "").split(",")[0].strip()
        if not rel_route:
            return None

        sumocfg_dir = os.path.dirname(os.path.abspath(sumocfg_path))
        return os.path.normpath(os.path.join(sumocfg_dir, rel_route))
    except Exception:
        return None


def _resolve_route_xml_path(
    scenario_key: str,
    scenario_name: str,
    netfile: str,
    json_path: str,
) -> str | None:
    """
    为离线渲染解析对应的 .rou.xml。

    优先级：
      1. 从 json 路径中匹配 data/raw/{scenario}/env 下已有的 route stem
      2. 回退到 {NETFILE}.sumocfg 中默认的 <route-files> 配置
    """
    project_root = _get_project_root()
    scenario_dir = os.path.join(project_root, "data", "raw", scenario_name)
    env_dir = os.path.join(scenario_dir, "env")

    route_candidates = {}
    if os.path.isdir(env_dir):
        for route_path in glob.glob(os.path.join(env_dir, "*.rou.xml")):
            stem = os.path.basename(route_path)
            if stem.endswith(".rou.xml"):
                stem = stem[:-8]
            route_candidates[stem] = route_path

    json_parts = set(os.path.normpath(os.path.abspath(json_path)).split(os.sep))
    for stem in sorted(route_candidates.keys(), key=len, reverse=True):
        if stem in json_parts:
            return route_candidates[stem]

    sumocfg_path = os.path.join(scenario_dir, f"{netfile}.sumocfg")
    if os.path.exists(sumocfg_path):
        return _read_route_file_from_sumocfg(sumocfg_path)

    print(
        f"[OfflineRender] 未能为场景 {scenario_key} 解析 sumocfg: {sumocfg_path}"
    )
    return None


class OfflineEventManager:
    """离线解析路由文件中的事件，并按仿真时间步筛选活跃事件。"""

    def __init__(self, project_root: str):
        self.project_root = project_root
        self.active_events = []

    def load_events_from_xml(self, route_xml_path: str) -> list[dict]:
        self.active_events = []
        if not route_xml_path or not os.path.exists(route_xml_path):
            print(f"[OfflineRender] 事件路由文件不存在: {route_xml_path}")
            return self.active_events

        try:
            tree = ET.parse(route_xml_path)
            root = tree.getroot()
            for trip in root.findall("trip"):
                params = {p.get("key"): p.get("value") for p in trip.findall("param")}
                if "pos_x" not in params or "pos_y" not in params:
                    continue

                stop = trip.find("stop")
                duration = float(stop.get("duration", 0)) if stop is not None else 0.0
                model_path = params.get("model_path", "")
                if model_path and not os.path.isabs(model_path):
                    model_path = os.path.normpath(
                        os.path.join(self.project_root, model_path)
                    )

                self.active_events.append(
                    {
                        "id": trip.get("id"),
                        "type": params.get("event_type", "unknown"),
                        "x": float(params["pos_x"]),
                        "y": float(params["pos_y"]),
                        "heading": float(params.get("heading", 0.0)),
                        "start_time": float(trip.get("depart", 0)),
                        "end_time": float(trip.get("depart", 0)) + duration,
                        "model_path": model_path,
                    }
                )
        except Exception as e:
            print(f"[OfflineRender] 解析事件路由失败 {route_xml_path}: {e}")
            self.active_events = []

        print(
            f"[OfflineRender] 从 {os.path.basename(route_xml_path)} 加载事件 {len(self.active_events)} 个"
        )
        return self.active_events

    def get_active_events(self, current_time: float) -> list[dict]:
        return [
            event
            for event in self.active_events
            if event["start_time"] <= current_time <= event["end_time"]
        ]


# ──────────────────────────────────────────────────────────────────────────────
# 核心渲染类
# ──────────────────────────────────────────────────────────────────────────────

class OfflineBEVGenerator:
    """
    封装 TSHubRenderer 的离线渲染器。
    同一个 Renderer 实例可连续处理多帧，TSHubRenderer.reset() 只在第一帧调用一次。
    """

    def __init__(
        self,
        scenario_glb_dir: str,
        sensor_cfg: dict,
        renderer_cfg: dict,
        route_xml_path: str | None = None,
    ):
        try:
            from tshub.tshub_env3d.vis3d_renderer.tshub_render import TSHubRenderer
            from tshub.tshub_env3d.vis3d_renderer.emergency.emergency_manager import (
                EmergencyManager3D,
            )
        except ImportError as e:
            raise ImportError(f"无法导入 TSHubRenderer，请确认 TransSimHub 已安装: {e}")

        self.renderer = TSHubRenderer(
            simid="offline_renderer",
            scenario_glb_dir=scenario_glb_dir,
            sensor_config=sensor_cfg,
            preset=renderer_cfg.get("preset", "720P"),
            resolution=renderer_cfg.get("resolution", 1.0),
            vehicle_model=renderer_cfg.get("vehicle_model", "low"),
            render_mode=renderer_cfg.get("render_mode", "offscreen"),
        )
        self.event_logic_manager = None
        self.emergency_renderer = EmergencyManager3D(
            self.renderer._showbase_instance,
            self.renderer._root_np,
            show_closure_zone=renderer_cfg.get("show_closure_zone", True),
        )
        if route_xml_path:
            self.event_logic_manager = OfflineEventManager(_get_project_root())
            self.event_logic_manager.load_events_from_xml(route_xml_path)
        self._reset_done = False

    # ------------------------------------------------------------------
    def process_state(self, full_data: dict, save_folder: str, current_time: float) -> dict:
        """
        渲染单帧状态并将图像保存到 save_folder。

        Args:
            full_data:   render_{t}.json 的完整内容（含 vehicle/tls/aircraft/person）
            save_folder: 图像输出目录（会自动创建）
            current_time: 当前仿真时间步，用于激活对应事件

        Returns:
            sensor_data: TSHubRenderer.step() 的原始返回值
        """
        current_state = {
            "vehicle":  full_data.get("vehicle", {}),
            "tls":      full_data.get("tls", {}),
            "aircraft": full_data.get("aircraft", {}),
        }

        if not self._reset_done:
            self.renderer.reset(current_state)
            self.renderer._showbase_instance.taskMgr.add(
                self.renderer.dummyTask, "dummyTask"
            )
            self._reset_done = True

        if self.event_logic_manager is not None and self.emergency_renderer is not None:
            active_events = self.event_logic_manager.get_active_events(current_time)
            self.emergency_renderer.update(active_events)

        sensor_data = self.renderer.step(current_state, should_count_vehicles=False)

        if sensor_data:
            os.makedirs(save_folder, exist_ok=True)
            for sensor_id, img_dict in sensor_data.items():
                img = img_dict.get("junction_front_all")
                if img is not None:
                    save_path = os.path.join(save_folder, f"{sensor_id}.png")
                    # TSHubRenderer 返回 RGB，cv2 需要 BGR
                    cv2.imwrite(save_path, img[:, :, ::-1])

        return sensor_data or {}

    # ------------------------------------------------------------------
    def close(self):
        if self.emergency_renderer is not None:
            self.emergency_renderer.clear()
        self.renderer.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# 渲染流程
# ──────────────────────────────────────────────────────────────────────────────

def run_offline_render(
    scenario_key: str,
    json_files: list[str],
    route_xml_path: str | None = None,
) -> None:
    """
    对给定的 JSON 文件列表执行离线渲染，结果写回各文件所在 render_info/ 的子目录。

    Args:
        scenario_key: SCENARIO_CONFIGS 中的场景键，e.g. "JiNan"
        json_files:   render_{t}.json 文件路径列表（绝对或相对均可）
    """
    if not json_files:
        print("[OfflineRender] 未指定任何 JSON 文件，退出。")
        return

    # ── 导入路径工具（依赖项目根目录在 sys.path 中）────────────────────────────
    try:
        from configs.scenairo_config import SCENARIO_CONFIGS
        from tshub.utils.get_abs_path import get_abs_path
    except ImportError as e:
        raise ImportError(
            f"无法导入项目模块，请在项目根目录下运行或将根目录加入 PYTHONPATH: {e}"
        )

    config = SCENARIO_CONFIGS.get(scenario_key)
    if not config:
        raise ValueError(
            f"场景 '{scenario_key}' 不存在于 SCENARIO_CONFIGS，"
            f"可选值: {list(SCENARIO_CONFIGS.keys())}"
        )

    SCENARIO_NAME = config["SCENARIO_NAME"]
    NETFILE       = config["NETFILE"]
    RENDERER_CFG  = config.get("RENDERER_CFG", {})
    SENSOR_CFG    = config.get("SENSOR_CFG", {})

    # get_abs_path 以调用方文件为基准解析相对路径
    path_convert     = get_abs_path(__file__)
    scenario_glb_dir = path_convert(f"../../data/raw/{SCENARIO_NAME}/3d_assets/")

    # ── 按仿真时间排序 ──────────────────────────────────────────────────────────
    try:
        json_files_sorted = sorted(json_files, key=_extract_step)
    except ValueError as e:
        raise ValueError(
            f"JSON 文件名格式错误，期望 render_{{整数}}.json，实际: {e}"
        )

    print(f"\n{'='*60}")
    print(f"  离线 BEV 渲染")
    print(f"  场景: {SCENARIO_NAME}")
    print(f"  待渲染帧数: {len(json_files_sorted)}")
    print(f"  3D 资产目录: {scenario_glb_dir}")

    if route_xml_path is None:
        route_xml_path = _resolve_route_xml_path(
            scenario_key=scenario_key,
            scenario_name=SCENARIO_NAME,
            netfile=NETFILE,
            json_path=json_files_sorted[0],
        )
    if route_xml_path:
        print(f"  事件路由: {route_xml_path}")
    else:
        print("  事件路由: 未解析到，将仅渲染普通交通参与者")

    print(f"{'='*60}\n")

    renderer = OfflineBEVGenerator(
        scenario_glb_dir=scenario_glb_dir,
        sensor_cfg=SENSOR_CFG,
        renderer_cfg=RENDERER_CFG,
        route_xml_path=route_xml_path,
    )

    success_count = 0
    for json_path in json_files_sorted:
        step = _extract_step(json_path)
        save_folder = _save_folder_for(json_path, step)

        print(f"[步骤 {step}] 读取: {json_path}")
        print(f"           输出: {save_folder}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            sensor_data = renderer.process_state(
                data,
                save_folder=save_folder,
                current_time=step,
            )
            n_images = sum(
                1 for v in sensor_data.values() if v.get("junction_front_all") is not None
            )
            print(f"           保存图像: {n_images} 张")
            success_count += 1

        except FileNotFoundError:
            print(f"  [警告] 文件不存在，跳过: {json_path}")
        except json.JSONDecodeError as e:
            print(f"  [警告] JSON 解析失败，跳过 {json_path}: {e}")
        except Exception as e:
            print(f"  [错误] 步骤 {step} 渲染失败: {e}")

    renderer.close()

    print(f"\n{'='*60}")
    print(f"  完成！成功 {success_count}/{len(json_files_sorted)} 帧")
    print(f"{'='*60}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="离线 BEV 渲染：从 render_info/render_{t}.json 重渲染并保存图像",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""示例:
  # 渲染单帧
  python src/bev_generation/offline_bev_render.py \\
      --scenario_key JiNan \\
      --json_files data/eval/JiNan/anon_3_4_jinan_real/qwen3-vl-4b/render_info/render_28.json

  # 渲染目录下所有帧
  python src/bev_generation/offline_bev_render.py \\
      --scenario_key JiNan \\
      --json_glob "data/eval/JiNan/anon_3_4_jinan_real/qwen3-vl-4b/render_info/render_*.json"
""",
    )
    parser.add_argument(
        "--scenario_key",
        type=str,
        required=True,
        help="SCENARIO_CONFIGS 中的场景键，e.g. JiNan, Hangzhou, Hongkong_YMT",
    )

    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--json_files",
        type=str,
        nargs="+",
        metavar="PATH",
        help="一个或多个 render_{t}.json 文件路径",
    )
    src_group.add_argument(
        "--json_glob",
        type=str,
        metavar="PATTERN",
        help='glob 模式，自动匹配所有帧，e.g. "render_info/render_*.json"',
    )
    parser.add_argument(
        "--route_xml",
        type=str,
        default=None,
        help="可选：显式指定事件路由 .rou.xml，适用于自定义 JSON 目录结构",
    )
    return parser


if __name__ == "__main__":
    # ── 判断运行模式 ────────────────────────────────────────────────────────────
    _is_debug = len(sys.argv) == 1  # 无命令行参数 → Debug 模式

    if _is_debug:
        print("[OfflineRender] Debug 模式（无 CLI 参数）")
        _render_tasks = _collect_debug_render_tasks()
    else:
        # CLI 模式
        _args = _build_parser().parse_args()
        _render_tasks = [
            _normalize_render_task(
                scenario_key=_args.scenario_key,
                json_files=_args.json_files,
                json_glob=_args.json_glob,
                route_xml_path=_args.route_xml,
            )
        ]

    # ── 将项目根目录加入 sys.path（兼容从任意目录启动的情况）──────────────────
    _project_root = _get_project_root()
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    for task_idx, task in enumerate(_render_tasks, start=1):
        print(
            f"[OfflineRender] 开始任务 {task_idx}/{len(_render_tasks)}: "
            f"{task['scenario_key']}"
        )
        run_offline_render(
            scenario_key=task["scenario_key"],
            json_files=task["json_files"],
            route_xml_path=task["route_xml_path"],
        )
