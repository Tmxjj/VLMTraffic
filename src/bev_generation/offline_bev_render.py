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

import cv2


# ──────────────────────────────────────────────────────────────────────────────
# Debug 模式配置（仅在直接 python offline_bev_render.py 时生效）
# ──────────────────────────────────────────────────────────────────────────────
DEBUG_SCENARIO_KEY = "JiNan"
DEBUG_JSON_FILES = [
    "data/eval/JiNan/anon_3_4_jinan_real/qwen3-vl-4b/render_info/render_28.json",
]
# 设为 None 则自动扫描 render_info/ 目录下所有 render_*.json
DEBUG_JSON_GLOB = None  # e.g. "data/eval/JiNan/.../render_info/render_*.json"


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


# ──────────────────────────────────────────────────────────────────────────────
# 核心渲染类
# ──────────────────────────────────────────────────────────────────────────────

class OfflineBEVGenerator:
    """
    封装 TSHubRenderer 的离线渲染器。
    同一个 Renderer 实例可连续处理多帧，TSHubRenderer.reset() 只在第一帧调用一次。
    """

    def __init__(self, scenario_glb_dir: str, sensor_cfg: dict, renderer_cfg: dict):
        try:
            from tshub.tshub_env3d.vis3d_renderer.tshub_render import TSHubRenderer
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
        self._reset_done = False

    # ------------------------------------------------------------------
    def process_state(self, full_data: dict, save_folder: str) -> dict:
        """
        渲染单帧状态并将图像保存到 save_folder。

        Args:
            full_data:   render_{t}.json 的完整内容（含 vehicle/tls/aircraft/person）
            save_folder: 图像输出目录（会自动创建）

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
        self.renderer.destroy()


# ──────────────────────────────────────────────────────────────────────────────
# 渲染流程
# ──────────────────────────────────────────────────────────────────────────────

def run_offline_render(scenario_key: str, json_files: list[str]) -> None:
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
    print(f"{'='*60}\n")

    renderer = OfflineBEVGenerator(
        scenario_glb_dir=scenario_glb_dir,
        sensor_cfg=SENSOR_CFG,
        renderer_cfg=RENDERER_CFG,
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

            sensor_data = renderer.process_state(data, save_folder=save_folder)
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
    return parser


if __name__ == "__main__":
    # ── 判断运行模式 ────────────────────────────────────────────────────────────
    _is_debug = len(sys.argv) == 1  # 无命令行参数 → Debug 模式

    if _is_debug:
        # Debug 模式：直接使用脚本顶部的 DEBUG_* 变量
        print("[OfflineRender] Debug 模式（无 CLI 参数）")
        _scenario_key = DEBUG_SCENARIO_KEY
        if DEBUG_JSON_GLOB:
            _json_files = sorted(glob.glob(DEBUG_JSON_GLOB))
            if not _json_files:
                raise FileNotFoundError(f"DEBUG_JSON_GLOB 无匹配文件: {DEBUG_JSON_GLOB}")
        else:
            _json_files = DEBUG_JSON_FILES
    else:
        # CLI 模式
        _args = _build_parser().parse_args()
        _scenario_key = _args.scenario_key
        if _args.json_glob:
            _json_files = sorted(glob.glob(_args.json_glob))
            if not _json_files:
                raise FileNotFoundError(f"--json_glob 无匹配文件: {_args.json_glob}")
        else:
            _json_files = _args.json_files

    # ── 将项目根目录加入 sys.path（兼容从任意目录启动的情况）──────────────────
    _project_root = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
    )
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    run_offline_render(scenario_key=_scenario_key, json_files=_json_files)
