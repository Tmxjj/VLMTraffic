"""Batch prompt test for data/test/prompt_test.

Directory convention:
  data/test/prompt_test/{scenario_name}/{step}/INT1_N.png
  data/test/prompt_test/{scenario_name}/{step}/INT1_E.png
  data/test/prompt_test/{scenario_name}/{step}/INT1_S.png
  data/test/prompt_test/{scenario_name}/{step}/INT1_W.png

The script scans each scenario/step folder, builds a scenario-aware prompt,
adds lane watermarks to the input images, calls VLMAgent once per step,
and saves the response to a txt file that explicitly records the scenario
and step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from add_lane_watermarks import add_lane_watermarks
from configs.prompt_builder import PromptBuilder


def _collect_step_images(step_dir: Path) -> list[str]:
    preferred_order = ["N", "E", "S", "W"]
    image_paths: list[str] = []

    for direction in preferred_order:
        matches = sorted(step_dir.glob(f"*_{direction}.png"))
        if matches:
            image_paths.append(str(matches[0]))

    if image_paths:
        return image_paths

    return [str(path) for path in sorted(step_dir.glob("*.png"))]


def _build_watermarked_images(
    image_paths: list[str],
    scenario_name: str,
    watermark_dir: Path,
    regenerate: bool,
) -> list[str]:
    watermark_dir.mkdir(parents=True, exist_ok=True)

    watermarked_paths: list[str] = []
    for image_path_str in image_paths:
        image_path = Path(image_path_str)
        output_path = watermark_dir / image_path.name

        if regenerate or not output_path.exists():
            add_lane_watermarks(
                input_path=str(image_path),
                output_path=str(output_path),
                scenario_name=scenario_name,
            )

        if not output_path.exists():
            raise FileNotFoundError(f"Watermarked image was not created: {output_path}")

        watermarked_paths.append(str(output_path))

    return watermarked_paths


def _sanitize_filename_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("._-") or "unknown"


def _get_response_output_path(step_output_dir: Path, agent: object) -> Path:
    api_type = _sanitize_filename_part(getattr(agent, "api_type", "unknown_api"))

    agent_config = getattr(agent, "config", {}) or {}
    raw_model_name = agent_config.get("model_name")
    if not raw_model_name:
        model_path = agent_config.get("model_path", "")
        raw_model_name = Path(model_path).name if model_path else "unknown_model"

    model_name = _sanitize_filename_part(str(raw_model_name))
    return step_output_dir / f"response_{api_type}_{model_name}.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch prompt test for data/test/prompt_test")
    parser.add_argument(
        "--input_root",
        type=str,
        default="data/test/prompt_test",
        help="输入目录，默认 data/test/prompt_test",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/test/prompt_test",
        help="输出目录，默认 data/test/prompt_test（会在每个 step 文件夹下生成 response.txt）",
    )
    parser.add_argument(
        "--scenario_name",
        type=str,
        default="SouthKorea_Songdo",
        help="可选：仅测试某个场景；留空则遍历所有场景",
    )
    parser.add_argument(
        "--current_phase_id",
        type=int,
        default=0,
        help="PromptBuilder 使用的当前相位编号",
    )
    parser.add_argument(
        "--watermark_subdir",
        type=str,
        default="watermarked_inputs",
        help="每个 step 下保存加水印图片的子目录名，默认 watermarked_inputs",
    )
    parser.add_argument(
        "--regenerate_watermarks",
        action="store_true",
        help="若已存在加水印图片，是否重新生成",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    from src.inference.vlm_agent import VLMAgent

    agent = VLMAgent()

    if args.scenario_name:
        scenario_dirs = [input_root / args.scenario_name]
    else:
        scenario_dirs = [path for path in input_root.iterdir() if path.is_dir()]

    for scenario_dir in sorted(scenario_dirs):
        if not scenario_dir.exists():
            continue

        scenario_name = scenario_dir.name
        for step_dir_name, step_dir in sorted(
            [(path.name, path) for path in scenario_dir.iterdir() if path.is_dir()],
            key=lambda item: (0, int(item[0])) if item[0].isdigit() else (1, item[0]),
        ):
            original_image_paths = _collect_step_images(step_dir)
            if not original_image_paths:
                continue

            step_output_dir = output_root / scenario_name / step_dir_name
            watermark_dir = step_output_dir / args.watermark_subdir
            image_paths = _build_watermarked_images(
                image_paths=original_image_paths,
                scenario_name=scenario_name,
                watermark_dir=watermark_dir,
                regenerate=args.regenerate_watermarks,
            )

            prompt = PromptBuilder.build_decision_prompt(
                current_phase_id=args.current_phase_id,
                scenario_name=scenario_name,
            )

            response, latency, action, thought = agent.get_decision(image_paths, prompt)

            step_output_dir.mkdir(parents=True, exist_ok=True)
            output_txt = _get_response_output_path(step_output_dir, agent)

            with output_txt.open("w", encoding="utf-8") as f:
                f.write(f"Scenario: {scenario_name}\n")
                f.write(f"Step: {step_dir_name}\n")
                f.write(f"Original Images: {json.dumps(original_image_paths, ensure_ascii=False)}\n")
                f.write(f"Watermarked Images: {json.dumps(image_paths, ensure_ascii=False)}\n")
                f.write(f"Latency: {latency:.2f}s\n")
                f.write(f"Action: phase={action[0]}, duration={action[1]}\n\n")
                f.write("Prompt:\n")
                f.write(prompt)
                f.write("\n\nResponse:\n")
                f.write(response)
                if thought:
                    f.write("\n\n Model Thought:\n")
                    f.write(thought)
                f.write("\n")

            print(
                f"[OK] scenario={scenario_name} step={step_dir_name} "
                f"saved={output_txt} watermarked_dir={watermark_dir} "
                f"action=({action[0]}, {action[1]})"
            )


if __name__ == "__main__":
    main()
