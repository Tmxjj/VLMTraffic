"""Batch prompt test for data/test/prompt_test.

Directory convention:
  data/test/prompt_test/{scenario_name}/{step}/INT1_N.png
  data/test/prompt_test/{scenario_name}/{step}/INT1_E.png
  data/test/prompt_test/{scenario_name}/{step}/INT1_S.png
  data/test/prompt_test/{scenario_name}/{step}/INT1_W.png

The script scans each scenario/step folder, builds a scenario-aware prompt,
calls VLMAgent once per step, and saves the response to a txt file that
explicitly records the scenario and step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from configs.prompt_builder import PromptBuilder
from src.inference.vlm_agent import VLMAgent


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
        default="",
        help="可选：仅测试某个场景；留空则遍历所有场景",
    )
    parser.add_argument(
        "--current_phase_id",
        type=int,
        default=0,
        help="PromptBuilder 使用的当前相位编号",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

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
            image_paths = _collect_step_images(step_dir)
            if not image_paths:
                continue

            prompt = PromptBuilder.build_decision_prompt(
                current_phase_id=args.current_phase_id,
                scenario_name=scenario_name,
            )

            response, latency, action, thought = agent.get_decision(image_paths, prompt)

            step_output_dir = output_root / scenario_name / step_dir_name
            step_output_dir.mkdir(parents=True, exist_ok=True)
            output_txt = step_output_dir / "response.txt"

            with output_txt.open("w", encoding="utf-8") as f:
                f.write(f"Scenario: {scenario_name}\n")
                f.write(f"Step: {step_dir_name}\n")
                f.write(f"Images: {json.dumps(image_paths, ensure_ascii=False)}\n")
                f.write(f"Latency: {latency:.2f}s\n")
                f.write(f"Action: phase={action[0]}, duration={action[1]}\n\n")
                f.write("Prompt:\n")
                f.write(prompt)
                f.write("\n\nResponse:\n")
                f.write(response)
                if thought:
                    f.write("\n\nThought:\n")
                    f.write(thought)
                f.write("\n")

            print(
                f"[OK] scenario={scenario_name} step={step_dir_name} "
                f"saved={output_txt} action=({action[0]}, {action[1]})"
            )


if __name__ == "__main__":
    main()