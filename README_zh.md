<!--
 * @Author: yufei Ji
 * @Date: 2026-01-12 16:54:26
 * @LastEditTime: 2026-01-14 16:34:38
 * @Description: this script is used to 
 * @FilePath: /VLMTraffic/README_zh.md
-->
# VLMTraffic

本项目实现了一个用于交通信号控制 (TSC) 的端到端视觉语言模型 (VLM) 框架。

## 概览

该框架处理来自交通路口的 BEV (鸟瞰图) 图像，并使用 VLM 做出信号控制决策。它包含仿真、数据集生成 (SFT & DPO)、训练和评估等模块。

## 目录结构

- `data/`: 包含原始仿真数据、生成的 BEV 图像以及处理后的 SFT/DPO 数据集。
- `models/`: 存储从 ModelScope 下载的基础模型和训练好的检查点 (checkpoints)。
- `src/`: 框架源代码。
    - `bev_generation/`: 封装来自 `VLMLight` 的图像生成逻辑。
    - `inference/`: VLM 推理和提示词 (prompt) 构建。
    - `dataset/`: SFT 和 DPO 数据集生成逻辑。
    - `training/`: 监督微调 (SFT) 和直接偏好优化 (DPO) 的训练脚本。
    - `evaluation/`: 端到端评估指标 (ATT, AQL, AWT)。
- `configs/`: 项目使用基于 YAML 的统一配置系统
- `scripts/`: 用于运行仿真、训练和数据生成的实用脚本。

## 配置管理

项目使用基于 YAML 的统一配置系统，位于 `configs/` 目录下。

- `env_config.yaml`: 仿真环境配置（场景、渲染器、传感器参数）。
- `model_config.yaml`: 模型路径、推理设置和 LoRA 参数。
- `train_config.yaml`: SFT 和 DPO 训练的超参数（学习率、批次大小、轮数）。

## 数据流向 (Data Flow)

1.  **仿真与数据生成**:
    - 基于sumo仿真
    - `BEVGenerator` (src/bev_generation) 基于仿真状态数据生成 BEV 图像。
2.  **数据集构建**:
    - 原始数据和专家决策由 `src/dataset` 下的生成器处理，构建 SFT 和 DPO 数据集，保存在 `data/` 中。
3.  **模型训练**:
    - `src/training` 中的训练模块读取数据集，对基础模型进行微调。
    - 训练好的检查点 (Checkpoints) 保存在 `models/checkpoints/`。
4.  **推理与评估**:
    - `VLMAgent` (src/inference) 加载微调后的模型。
    - 接收 BEV 图像和提示词（Prompt），生成控制决策并反馈回仿真环境。
    - `Evaluator` (src/evaluation) 统计 ATT, AQL, AWT 等指标。

## 依赖项

请参考 `requirements.txt`。

##以此类推...

## 快速开始

1.  **环境设置**: 安装依赖项。
```bash
pip install -r requirement.txt
git clone https://github.com/Traffic-Alpha/TransSimHub.git
cd TransSimHub # 原地址为https://github.com/Traffic-Alpha/TransSimHub.git，本研究基于此进行了二次开发
pip install -e ".[all]"
```
2.  **模型下载**: 运行 `python scripts/download_model.py` 获取基础 VLM 模型。
3.  **仿真 & BEV 生成**: BEV 生成依赖于 `VLMLight` 项目。请确保 `VLMLight` 可访问。
4.  **训练**: 使用 `src/training/sft_trainer.py` 和 `src/training/dpo_trainer.py` 进行训练。
5.  **评估**: 运行 `src/evaluation/evaluator.py` 进行评估。

## 指标

系统评估以下指标：
- 平均旅行时间 (Average Traveling Time, ATT)
- 平均队列长度 (Average Queue Length, AQL)
- 平均等待时间 (Average Waiting Time, AWT)
- 特殊车辆 (救护车、消防车等) 的相关指标

## 3D模型文件下载

1. **地图3D文件**：
下载地址：https://drive.google.com/drive/folders/1oTIScFrKsmmwSaG6bGCBmkSHD3fWdq86?usp=sharing
安装路径：data/raw/Hongkong_YMT/3d_assets

2、**车辆3D文件**：
下载地址：https://github.com/Traffic-Alpha/TransSimHub/releases/download/v5/vehicles_models.zip
安装路径：TransSimHub/tshub/tshub_env3d/_assets_3d