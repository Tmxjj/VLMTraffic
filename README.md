<!--
 * @Author: yufei Ji
 * @Date: 2026-01-12 16:48:03
 * @LastEditTime: 2026-01-14 22:33:33
 * @Description: this script is used to 
 * @FilePath: /VLMTraffic/README.md
-->
# VLMTraffic

[中文版](./README_zh.md) | [English](./README.md)

This project implements an end-to-end Vision-Language Model (VLM) framework for Traffic Signal Control (TSC).

## Overview

The framework processes BEV (Bird's Eye View) images from traffic intersections and uses a VLM to make signal control decisions. It includes modules for simulation, dataset generation (SFT & DPO), training, and evaluation.

## Directory Structure

- `data/`: Contains raw simulation data, generated BEV images, and processed SFT/DPO datasets.
- `models/`: Stores base models downloaded from ModelScope and trained checkpoints.
- `src/`: Source code for the framework.
    - `bev_generation/`: Wraps the image generation logic from `VLMLight`.
    - `inference/`: VLM inference and prompt construction.
    - `dataset/`: SFT and DPO dataset generation logic.
    - `training/`: Training scripts for Supervised Fine-Tuning and Direct Preference Optimization.
    - `evaluation/`: End-to-end evaluation metrics (ATT, AQL, AWT).
- `configs/`: unified YAML-based configuration system
- `scripts/`: Utility scripts for running simulations, training, and data generation.

## Configuration

The project uses a unified YAML-based configuration system located in `src/configs/`.

- `env_config.yaml`: Simulation environment settings (scenario, renderer, sensor parameters).
- `model_config.yaml`: Model paths, inference settings, and LoRA parameters.
- `train_config.yaml`: Hyperparameters for SFT and DPO training (learning rate, batch size, epochs).

## Data Flow

1.  **Simulation & Data Generation**:
    - The `TSCEnvWrapper` (src/utils/tsc_env/tsc_wrapper.py) interacts with SUMO.
    - `BEVGenerator` (src/bev_generation) creates BEV images from simulation states data.
2.  **Dataset Construction**:
    - Raw data and expert decisions are processed by `SFTDatasetGenerator` and `DPODatasetGenerator` (src/dataset) to create training data saved in `data/`.
3.  **Training**:
    - `SFTTrainer` and `DPOTrainer` (src/training) consume datasets to fine-tune the base model.
    - Checkpoints are saved to `models/checkpoints/`.
4.  **Inference & Evaluation**:
    - `VLMAgent` (src/inference) loads the trained model.
    - It receives BEV images and prompts to generate traffic decisions, which are executed back in the simulation.
    - `Evaluator` (src/evaluation) computes metrics like ATT, AQL, and AWT.

## Dependencies

Please refer to `requirements.txt`.

## Getting Started

1.  **Environment Setup**: Install dependencies.
```bash
pip install -r requirement.txt
git clone https://github.com/Tmxjj/TransSimHub.git
cd TransSimHub # This study is based on secondary development of https://github.com/Traffic-Alpha/TransSimHub.git
pip install -e ".[all]"
```
2.  **Model Download**: Run `python scripts/download_model.py` to fetch the base VLM.
3.  **Simulation & BEV Generation**: 
4.  **Training**: Use `src/training/sft_trainer.sh`.
5.  **Evaluation**: Run `src/evaluation/evaluator.py`.

## Metrics

The system evaluates:
- Average Traveling Time (ATT)
- Average Queue Length (AQL)
- Average Waiting Time (AWT)
- Metrics for special vehicles (Ambulance, Fire trucks, etc.)

## 3D Model Asset Download

1. **Map 3D Assets**:
Download URL: https://drive.google.com/drive/folders/1oTIScFrKsmmwSaG6bGCBmkSHD3fWdq86?usp=sharing
Installation Path: `data/raw/Hongkong_YMT/3d_assets`

2. **Vehicle 3D Assets**:
Download URL: https://github.com/Traffic-Alpha/TransSimHub/releases/download/v5/vehicles_models.zip
Installation Path: `TransSimHub/tshub/tshub_env3d/_assets_3d`
