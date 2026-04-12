# E2ELight - 端到端视觉语言模型交通信号控制框架

## 一. 出发点
现有的“描述”——“决策”两阶段方法（用小模型/LVLM提取文本场景描述，再输入 LLM 进行决策）带来以下问题：
1. **推理延迟**：两阶段推理会显著增加耗时，难以满足交通信号控制的毫秒级实时性要求。
2. **针对紧急车辆场景、复杂交叉口的泛化能力差**：小模型/LVLM提取文本场景描述（如每个车道上车辆数），会损失整个道路拓扑和车辆视觉信息，导致其在紧急车辆、复杂交叉口的泛化能力不足。

针对上述问题，本项目的两个核心贡献及后续待验证的优化方向如下：

- **贡献 1：提出了基于 LVLM (Large Vision-Language Model) 的交通信号控制框架**
  直接处理来自交通路口的 BEV (鸟瞰图) 图像，并使用微调后的 LVLM 直接输出推理决策过程和信号控制决策（思维链 CoT）。引入视觉模态保留完整的路况语义，捕捉文本无法涵盖的细粒度时空特征。

- **贡献 2：针对“多目标幻觉（Multi-object Hallucination）尤其是计数幻觉（Counting Hallucination）和位置幻觉（Positional Hallucination）”的问题，提出以下优化方向**
  
  - **优化方向 1：针对多模态思维链的“逻辑因果”细粒度对齐 (Causal Step-DPO for Multimodal CoT)**
    - **目前痛点**：当前 Prompt 中使用了典型的思维链（CoT）：`Scene Understanding -> Scene Analysis -> Selection Logic`。目前 DPO 把一整段话切成句子来算 Loss，但忽略了句子之间的因果依赖（例如，因为数错了车导致拥堵等级判断错误，最后选错了相位）。
    - **核心创新**：提出一种逻辑因果感知的细粒度 DPO。在构建偏好对时，故意针对 CoT 的不同阶段进行干预。比如设计：*样本对 A（视觉感知错误，但逻辑推导正确）；样本对 B（视觉感知正确，但逻辑推导错误）*。在 C-DPO 的基础上设计阶段性 Mask 机制，引导模型：如果前提条件（如车道拥堵数）给定了，哪怕前提是错的，逻辑推导必须是对的。将“感知幻觉”和“逻辑幻觉”解耦并分别计算 DPO 惩罚权重。
    - **落地场景 (TSC)**：这能大幅提升在复杂交通路口（如香港左侧通行特例）下的决策稳定性，确保即使模型看不清某辆车，其给出的信号灯调度逻辑依然符合交通工程常识。

  - **优化方向 2：基于物理仿真器反馈的令牌级环境 DPO (Simulation-Guided Token-Level DPO)**
    - **目前痛点**：现有研究证明了可以用目标检测器（Grounding DINO 等）代替人去给模型打分。但对于交通控制任务来说，“选对选错”的终极裁判不是视觉检测器，而是物理世界的交通效率。
    - **核心创新**：实现 “RLHF without Human”。将 LVLM 接入 SUMO 或 CityFlow 等交通仿真器。模型每输出一个动作 Token（例如 Phase 1），立刻在仿真器中运行 10 秒，拿到真实的物理奖励（如：排队长度减少了多少，是否有救护车通过）。将真实的物理环境 Reward 直接转换为细粒度 DPO 公式中的加权系数 $\gamma$（由仿真器动态赋予）。
    - **落地场景 (TSC)**：如果模型因为漏看了救护车而选择了错误的相位，仿真器会返回极大的负 Reward。此时 DPO 直接对“漏看救护车”的那些 Token 施加毁灭性的打击。这将是交通领域首个闭环环境反馈多模态偏好对齐框架。
  
  - **优化方向 3：空间规则与数值距离感知的连续偏好对齐 (Spatial-Rule & Numerically-Sensitive Continuous DPO)**
  - **目前痛点**：现有的 DPO 都是“离散的、非黑即白的”。预测 4 辆车（实际 5 辆）和预测 15 辆车（实际 5 辆），在标准 DPO 眼里都是等价的“负样本（$y_l$）”，惩罚力度完全一样。同时，在交通场景下，模型往往是因为违反了 Prompt 中定义的“停止线约束”或“方向约束”从而导致数量数错，现有的优化方法无法纠正这一根本原因。
  - **核心创新**：提出一种基于数值误差与物理规则的连续动态惩罚机制。一方面引入“数值距离惩罚（Distance-Calibrated Margin）”，打破标准 DPO 固定的 $\beta$ 裕度，误差 $|N_{pred} - N_{gt}|$ 越大，该负样本的排斥力度越强，让模型明白“数错1辆是小错，数错10辆是大错”；另一方面进行“空间规则逆向采样”，故意构造包含了越过停止线（负样本 $y_{l_1}$）或对向车道车辆（负样本 $y_{l_2}$）的“踩坑”负样本，与严格遵守规则的正确输出（正样本 $y_w$）形成对比。
  - **落地场景 (TSC)**：结合 Prompt 中的视觉约束（Visual Constraints）。在 C-DPO 计算时，不仅 Mask 掉前置上下文，还专门针对那些由于“越线”或“逆向”导致计数错误的 Token 施加定向的、带有物理意义的惩罚权重。这不仅能有效治愈模型的“计数幻觉”，更能强行将人类的交通拓扑规则直接刻入 LVLM 的模型权重中。

  - **优化方向 4：基于仿真器双路可验证奖励的 LVLM 强化学习 (Simulation-Grounded Dual-Verifiable RLVR for LVLM-TSC)**
  - **目前痛点**：优化方向 1-3 均属离线 DPO 范式——需手工构造偏好对、信号稀疏、感知与决策孤立优化。而 SUMO 仿真器对每一次信号决策都能即时提供两类硬标签可验证信号，却完全未被利用：(1) e2 检测器提供每条 movement 的真实排队车辆数，可直接验证模型 Scene Understanding 的计数准确性；(2) rollout 若干仿真步后的 ATT/AQL 变化量，可直接量化信号决策的交通效率。
  - **核心创新**：提出以 GRPO（Group Relative Policy Optimization）范式对 LVLM 进行在线强化学习微调，完全摆脱偏好对构造的依赖。定义双路可验证奖励：**感知奖励 $r_{\text{perc}}$**（SUMO `jam_length_vehicle` 与模型预测车辆数的归一化误差之负数，直接治愈计数幻觉）和 **效率奖励 $r_{\text{env}}$**（执行所选相位后 30 仿真步的 $\Delta \text{ATT}$，作为交通效率终极裁判）。联合奖励 $r = \alpha \cdot r_{\text{perc}} + \beta \cdot r_{\text{env}}$ 使感知正确性与控制决策质量得到统一梯度信号的同步约束。
  - **GRPO 训练流程**：对同一 BEV 图像采样 $G=8$ 个完整 CoT 响应（Scene Understanding → Scene Analysis → Selection Logic）；用双路奖励对每个响应打分；计算组内相对优势 $A_i = (r_i - \text{mean}(r)) / \text{std}(r)$；GRPO 梯度更新 LVLM，无需 Critic 网络和 Value Model。
  - **与 DPO 方向的本质区别**：DPO 1-3 均为离线静态偏好信号，本方向为在线动态验证信号；不依赖人工标注，仿真器即裁判；模型可自发涌现精确计数策略（类 DeepSeek-R1 "aha moment" 现象）。Traffic-R1（2025）已证明 RLVR 范式在纯文本 TSC 中的有效性，本方向将其首次扩展至 **BEV 视觉输入 + 多模态 CoT** 场景，是交通领域首个视觉-仿真闭环 RLVR 框架。
  - **落地可行性**：e2 检测器（$r_{\text{perc}}$ 数据源）与 SUMO rollout（$r_{\text{env}}$ 数据源）均为项目现有基础设施，无需新增硬件；可在现有 SFT+DPO checkpoint 基础上继续训练，实施路径清晰。

## 二. 项目概览
本项目实现了一个用于交通信号控制 (TSC) 的端到端视觉语言大模型 (LVLM) 框架。该框架直接处理来自交通路口的 BEV (鸟瞰图) 图像，并使用微调后的 LVLM 直接输出信号控制决策。项目包含了完整的 Pipeline，涵盖基于 SUMO/TransSimHub 的仿真、数据生成 (SFT & DPO)、模型训练以及评估。

## 三. 核心目录与模块
- `data/`：存放原始仿真数据、生成的 BEV 图像以及处理后的 SFT（监督微调）/ DPO（偏好优化）数据集。
- `models/`：用于存储从 ModelScope 等下载的基础 LVLM 模型和训练后生成的 Checkpoints。
- `src/`：框架核心代码库。
  - `bev_generation/`：负责与仿真环境交互并生成路口 BEV 图像。
  - `dataset/`：负责 SFT 和 DPO 训练数据集的构建与处理。
  - `inference/`：LVLM 模型推理与 Prompt 构建逻辑。
  - `training/`：SFT 和 DPO 的训练核心模块。
  - `evaluation/`：端到端指标评估 (ATT, AQL, AWT 等)。
- `configs/`：基于 YAML 的统一配置目录，管理仿真环境 (`env_config.yaml`)、模型与 Prompt (`model_config.yaml` / `prompt_builder*.py`) 以及训练参数 (`train_config.yaml`)。
- `scripts/`：大量实用的辅助脚本，包括模型下载、仿真运行、评测任务投递 (`run_eval_gpu*.sh`) 等。
- `TransSimHub/`：底层 3D 交通仿真环境基座（基于开源仓库做的二次深度开发集成）。

## 四.数据集与场景介绍

本研究用于训练和评测的数据集涵盖了全球多个城市不同类型的交通路口，包括真实监控数据和针对特定场景（如大规模路网、紧急车辆、特殊路口拓扑）的验证数据。

- 1 核心训练与测试数据集 (真实监控数据)

这些数据集基于真实世界采集，并模拟了高峰流量等不同时段的交通状况，是模型基础训练和同源验证的基石。

| 数据集/场景名称 | 简介与特点 | 交叉口数量 | 相位描述 (Action Space for Prompt) | 适用性与用途 | 车流特征摘要 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Jinan (China)** | 真实监控数据采集。包含3个时段采集的数据。基于真实采集数据，模拟了高峰流量数据和全天候数据（包含早晚高峰）。 | 12 | 原为9相位，**Prompt中简化为4相位：**<br>NTST (南北直行), NLSL (南北左转), ETWT (东西直行), ELWL (东西左转) | **数据生产、训练和测试** | - |
| **Hangzhou (China)** | 真实监控数据。包含3个时段采集的数据。基于真实采集数据，模拟了高峰流量数据。 | 16 | 同 Jinan 数据 (4相位) | **数据生产、训练和测试** | - |

- 2 泛化性验证数据集 (拓扑与规模迁移)

为了验证模型对复杂、罕见路口拓扑结构以及大规模城市级路网的自适应与协同控制能力，本研究引入了以下特定城市的测试场景（这些数据**不参与**模型的训练）。

| 数据集/场景名称 | 简介与特点 | 交叉口数量 | 相位描述 (Action Space for Prompt) | 适用性与用途 | 车流特征摘要 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Songdo (South Korea)** | 新开发城区，大型交叉口。每个方向多达5车道，交通流量大。 | 1 | **4相位：** 东西/南北 直行/左转，右转无限制 | **泛化性（拓扑迁移验证）：** 验证模型能否适应复杂且庞大的路口拓扑。 | Total Vehicles: 598<br>Effective Duration: 6.98 min |
| **Yau Ma Tei (Hong Kong)** | 稠密市中心，道路狭窄。包含特定的转向限制。只有南进口有右转，东进口有左转。 | 1 | 原始逻辑为四个进口道轮流放行。<br>➡️ **修改为3相位：** 东西/南北直行，南进口右转（东进口左转一直放行）。 | **泛化性（拓扑迁移验证）：** 验证模型能否适应复杂、受限的路口拓扑及左行特例。 | Total Vehicles: 267<br>Effective Duration: 417.88 s |
| **Massy (France)** | 郊区 T字路口 (T-junction)。车道配置特殊，流量较小。 | 1 | **2相位：** 南北进口道放行，西进口道放行（右转无限制）。 | **泛化性（拓扑迁移验证）：** 验证模型能否适应异构路口拓扑。 | Total Vehicles: 207<br>Effective Duration: 6.98 min |
| **New York (USA)** | 曼哈顿上东区，基于出租车轨迹数据。属于超大规模复杂路网。基于真实采集数据，模拟高峰流量数据。 | 196 | 同 Jinan 数据 (4相位) | **泛化性（规模迁移验证）：** 用于验证模型由单路口/小路网向大规模路网扩展时的协同控制与泛化能力。 | - |

- 3 紧急事件验证数据集

| 数据集/场景名称 | 简介与特点 | 交叉口数量 | 相位描述 (Action Space for Prompt) | 适用性与用途 | 车流特征摘要 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Emergency** | 基于 Jinan、Hangzhou 场景注入了消防车、救护车和警车等紧急车辆。 | 1 | 同 Jinan/Hangzhou (4相位) | **泛化性（紧急车辆迁移验证）：** 验证模型能否自适应紧急车辆场景，实现优先放行。 | - |

## 五. 目前已经完成的工作
- **仿真环境打通与数据生成**：已成功集成底层 3D 交通仿真环境基座（TransSimHub / SUMO），实现了交通路口 BEV 图像与路网状态数据的自动化闭环生成。
- **思维链 (CoT) 设计与构建**：确立了基于 `Scene Understanding -> Scene Analysis -> Selection Logic` 的多模态视觉-语言推理决策范式。
- **训练/评测基础设施搭建**：完成了基于 YAML 的系统级配置解耦 (`configs/`)，并编写了支持多 GPU 异步调度的批量评测自动化脚本 (`scripts/run_eval_gpu*.sh`)。
- **SFT (监督微调) 基线**：完成了基础 SFT 数据集的构建与模型微调训练。已跑通完整的端到端评估 Pipeline，SFT 数据集的构建脚本文件夹为（`src/dataset/golden_gener`）
- **DPO（直接偏好对齐）**：完成了DPO数据集构造和模型直接偏好对齐训练。其中DPO的正样本为SFT数据，负样本为 SFT 模型生产，DPO 数据集的构建脚本和数据储存文件夹为（`src/dataset/DPO_data_construct`），最终用于训练的DPO 数据集为`/home/jyf/code/trafficVLM/code/VLMTraffic/src/dataset/DPO_data_construct/dpo_dataset_sft_model_gener_cleaned.jsonl`，其正负样本分析结果为`src/dataset/DPO_data_construct/dpo_dataset_analysis_report.txt`.
- **已有消融实验结果**：
见`results/ablation_result.csv`
- **已有部分对比实验结果**：
见`results/comparsion_result.csv`

## 六. 常用指令与开发规范
**⚠️ 注意：执行所有命令前，请务必确保已激活名为 `VLMTraffic` 的虚拟环境！**
- 
- 代码风格：严格遵循 PEP-8，添加详细的中文注释。
- 模型训练和场景验证部分在远程服务器上进行，因此涉及到模型训练和场景验证的模块不需要运行代码，只需要提供运行代码即可


## 七、关于相关研究文献汇总

### 1.VLLM幻觉

| 论文简称 / 核心亮点 | 论文完整名称 | arXiv 链接 | GitHub 开源代码 |
| :--- | :--- | :--- | :--- |
| **HSA-DPO**<br>*(引入幻觉严重度权重)* | Detecting and Mitigating Hallucination in Large Vision Language Models via Fine-Grained AI Feedback (AAAI 2025) |[2404.14233](https://arxiv.org/abs/2404.14233) | [Mr-Loevan/HSA-DPO](https://github.com/Mr-Loevan/HSA-DPO) |
| **M-HalDetect (FDPO)**<br>*(基于文本块的细粒度惩罚)* | Detecting and Preventing Hallucinations in Large Vision Language Models (AAAI 2024) | [2308.06394](https://arxiv.org/abs/2308.06394) |[hendryx-scale/mhal-detect](https://github.com/hendryx-scale/mhal-detect) |
| **RLHF-V (DDPO)**<br>*(免打分、密集修正反馈)* | RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback (CVPR 2024) | [2312.00849](https://arxiv.org/abs/2312.00849) | [RLHF-V/RLHF-V](https://github.com/RLHF-V/RLHF-V) |
| **ViGoR**<br>*(结合传统CV目标检测)* | ViGoR: Improving Visual Grounding of Large Vision Language Models with Fine-Grained Reward Modeling | [2402.06118](https://arxiv.org/abs/2402.06118) | [amazon-science/vigor](https://github.com/amazon-science/vigor) |
| **SENTINEL (C-DPO)**<br>*(屏蔽上下文、句子级早期干预)* | Mitigating Object Hallucinations via Sentence-Level Early Intervention (ICCV 2025 投稿) |[2507.12455](https://arxiv.org/abs/2507.12455) | [pspdada/SENTINEL](https://github.com/pspdada/SENTINEL) |
| **TPO**<br>*(Token级视觉锚点偏好)* | Token Preference Optimization with Self-Calibrated Visual-Anchored Rewards for Hallucination Mitigation | [ArXiv 检索](https://arxiv.org/search/?query=Token+Preference+Optimization+with+Self-Calibrated+Visual-Anchored+Rewards+for+Hallucination+Mitigation&searchtype=title) | [见 Awesome 仓库](https://github.com/showlab/Awesome-MLLM-Hallucination) |
| **HDPO**<br>*(靶向幻觉目标直接偏好)* | Mitigating Hallucination in Multimodal Large Language Model via Hallucination-targeted Direct Preference Optimization | [ArXiv 检索](https://arxiv.org/search/?query=Mitigating+Hallucination+in+Multimodal+Large+Language+Model+via+Hallucination-targeted+Direct+Preference+Optimization&searchtype=title) | [见 Awesome 仓库](https://github.com/showlab/Awesome-MLLM-Hallucination) |
| **V-DPO**<br>*(视觉特征引导对齐)* | Mitigating Hallucination in Large Vision Language Models via Vision-Guided Direct Preference Optimization (EMNLP 2024) | [ArXiv 检索](https://arxiv.org/search/?query=Mitigating+Hallucination+in+Large+Vision+Language+Models+via+Vision-Guided+Direct+Preference+Optimization&searchtype=title) | [见 Awesome 仓库](https://github.com/showlab/Awesome-MLLM-Hallucination) |
| **EAGLE**<br>*(增强视觉接地防幻觉)* | Enhanced Visual Grounding Minimizes Hallucinations in Instructional Multimodal Models | [ArXiv 检索](https://arxiv.org/search/?query=Enhanced+Visual+Grounding+Minimizes+Hallucinations+in+Instructional+Multimodal+Models&searchtype=title) | [见 Awesome 仓库](https://github.com/showlab/Awesome-MLLM-Hallucination) |
| **F-CLIPScore**<br>*(免训练细粒度视觉打分)* | Vision-Encoders (Already) Know What They See: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore | [ArXiv 检索](https://arxiv.org/search/?query=Vision-Encoders+(Already)+Know+What+They+See%3A+Mitigating+Object+Hallucination+via+Simple+Fine-Grained+CLIPScore&searchtype=title) | [见 Awesome 仓库](https://github.com/showlab/Awesome-MLLM-Hallucination) |
| **ROPE**<br>*(多目标幻觉基准分析)* | Multi-Object Hallucination in Vision-Language Models | [ArXiv 检索](https://arxiv.org/search/?query=Multi-Object+Hallucination+in+Vision-Language+Models&searchtype=title) |[见 Awesome 仓库](https://github.com/showlab/Awesome-MLLM-Hallucination) |
| **CAP**<br>*(约束感知提示缓解空间幻觉)* | Mitigating Hallucinations in Multimodal Spatial Relations through Constraint-Aware Prompting | [ArXiv 检索](https://arxiv.org/search/?query=Mitigating+Hallucinations+in+Multimodal+Spatial+Relations+through+Constraint-Aware+Prompting&searchtype=title) | [见 Awesome 仓库](https://github.com/showlab/Awesome-MLLM-Hallucination) |
| **MASH-VLM**<br>*(时空解耦防幻觉)* | Mitigating Action-Scene Hallucination in Video-LLMs through Disentangled Spatial-Temporal Representations (CVPR 2025) | [ArXiv 检索](https://arxiv.org/search/?query=Mitigating+Action-Scene+Hallucination+in+Video-LLMs+through+Disentangled+Spatial-Temporal+Representations&searchtype=title) |[见 Awesome 仓库](https://github.com/showlab/Awesome-MLLM-Hallucination) |

### 2.LLM for traffic signal control




这里为您将提供的“大语言模型用于交通信号控制（LLM for TSC）”相关文献整理成您所要求的标准 Markdown 表格格式，并**新增了“论文的详细介绍”这一列**，对每篇论文的方法论和技术细节进行了更深度的拆解与补充，方便您进行横向对比和文献综述。

| 论文名称及链接 | 期刊/时间 | 概述 | 论文的详细介绍 (新增) | 评论 / 优缺点分析 | 控制策略 (输入与动作) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VLMLight**: Safety-Critical Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning Architecture<br>🔗[Arxiv 2505](https://arxiv.org/pdf/2505.19486) | 2025<br>arXiv | 提出了融合 VLM、LLM 和 RL 的交通信号控制框架。构建了多视角图像仿真平台，VLM 提取结构化文本，LLM 作元控制器，实现“常规用 RL，紧急用 LLM”的双分支策略，显著降低紧急车辆等待时间。 | **1. 感知层**：利用 Qwen2.5-VL 将路口原始多视角图像转化为结构化文本（如拥堵度、特种车辆）。<br>**2. 元控制层**：LLM 充当“路由器”，根据文本描述判断当前是 Normal 还是 Special 场景。<br>**3. 执行层**：正常场景直接调用 PPO 模型（快分支）；紧急场景激活 Agent 多步推理（慢分支）让出特权绿灯。<br>整体思路是用 LLM/VLM 解决长尾紧急场景，用 RL 保证常规效率。 | **缺点**：<br>1. **推理延迟/功能冗余**：VLM 提取信息 + LLM 分发意图导致链路过长。<br>2. **仿真局限**：自研视觉器缺乏天气、光照变化，影响 sim-to-real 泛化。<br>3. **模型过重无微调**：使用了32B/72B超大模型且未做SFT，极易产生视觉幻觉。建议改用小参数 VLM 结合图像-文本问答对进行 SFT。<br>4. **冗余验证**：验证动作是否合法意义不大，模型可通过约束生成直接控制。 | **感知输入**：多视角 BEV 图像。<br>**控制逻辑**：<br>- 特殊场景：优先放行救护车/消防车。<br>- 常规场景：基于 PPO 历史状态选择动作。<br>- 仿真循环：首步用预训练 PPO 决策并渲染，后续步获取图像 $\rightarrow$ VLM 场景理解 $\rightarrow$ LLM 快慢分支判断 $\rightarrow$ 输出 Phase。 |
| **Traffic-R1**: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control Systems<br>🔗[Arxiv 2508](https://arxiv.org/abs/2508.02344) | 2025<br>arXiv | 提出基于强化学习训练的 3B 参数 LLM（致敬 DeepSeek-R1 范式）。通过两阶段 RL 训练，具备零样本泛化能力，支持边缘设备部署及异步多路口协调，已在真实世界落地管理10个路口。 | **1. 训练范式革命**：放弃传统的 SFT，直接采用“离线专家指导 RL” + “在线开放世界 RL”两阶段训练，赋予模型类人推理（Thinking）能力。<br>**2. 工程落地**：在极小参数量（3B）下实现了边缘设备实时推理，并通过异步通信网络解决了多路口协同的延迟问题。<br>**3. 真实部署**：是少数真正走到线下部署（日均5.5万驾驶员）并验证能降低 9.3% 排队长度的工作。 | **特点**：<br>1. LLM 的输入是纯结构化**文本描述**（提取自雷达或视觉传感器），而非原始图像数据。<br>2. 重点在于“强化学习微调 LLM”在交通领域的落地，证明了小模型+RL的巨大潜力。 | **感知输入**：场景结构化文本描述。<br>**控制逻辑**：模型内部生成长程 CoT 推理，最终输出下一步的最优红绿灯相位（Action）。支持相邻路口的异步状态共享。 |
| **LLM-assisted light**: Leveraging LLM capabilities for human-mimetic TSC in complex urban environments<br>🔗[Arxiv 2403](https://arxiv.org/abs/2403.08337) | 2024<br>arXiv | 提出 LA-Light 框架，将 LLM 作为决策中心“大脑/指挥官”。LLM 通过调用工具（Tools）感知环境，并将现有的 RL 算法作为“顾问”辅助决策，最终输出可解释的方案。 | **典型的 Agent（智能体）架构**：<br>不强求 LLM 自己算出最优解，而是让 LLM 充当调度枢纽。框架包含：<br>- **记忆模块**：记录历史交通状态。<br>- **工具调用**：调用摄像头 API 获取排队长度，调用 RL 算法获取“建议动作”。<br>- **推理决策**：LLM 综合“自己看到的”和“RL 建议的”，做最终拍板。 | **特点**：<br>架构设计非常符合 Agent 哲学，包含了需求理解、工具调用、反馈循环。但同样面临大模型 API 调用带来的高延迟问题，且对 RL “顾问”的依赖度较高。 | **感知输入**：通过调用 Tool 获取的环境状态文本。<br>**控制逻辑**：综合推理后，输出指定的 Phase 控制命令。 |
| **LLMLight**: Large Language Models as Traffic Signal Control Agents<br>🔗[KDD 2025](https://dl.acm.org/doi/abs/10.1145/3690624.3709379) | 2025<br>KDD | 基于 Qwen2-14B 的三阶段对齐训练：GPT-4 收集轨迹 $\rightarrow$ LoRA 模仿微调 $\rightarrow$ Critic 引导的排序损失 (RBC Loss) 后对齐，使小模型具备强大的零样本控制能力。 | **非常经典的对齐（Alignment）管线**：<br>**Stage 1**：利用高级模型（GPT-4）生成含有思维链的决策轨迹，并用预训练的 RL（Advanced-CoLight）作为 Critic 剔除烂数据。<br>**Stage 2**：用筛选后的优质数据对 14B 模型进行 SFT。<br>**Stage 3**：用 Critic 的 Q 值作为奖励信号，通过 RBC Loss 让模型进一步对齐交通效率最大化目标。 | **缺点/反思**：<br>1. **信息压缩过度**：输入给 LLM 的仅是各车道排队长度，丢失了路口几何拓扑、车道数等核心物理信息。<br>2. **为了 LLM 而 LLM**：训练重度依赖 Advanced-CoLight（一个RL模型）来当裁判。这说明 LLM 只是在“拟合”一个性能优异的 RL 模型，但 LLM 的加入确实弥补了纯 RL 在可解释性和跨路口泛化上的短板。 | **感知输入**：文本化排队长度。<br>**控制逻辑**：端到端输出推理过程和动作（下一个绿灯相位）。 |
| **Prompt to Transfer**: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning<br>🔗[AAAI 2024] | 2024<br>AAAI | 利用 LLM 作为世界模型知识库（World Model KB），通过 Prompt Learning 辅助传统 RL 算法适应系统动力学差异，解决 Sim-to-Real 鸿沟。 | **解决仿真到现实的痛点**：<br>RL在仿真器（如 SUMO）中训练得很好，但到了真实世界（雨雪天、刹车距离变长）就崩溃。本文不让 LLM 直接控灯，而是让 LLM 提供“常识”（例如：下雪天车速会变慢10%），用这些常识去动态修正 RL 模型的输入状态或 Reward 函数。 | **特点**：<br>典型的应用型论文，切入点很巧。针对仿真场景中**缺乏恶劣环境真实数据**的痛点，用 LLM 的内部先验知识来做“软补偿”。 | **感知输入**：环境描述（天气、路况）。<br>**控制逻辑**：LLM 输出环境动态调整参数 $\rightarrow$ 传统 RL 结合参数输出具体 Phase。 |
| **The Crossroads of LLM and Traffic Control**: A Study on LLMs in Adaptive TSC<br>🔗[IEEE TITS 25] | 2025<br>TITS | 提出通用能力智能体（GCA），结合 Zero-Shot CoT 与 Actor-Critic 机制，让 GPT-3.5 像人类调度员一样逻辑推理，并根据文本反馈自我修正，效率超越传统感应控制。 | **免微调范式（Zero-shot + Reflection）**：<br>完全没有训练过程。直接把交通状态塞给 GPT-3.5，利用大模型强大的原生推理能力配时。如果配得不好（比如下一秒某个方向排队更长了），Critic 会生成一段“批评文本”，让 GPT-3.5 在下一轮“反思”并调整策略。 | **缺点**：<br>1. **感知极度理想化**：前提是 LLM 能拿到无噪声的完美文本（如“精确的5辆车”），但在真实世界，传感器和 CV 模型不可能100%准。LLM 对感知误差的容忍度存疑。<br>2. **API 延迟**：严重限制实时毫秒级调度。 | **感知输入**：理想化的结构化交通文本。<br>**控制逻辑**：<br>不仅输出选择的 **相位 (Phase)**，还直接输出该相位的 **具体持续时长 (Duration)**。（相比只选相位的模型进了一大步） |
| **LLM-Driven Urban Traffic Signal Control**<br>🔗[ANZCC 2024] | 2024<br>ANZCC | 提出基于 ACP 方法的框架，将 LLM 定位为人类与底层算法的”翻译官”。设计了自主、反馈、人工接管三种模式，强调可解释性与安全性。 | **概念性系统架构**：<br>系统不信任 LLM 算读秒，而是让 LLM 充当”操作员接口”。人类用自然语言下达宏观指令（如”优先疏散东向拥堵”），LLM 负责把这句话翻译成底层算法的代码或参数；同时，把底层的执行结果翻译成人话汇报给操作员。 | **致命缺点**：<br>1. **空洞无物**：通篇只有流程图，没有任何仿真实验、对比基线和具体的 Case Study。<br>2. **大材小用**：LLM 沦为了纯粹的 NLP 翻译机，其最强大的因果推理和时空规划能力在控制环节完全缺席。 | **感知输入**：人类自然语言指令 / 宏观交通报告。<br>**控制逻辑**：翻译为规则/RL 算法的运行参数。 |

### 3. RLVR（Reinforcement Learning with Verifiable Rewards）

RLVR 是以”无需人工偏好标注、用确定性验证器直接给出奖励”为核心的 RL 范式。以下文献从基础算法 → VLM 视觉推理扩展 → 交通仿真应用三个层次覆盖本项目所需背景。

| 论文简称 / 核心亮点 | 论文完整名称 | arXiv 链接 | GitHub 开源代码 |
| :--- | :--- | :--- | :--- |
| **DeepSeek-R1**<br>*(RLVR+GRPO 范式奠基作)* | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025) | [2501.12948](https://arxiv.org/abs/2501.12948) | [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) |
| **R1-V**<br>*(VLM 视觉计数 RLVR，$3 成本触发计数”aha moment”)* | R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3 | [GitHub Only](https://github.com/Deep-Agent/R1-V) | [Deep-Agent/R1-V](https://github.com/Deep-Agent/R1-V) |
| **VLM-R1**<br>*(稳定可泛化的 R1 风格 VLM)* | VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model | [2504.07615](https://arxiv.org/abs/2504.07615) | [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) |
| **CrowdVLM-R1**<br>*(模糊 GRPO 奖励解决人群计数)* | CrowdVLM-R1: Expanding R1 Ability to VLM for Crowd Counting using Fuzzy Group Relative Policy Reward | [2504.03724](https://arxiv.org/abs/2504.03724) | — |
| **PEARL**<br>*(感知证据锚定 RL，防止视觉幻觉与奖励 Hacking)* | Perceptual-Evidence Anchored Reinforced Learning for Multimodal Reasoning | [2511.18437](https://arxiv.org/abs/2511.18437) | — |
| **KAWHI**<br>*(视觉-几何信息注入奖励重加权，兼容 GRPO)* | Bridging Visual Representation and Reinforcement Learning from Verifiable Rewards in Large Vision-Language Models | [2603.27375](https://arxiv.org/abs/2603.27375) | — |
| **SynthRL**<br>*(可验证数据合成扩展 RLVR 训练集)* | SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis | [2506.02096](https://arxiv.org/abs/2506.02096) | — |
| **R1Sim**<br>*(R1 风格 GRPO + 安全奖励用于交通轨迹仿真)* | Learning Rollout from Sampling: An R1-Style Tokenized Traffic Simulation Model | [2603.24989](https://arxiv.org/abs/2603.24989) | — |
| **LaViPlan**<br>*(RLVR 用于语言引导视觉路径规划，ICCV 2025W)* | LaViPlan: Language-Guided Visual Path Planning with RLVR | [ICCV2025W](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/papers/Oh_LaViPlan__Language-Guided_Visual_Path_Planning_with_RLVR_ICCVW_2025_paper.pdf) | — |
| **Traffic-R1**<br>*(见 §2，RLVR 文本 TSC 落地，含真实部署验证)* | Traffic-R1: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control | [2508.02344](https://arxiv.org/abs/2508.02344) | — |
| **CoLLMLight**: Cooperative LLM Agents for Network-Wide TSC<br>🔗[ICLR 2026在投] | 2026<br>ICLR | 将控制范围扩展至**全路网**。引入结构化时空图谱和**复杂度感知推理**（动态调整推理深度），并通过**自定义物理价值函数**构建数据集，进行两阶段 SFT+自监督微调。 | **路网级协同与微调创新**：<br>**1. 动态算力分配**：基于路口繁忙度，决定使用【无协调、简单协调、复杂协调】。复杂时会预测邻近路口的未来状态。<br>**2. 自定义物理价值标签**：没有用 RL 跑数据，而是直接用公式 `V(动作) = 排队时间成本 + 行驶时间成本` 穷举算出一个最大值。把这个物理最优动作作为标签，让模型进行 SFT 和自监督纠错学习。 | **评论/优缺点**：<br>1. **延迟危机**：多路口时空提示词极其复杂，长文本推理必然导致严重延迟。唯一的解法是：保证“推理时间 < 绿灯相位持续时间”，用滞后一个时间步的状态提前做决策。<br>2. 核心亮点在于**放弃了 RL 框架，直接用物理公式造数据集**，降低了工程门槛。 | **感知输入**：当前路口 + 上下游车道状态（占用率、排队时间）。<br>**控制逻辑**：<br>基于动态推理深度，评估未来所有可行 Phase 的综合成本，选择最优 Phase。 |