# E2ELight - 端到端视觉语言模型交通信号控制框架

## 一. 出发点（Introduction）—— 利用 LVLM 的世界知识理解动态交通事件，对常规 / 动态事件进行快 / 慢思考

交通信号控制 (Traffic Signal Control, TSC) 作为城市交通治理的核心环节，直接决定了道路通行效率、车辆排放水平以及紧急救援响应速度。随着城市规模扩张与交通流量剧增，传统的固定配时与感应控制方案已难以应对路口频繁出现的复杂交通事件——救护车接近路口、校车 / 公交车请求优先通行、前方车祸导致单向堵死、散落货物占用车道、施工临时改变车道归属、学校放学时段大量行人涌出等长尾场景。此类事件对决策的影响是决定性的：一次错误的相位选择可能延误紧急车辆数十秒，甚至造成生命财产损失。因此，TSC 的真实难点不在于稳态车流的统计优化，而在于**对动态交通事件的语义理解与即时响应**。

近年来，基于强化学习 (Reinforcement Learning, RL) 的方法 [Wei et al., 2019; Chen et al., 2020] 通过端到端学习路口状态到信号相位的映射，在标准场景下取得显著性能提升。然而，主流 RL 方案依赖卷积神经网络 (CNN) 或目标检测器从原始视觉输入中提取结构化状态向量（如各车道车辆数、平均速度），再交由策略网络输出动作。这种"感知-决策"流水线存在根本性局限：CNN 所输出的状态表征**缺乏语义理解能力**——它无法识别其中一辆是救护车，更无法激活"紧急车辆在场必须立即清空其行进方向"这类交通常识。任何基于固定标注类别训练的判别式模型，其理解边界均被刚性限定于训练分布之内，面对分布外事件时只能将其视作"未知形状"予以忽略。这种语义损失并非工程实现层面的缺陷，而是**判别式架构的根本性局限**——此类方法既无法对事件进行"理解"，更无法推理"事件对未来交通流的影响"。

近年来，已有研究开始尝试将大语言模型 (LLM) 引入 TSC 决策环节，代表性工作包括 LLMLight [Lai et al., 2024]、LA-Light [Wang et al., 2024] 与 VLMLight [Zhang et al., 2025] 等。这些方法普遍采用"感知 → 描述 → 决策"的**多阶段多模块流水线**架构：视觉信号需依次经过 VLM / 检测器（生成文字描述）、LLM 推理器（生成决策）、外部路由器（在常规与事件场景间切换执行分支）等多个异构模块的串行处理后，方可输出最终相位。以 VLMLight 为例，其完整推理流程涉及 VLM 场景摘要、LLM 控制、RL 快分支与 LLM 慢分支等至少 4 个独立子模块的协同运行；LLMLight 虽未引入外部路由，但视觉感知与决策推理仍被分离至视觉前端与 LLM 决策器两个异构环节。此类**非端到端流水线架构**存在三类显著缺陷：① **视觉语义的二次损失**：将高维视觉场景强行压缩为数句文本描述后，紧急车辆的视觉特征、车辆空间分布、队列几何形态等关键结构信息被大幅削弱，下游推理模型所接收的已是退化的观测表征，且文本描述环节本身可能引入幻觉性噪声；② **LVLM 固有的推理能力未被利用**：此类方法并未将 LVLM 作为统一推理主体，而是将其降格为前置判别器或摘要器，真正的决策推理被剥离至下游 LLM 或外部 RL 分支完成——这种设计割裂了视觉感知与语义推理，延长了推理链路，也使 LVLM 本可胜任的多步思考能力完全闲置；③ **评测场景覆盖不足**：LLMLight 等工作仅在常规车流路网上验证其 CoT 决策能力，**完全未涉及紧急车辆、异常拥堵等事件场景**；VLMLight 虽引入紧急车辆测试，却将实验**局限于单一标准 4 叉路口**，没有验证跨拓扑（T 字路口、左行特例、非对称多车道）与跨规模（大规模路网）的泛化能力。现有工作普遍缺乏对"**事件迁移 × 拓扑迁移 × 规模迁移**"三维泛化能力的系统性验证。

此外，我们的核心洞察是：**交通事件的正确响应本质上是一个"慢思考"任务——它需要先识别事件、再推理事件对未来交通流的影响、最后选择适宜的响应动作；而常规车流场景则仅需要"快思考"——基于当前压力即可直接完成决策**。现有双阶段方法将"快思考"与"慢思考"分配给两个不同的模型承担，其隐含假设是单一模型难以同时胜任这两种截然不同的推理模式。LVLM 的出现改变了这一前提：在互联网规模图文语料上预训练的 LVLM 已将"视觉感知 → 语义理解 → 操作知识"内化为联合表征，使其既能在常规场景直接生成短推理链完成快速决策，也能在事件场景展开多步推理链进行深入分析。关键在于，这两种推理模式本就内嵌于同一模型的联合表征之中——**通过适当的任务对齐即可将其显式激活，在单一 LVLM 内部完成快慢思考的自适应切换，无需引入外部路由器或双分支架构**。

基于上述洞察，我们提出 **E2ELight**——一个面向交通信号控制的**端到端** LVLM 框架：从多视角进口道图像输入到最终相位决策，整个决策流程**由一个微调 LVLM 通过单次前向推理直接完成**，系统中不存在 LVLM 之外的任何独立模型、检测器、文本摘要模块、外部路由器或下游决策器。这种极简的端到端设计直接消除了多阶段流水线架构的所有耦合点与失败源。

在此端到端骨架之上，E2ELight 的方法论核心是**单模型自适应的快慢思考 CoT**：常规车流场景下，模型触发**短路径推理** `Scene Understanding → Phase Selection`，以最小开销完成高效决策；事件场景下，模型触发**长路径推理** `Scene Understanding → Event Recognition → Impact Reasoning → Scene Analysis → Selection Logic → Phase`，依次完成**事件类型识别**（识别出当前路口属于**紧急车辆、校车与公交车、交通事故、占道、行人过街**六类典型交通事件中的何种，或不存在事件）、**事件影响推理**（该事件对未来若干仿真步内各方向交通流的影响）、以及最终的**最优相位选择**。与 VLMLight、LLMLight 等非端到端方法相比，E2ELight 的差异是结构性的：**我们没有把"快"和"慢"分配给两个不同的模型，也没有把"视觉"和"决策"分离到两个不同的模块——所有感知、理解、推理与决策行为统一发生在 LVLM 的同一次生成过程之中**，快慢思考的切换完全由 LVLM 自身基于视觉观察在 CoT 生成过程中自适应决定，既充分发挥了 LVLM 固有的多步推理能力，也从根本上规避了异构模块拼接带来的架构冗余与信息失真。

为使上述核心机制在真实部署场景中可行，E2ELight 包含两个关键支撑决策：

**(i) 多视角进口道图像输入。** E2ELight 不采用难以大规模部署的全局 BEV（其获取通常需要专用高架摄像头或多目重建设备）。我们将每个路口的视觉输入定义为**各进口道方向独立俯拍图像**的集合——即一个四进口道路口对应 4 张输入图像——这些图像可直接取自国内"天网 / 雪亮工程"已广泛部署的杆式摄像头，无需任何新增硬件。相比全景 BEV，多视角进口道输入具有三重优势：部署门槛显著降低、单视角拍摄距离近且目标密度较低（有效缓解多目标计数幻觉）、事件对象的关键视觉特征——紧急车辆警灯与警示标识、校车 / 公交车的车身涂装与尺寸、交通事故导致的车辆异常停驻与碰撞姿态、散落货物与路面占用物的几何轮廓、行人群体的密度与过街动线、施工围挡与锥桶的几何标识——在近距离视图下均更为清晰可辨，共同为慢思考路径中的事件识别子步骤提供了可靠的视觉证据。

**(ii) 仿真器反馈的闭环对齐。** 我们的对齐流水线采用 **SFT + 在线 RLVR** 两阶段设计：首先通过 SFT 建立基础任务对齐，使模型掌握快慢思考双 CoT 模板的正确格式与基本决策逻辑；进而引入基于仿真器**双路可验证奖励**的在线 RLVR——以 SUMO e2 检测器的真实排队车辆数作为**感知奖励**缓解计数幻觉，以 rollout 若干仿真步后的 ATT/AQL 变化量作为**效率奖励**直接量化决策质量——将物理交通效率作为终极监督信号直接作用于 LVLM 训练过程，实现感知准确性与决策质量在统一梯度下的同步约束。

**本文的主要贡献总结如下：**

- 我们提出了**首个真正端到端的 TSC LVLM 框架 E2ELight**——从多视角视觉输入到信号相位决策全程由单一微调 LVLM 的**单次前向推理**完成，无需任何前置检测器、中间文本描述、外部路由器或下游决策模块。在此端到端骨架之上，E2ELight 通过**单模型自适应快慢思考 CoT** 实现对常规与事件场景的统一应对：常规场景触发短路径快速决策，事件场景展开 `Event Recognition → Impact Reasoning → Selection Logic` 的长路径慢思考。相较 LLMLight、VLMLight 等多阶段多模块流水线方法，E2ELight 在架构简洁性、推理链路长度、部署可维护性以及 LVLM 固有推理能力的利用率等方面均具显著优势。

- 我们提出了**面向视觉幻觉的双路可验证奖励在线 RLVR 对齐机制**。以 SUMO 仿真器作为终极裁判，对 LVLM 的每一次 CoT 输出即时给出两类硬标签奖励：感知奖励（e2 检测器排队车辆数 vs 模型预测值）与效率奖励（rollout 后的 ATT/AQL 变化量），通过 GRPO 范式完成在线强化学习微调。该机制将物理交通效率作为可验证奖励信号直接作用于 LVLM 训练过程，实现感知准确性与决策质量在统一梯度下的同步优化。这是交通领域首个基于**视觉-仿真闭环**的 RLVR 框架。

- 进行了**覆盖多种动态交通事件的系统化评测**。在 JiNan / Hangzhou 真实监控数据基础上，进一步在 SouthKorea_Songdo（非对称 6 车道大型路口）、France_Massy（T 字路口）、Hongkong_YMT（左行特例）、NewYork（196 路口大规模路网）以及**六类交通事件注入场景**（紧急车辆、校车与公交车、交通事故、占道（施工、抛洒物）、行人过街）上验证模型的零样本泛化能力，同时覆盖**拓扑迁移、规模迁移、事件迁移**三个维度。这一评测体系直接针对 LLMLight 仅在常规路网验证、VLMLight 仅在单一 4 叉路口验证紧急车辆的短板，提供了当前 LLM-TSC 文献中最全面的泛化能力测试。实验结果表明 E2ELight 在所有场景下均显著优于固定配时、MaxPressure 以及现有 LLM 增强方法。

---

针对上述命题，本项目的两个核心贡献及后续待验证的优化方向具体展开如下：

- **贡献 1：提出了真正端到端的 TSC LVLM 框架，实现单模型自适应快慢思考 CoT**
  以**单路口各进口道方向的多视角俯拍图像**（4 进口道路口对应 4 张输入）作为视觉输入，使用微调后的 LVLM 通过**单次前向推理**直接输出 CoT 推理链与信号控制决策。整个系统中不存在 LVLM 之外的任何独立检测器、文本摘要模块、外部路由器或下游决策器。在此端到端骨架之上，E2ELight 通过**快慢思考双 CoT 模板**应对不同场景：常规场景触发**短路径** `Scene Understanding → Phase Selection`（快思考），事件场景触发**长路径** `Scene Understanding → Event Recognition → Impact Reasoning → Scene Analysis → Selection Logic → Phase`（慢思考），快慢切换完全由 LVLM 自身基于视觉观察在 CoT 生成过程中自适应决定，无需外部路由器或双分支架构。

- **贡献 2：针对"多目标幻觉（Multi-object Hallucination）尤其是计数幻觉（Counting Hallucination）和位置幻觉（Positional Hallucination）"的问题，提出基于仿真器双路可验证奖励的 LVLM 在线强化学习 (Simulation-Grounded Dual-Verifiable RLVR for LVLM-TSC)**
  - **目前痛点**：现有 LLM-TSC 工作在对齐阶段普遍依赖 SFT 或离线 DPO——前者信号单一、无法治愈视觉幻觉；后者需手工构造偏好对、信号稀疏、感知与决策孤立优化。而 SUMO 仿真器对每一次信号决策都能即时提供两类硬标签可验证信号，却完全未被利用：(1) e2 检测器提供每条 movement 的真实排队车辆数，可直接验证模型 Scene Understanding 的计数准确性；(2) rollout 若干仿真步后的 ATT/AQL 变化量，可直接量化信号决策的交通效率。
  - **核心创新**：提出以 GRPO（Group Relative Policy Optimization）范式对 LVLM 进行在线强化学习微调，完全摆脱偏好对构造的依赖。定义双路可验证奖励：**感知奖励 $r_{\text{perc}}$**（SUMO `jam_length_vehicle` 与模型预测车辆数的归一化误差之负数，直接治愈计数幻觉）和 **效率奖励 $r_{\text{env}}$**（执行所选相位后 30 仿真步的 $\Delta \text{ATT}$，作为交通效率终极裁判）。联合奖励 $r = \alpha \cdot r_{\text{perc}} + \beta \cdot r_{\text{env}}$ 使感知正确性与控制决策质量得到统一梯度信号的同步约束。
  - **GRPO 训练流程**：对同一组多视角进口道图像输入采样 $G=8$ 个完整 CoT 响应（可为短路径或长路径）；用双路奖励对每个响应打分；计算组内相对优势 $A_i = (r_i - \text{mean}(r)) / \text{std}(r)$；GRPO 梯度更新 LVLM，无需 Critic 网络和 Value Model。
  - **相对其他对齐范式的优势**：相较于离线 DPO 的静态偏好信号，RLVR 为在线动态验证信号——不依赖人工标注、仿真器即裁判、模型可自发涌现精确计数与事件响应策略（类 DeepSeek-R1 "aha moment" 现象）。Traffic-R1（2025）已证明 RLVR 范式在纯文本 TSC 中的有效性，本方向将其首次扩展至**多视角进口道图像输入 + 快慢思考 CoT** 场景，是交通领域首个视觉-仿真闭环 RLVR 框架。
  - **落地可行性**：e2 检测器（$r_{\text{perc}}$ 数据源）与 SUMO rollout（$r_{\text{env}}$ 数据源）均为项目现有基础设施，无需新增硬件；可在 SFT checkpoint 基础上直接开启 RLVR 训练，实施路径清晰。

## 二. 项目概览
本项目实现了一个用于交通信号控制 (TSC) 的**真正端到端**视觉语言大模型 (LVLM) 框架。该框架以**单路口各进口道方向的多视角俯拍图像**（一个四进口道路口对应 4 张输入图像）+进口道上游道路的俯拍图像（一个进口道有一张）为视觉输入，由单一微调 LVLM 通过**单次前向推理**直接输出相位决策，系统中不存在 LVLM 之外的任何独立检测器、文本摘要模块、外部路由器或下游决策器。方法论核心为**单模型自适应快慢思考 CoT**：常规场景触发短路径 (`Scene Understanding → Phase Selection`)，事件场景触发长路径 (`Scene Understanding → Event Recognition → Impact Reasoning → Scene Analysis → Selection Logic → Phase`)。项目包含完整 Pipeline，涵盖基于 SUMO/TransSimHub 的仿真与多视角进口道图像生成、SFT 数据构建、SFT 训练、基于仿真器双路可验证奖励的在线 RLVR 微调，以及覆盖常规、拓扑迁移、规模迁移与事件迁移的端到端评估。

相位空间选择：相位选择和相位时间的动态生成：考虑上游道路在绿灯期间预期到达车辆数，决定下一个相位以及该相位绿灯时间（绿灯时间为一个候选合集：10、15、20、25、30、35）（候选空间大小 n*m）
## 三. 核心目录与模块
- `data/`：存放原始仿真数据、生成的多视角进口道图像以及处理后的 SFT 数据集与 RLVR rollout 缓存。
- `models/`：用于存储从 ModelScope 等下载的基础 LVLM 模型和训练后生成的 Checkpoints。
- `src/`：框架核心代码库。
  - `bev_generation/`：负责与仿真环境交互并生成单路口多视角进口道图像。
  - `dataset/`：负责 SFT 训练数据集的构建与处理。
  - `inference/`：LVLM 模型推理与 Prompt 构建逻辑。
  - `training/`：SFT 与基于 GRPO 的在线 RLVR 训练核心模块。
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

- 3 事件场景验证数据集


| 数据集/场景名称 | 简介与特点 | 交叉口数量 | 相位描述 (Action Space for Prompt) | 适用性与用途 | 车流特征摘要 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Emergency** | 基于 Jinan、Hangzhou 场景注入了消防车、救护车和警车等紧急车辆。 | 1 | 同 Jinan/Hangzhou (4相位) | **泛化性（紧急车辆迁移验证）：** 验证模型能否自适应紧急车辆场景，实现优先放行。 | - |
| **School/City Bus** | 注入校车与城市公交车，触发优先通行响应。 | 1 | 同 Jinan/Hangzhou (4相位) | **泛化性（校车与公交车迁移验证）** | - |
| **Traffic Accident** | 注入车辆碰撞与异常停驻，导致单向堵死。 | 1 | 同 Jinan/Hangzhou (4相位) | **泛化性（事故场景迁移验证）** | - |
| **Road Debris** | 注入散落货物、道路施工和路面占用物。 | 1 | 同 Jinan/Hangzhou (4相位) | **泛化性（占道（施工、抛洒物）占用迁移验证）** | - |
| **Pedestrian Crossing** | 注入大规模行人过街（如学校放学时段）。 | 1 | 同 Jinan/Hangzhou (4相位) | **泛化性（行人过街迁移验证）** | - |

## 五. 目前已经完成的工作
- **仿真环境打通与数据生成**：已成功集成底层 3D 交通仿真环境基座（TransSimHub / SUMO）。渲染输出已从全局 BEV 升级为**8张多视角图像**（4张进口道停止线视图 + 4张上游道路视图），具体架构见第八节。
- **训练/评测基础设施搭建**：完成了基于 YAML 的系统级配置解耦 (`configs/`)，并编写了支持多 GPU 异步调度的批量评测自动化脚本 (`scripts/run_eval_gpu*.sh`)。
- **泛化性验证场景测试基础设施**：
  - 完成了 4 大泛化场景（SouthKorea_Songdo、France_Massy、Hongkong_YMT、NewYork）的评测基础设施搭建。
  - 清理了泛化场景路由文件中混入的紧急车辆（France_Massy 5辆、Hongkong_YMT 6辆、SouthKorea_Songdo 8辆），替换为普通 background 车辆类型（`scripts/clean_emergency_vehicles.py`）。
  - 针对各场景路口拓扑差异（T字路口、左行特例、非对称5/6车道路口、196路口大路网），在 BEV 图像上实现了分场景类别的车道数字水印叠加（`scripts/add_lane_watermarks.py`），并支持文字旋转以对齐停止线方向。
  - 完成泛化性指标收集脚本（`src/evaluation/generalization_metrics.py`）及结果模板（`results/generalization_result.csv`），覆盖 ATT、AWT、AQL 三项指标，支持 FixedTime、MaxPressure 和 VLM 方法横向对比。
  - 完成合并版批量评测脚本（`run_batch_generalization.sh`），支持 `--baseline-only`、`--port/--model_name`、`--with-baseline` 三种运行模式，并从路由文件动态计算 `max_steps`（公式：`ceil((max_depart + 300s) / 30s)`，上下限 [10, 200]）。
- **事件场景测试基础设施（全部 5 类）**：所有五类事件场景（紧急车辆、校车与公交车、交通事故、占道/路面碎片、行人过街）的路由文件生成脚本与评测脚本均已完成。
  - 使用 `scripts/event_scene_generation/` 下的脚本为全部 6 个数据集生成各类事件路由文件（`_emergy` / `_bus` / `_accident` / `_debris` / `_pedestrian` 后缀）；`batch_generate_all_scenes.sh` 一键批量生成。 **⚠️ 行人渲染效果较差，暂不考虑**
  - `scripts/event_scene_generation/visualize_event_network.py` 为每个数据集生成标注各类事件位置的路网可视化图（暗色主题，彩色标记）。
  - `src/bev_generation/online_bev_render.py` 通过 `RENDER_EVENT_TYPES` 列表支持 6 种事件类型（含 normal）的顺序渲染，输出到 `data/test/{SCENARIO}/{event_type}/{step}/`。
  - TransSimHub `vehicle.py` MODEL_MAPPING 修复 `pedestrian` 键重复 bug，新增 `HIGH_POLY_ONLY` 集合（bus / school_bus / crash_vehicle / pedestrian_* 等仅存在于 high_poly 资产中的车型）。
- **评测主脚本（`src/evaluation/run_eval.py`）**：从根目录 `vlm_decision.py` 重构迁入 `src/evaluation/`，保留 `--fixed_time`、`--max_pressure`、VLM 三种模式接口。**输出路径统一为 `data/eval/{dataset}/{route_file_name}/{method}/`**。                                                                                                                         
  - **批量评测脚本**：         
  - `scripts/run_batch_event_eval.sh`：事件场景批量评测，覆盖全部 5 类事件 × 6 个数据集；支持 `--event {type}` 单类过滤；调用 `src/evaluation/run_eval.py`。            
  - `run_batch_eval.sh`、`run_batch_generalization.sh`：已更新为调用 `src/evaluation/run_eval.py` 并传入对应 `--scene_type`（JiNan/Hangzhou 为 `normal`，NewYork triple +变体为 `normal_triple`）。 
  - **指标设计（各场景 5 项核心指标）**：
    - 紧急车辆：ATT / AWT / AQL / **EATT** / **EAWT**（Emergency ATT/AWT，按 vType=emergency/police/fire_engine 过滤）
    - 校车与公交车：ATT / AWT / AQL / **BATT** / **BAWT**（Bus ATT/AWT，按 vType=bus/school_bus 过滤）
    - 交通事故：ATT / AWT / AQL / **MaxQL** / **TPT**（峰值排队长度 / 普通车辆通行数，过滤 accident_* 事件车辆）
    - 路面碎片：ATT / AWT / AQL / **MaxQL** / **TPT**（过滤 debris_* 事件车辆）
    - 行人过街：ATT / AWT / AQL / **MaxQL** / **TPT**（过滤 ped_* 事件车辆）                                                                                                      
  - `src/evaluation/metrics.py` 扩展 `calculate_from_files()` 接口：新增 `event_id_prefixes` 参数、`MaxQL`（全程队列峰值）、`TPT`（到达普通车辆计数）、`special_vtypes` 可覆盖。 
  - **统一指标收集脚本（`src/evaluation/collect_metrics.py`）**：替代 6 个独立 `*_metrics.py` 脚本，通过 `--type {generalization|emergency|bus|accident|debris|pedestrian|all}` 统一入口 ；`scene_to_cols` 以 `(scenario, route_file_name)` 为键映射 CSV 列；路径格式与 `run_eval.py` 输出一致（`data/eval/{dataset}/{route_file_name}/{method}/`）。                                       
   - 结果模板：`results/bus_result.csv`、`accident_result.csv`、`debris_result.csv`、`pedestrian_result.csv`、`emergency_result.csv`、`generalization_result.csv`。     

## 六. 3D 仿真渲染架构说明

### 事件场景双线渲染机制

TransSimHub 的 3D 渲染存在**两条完全独立的渲染通路**，不同事件类型走不同通路，不可混淆：

| 事件类型 | SUMO 中的 vType | 渲染通路 | 负责模块 |
| :--- | :--- | :--- | :--- |
| 紧急车辆（ambulance / police / fire_truck） | 真实运动车辆 | `scene_sync` → `Vehicle3DElement` | `SceneSync._manage_vehicle_element()` |
| Bus / School Bus | 真实运动车辆 | `scene_sync` → `Vehicle3DElement` | `SceneSync._manage_vehicle_element()` |
| 路障（barrier_A~E） | trip+stop 占位假车 | `EventManager` → `EmergencyManager3D` → `Emergency3DElement` | `tshub_env3d.emergency_renderer` |
| 碰撞残骸（crash_vehicle） | trip+stop 占位假车 | `EventManager` → `EmergencyManager3D` → `Emergency3DElement` | `tshub_env3d.emergency_renderer` |
| 倒地/过街行人（pedestrian_lying/crossing） | trip+stop 占位假车 | `EventManager` → `EmergencyManager3D` → `Emergency3DElement` | `tshub_env3d.emergency_renderer` |
| 路障封闭区矩形（ClosureZone）归路障占道 | 无 SUMO 实体 | `EventManager` → `EmergencyManager3D` → `ClosureZone3DElement` | `tshub_env3d.emergency_renderer` |

**通路一（动态车辆）**：SUMO 每步返回实时位置/朝向 → `scene_sync` 创建/更新 `Vehicle3DElement`。
**通路二（静态障碍物）**：`__init__` 时一次性解析路由文件 `<param>` 标签获取固定坐标 → 每步按时间窗口过滤活跃事件 → `EmergencyManager3D.update()` 增删 3D 节点。

**关键过滤器**：`scene_sync._is_event_vehicle()` 识别 trip+stop 假车并跳过，防止在 SUMO stop 位置产生幽灵背景车辆。匹配规则：精确匹配 `_EVENT_VTYPE_EXACT` 集合（`crash_vehicle`、`pedestrian_*`、`barrier_A~E`、`tree_branch_*`）或前缀匹配 `barrier_`、`tree_branch_`（兼容带长度后缀的变体如 `barrier_A_5.00`）。


## 七. 常用指令与开发规范
**⚠️ 注意：执行所有命令前，请务必确保已激活名为 `VLMTraffic` 的虚拟环境！**
- 
- 代码风格：严格遵循 PEP-8，添加详细的中文注释。
- 模型训练和场景验证部分在远程服务器上进行，因此涉及到模型训练和场景验证的模块不需要运行代码，只需要提供运行代码即可

---

## 八. 模型输入输出与动作空间定义（已实现）

### 8.1 视觉输入：8张多视角图像

每个路口在每个决策步提供 **8张图像**，顺序固定：

| 序号 | 类型 | 内容描述 | 传感器命名（element_id） |
| :--- | :--- | :--- | :--- |
| Image 1 | 进口道停止线视图 | North 进口，车辆朝南排队 | `{jid}_N` |
| Image 2 | 进口道停止线视图 | East 进口，车辆朝西排队 | `{jid}_E` |
| Image 3 | 进口道停止线视图 | South 进口，车辆朝北排队 | `{jid}_S` |
| Image 4 | 进口道停止线视图 | West 进口，车辆朝东排队 | `{jid}_W` |
| Image 5 | 上游道路视图 | North 上游来车（预计绿灯期间到达） | `upstream_{jid}_N` |
| Image 6 | 上游道路视图 | East 上游来车 | `upstream_{jid}_E` |
| Image 7 | 上游道路视图 | South 上游来车 | `upstream_{jid}_S` |
| Image 8 | 上游道路视图 | West 上游来车 | `upstream_{jid}_W` |

**进口道摄像头**：安装在停止线处，俯拍进口道排队车辆（对应现有 `junction_front_all` 传感器，已有基础设施）。  
**上游摄像头**：安装在进口道来车方向的上游路段（即上游路口的出口车道处），拍摄正在驶来的车辆。摄像机位置取自 `BaseTLS.in_road_upstream_point`（车道形状的起始点 `shape[0]`），方向与车辆行驶方向一致。

**方向索引约定**：N=0，E=1，S=2，W=3（顺时针）。方向由 SUMO 车头朝向推算：`approach_bearing = (heading + 180°) % 360°`，再映射到最近的 N/E/S/W 象限。

**传感器图像 key 格式**（`sensor_imgs` 字典）：
```
进口道：sensor_imgs["{jid}_{dir}"]["junction_front_all"]
上游：  sensor_imgs["upstream_{jid}_{dir}"]["junction_front_all"]
```

**图像采集函数**：`Evaluator._collect_8_images(jid, sensor_imgs, step_dir)` → `List[str | None]`（有序8条路径，缺失为None）。

---

### 8.2 动作空间：联合（相位 × 绿灯时长）

动作空间从"单相位选择"升级为**联合动作空间**：

```
action = {
    'phase_id': int   ∈ [0, num_phases)    # 下一个绿灯相位编号
    'duration': int                         # 实际绿灯时长（秒），VLM 模式取自候选集
}
```

**绿灯时长常量定义**（`src/utils/tsc_env/tsc_wrapper.py`，所有模块统一引用）：
```python
GREEN_DURATION_CANDIDATES = [10, 15, 20, 25, 30, 35]  # VLM 可选绿灯时长（秒）
FIXED_TIME_GREEN_DURATION = 27  # FixedTime / MaxPressure 专用：27s + 3s 黄灯 = 30s 整步
```

**时长选择依据**（VLM 模式）：
- Images 1-4（停止线排队）→ 当前相位的实际压力
- Images 5-8（上游来车）→ 绿灯期间的预期到达量
- 排队长 / 上游车多 → 选择较长时长；轻流量 → 选择较短时长

**VLM 输出校验流程**（`run_eval.py`）：
1. 解析 `Action: phase=X, duration=Y` 得到 `raw_dur`
2. 吸附：`actual_dur = min(GREEN_DURATION_CANDIDATES, key=λ x: |x - raw_dur|)`
3. 若 `raw_dur ∉ GREEN_DURATION_CANDIDATES`，记录 WARNING 日志
4. 防御性 `assert actual_dur in GREEN_DURATION_CANDIDATES`（保证最终合法）
5. 传入 `{'phase_id': p_id, 'duration': actual_dur}` 给环境

**底层执行格式**（TransSimHub / SUMO 层）：`(phase_id: int, green_duration: int)`，即实际秒数元组。

**格式兼容性**（`TSCEnvWrapper._decode_action`）：
| 传入格式 | 说明 |
| :--- | :--- |
| `{'phase_id': X, 'duration': Y}` | **推荐格式**，直接使用实际绿灯秒数 |
| `{'phase_id': X, 'duration_idx': Y}` | 索引格式，向后兼容旧代码 |
| `(phase_id, green_duration_seconds)` | 元组格式（TransSimHub 内部） |
| `int` | 旧格式兼容（duration 取 FIXED_TIME_GREEN_DURATION=27s） |

---

### 8.3 Prompt 输出格式

```
Action: phase=<phase_id>, duration=<seconds>
```

示例：`Action: phase=1, duration=25`

**CoT 模板结构（标准输出）**：
```
Thought: [
Scene Understanding:
- Lane Analysis (Mandatory): <各进口道各车道排队数>
- Phase Mapping: Phase ID (<方向>): <拥堵等级> | <原因>
Scene Analysis:
- Emergency Check: <None 或 事件描述>
- Final Condition: <Normal / Special>
Selection Logic:
- Rule Identification: <规则名>
- Reasoning: <一句话原因>
- Conclusion: Phase <ID>
Duration Selection:
- Stop-line queue pressure (Images 1-4): <简要评估>
- Upstream arrival estimate (Images 5-8): <简要评估>
- Selected Duration: <X> seconds | Reasoning: <一句话>
]

Action: phase=<phase_id>, duration=<seconds>
```

**VLMAgent 解析逻辑**（`_parse_action`）：优先匹配新格式 `phase=X, duration=Y`；duration 若不在候选集则吸附到最近合法值；兼容旧格式 `Action: X`（duration 默认 25s）。

---

### 8.4 上下游协同机制：EventBulletin（已实现）

路口间异步事件广播板，使检测到交通事件的路口能将影响告知下游路口，从而在 Prompt 层面实现协同决策。

#### 设计原则

| 维度 | 设计决策 |
| :--- | :--- |
| **架构层次** | 纯 Prompt 层协同，不修改仿真底层（TransSimHub/SUMO）和任何 RL/VLM 权重 |
| **通信时序** | 异步——路口 A 决策完才广播，路口 B 在下一步才读到，符合现实无线通信延迟 |
| **拓扑推断** | 从 `infos['vehicle_next_tls']`（SUMO subscription 112）自动统计投票，无需手动配置邻居表 |
| **过期机制** | TTL = `ceil(green_duration / 30)`步，与选定绿灯时长动态绑定，避免陈旧事件误导决策 |
| **触发条件** | VLM CoT 输出中 `Final Condition: Special`，且 `Emergency Check` 行非 None |

#### 数据流

```
[路口 A VLM 推理]
    ↓ CoT 输出 "Final Condition: Special"
EventBulletin.broadcast(from_jid=A, green_duration=25, ...)
    ↓ 解析事件类型 + 描述
    ↓ get_downstream(A) → [B, C]（票数 ≥ 2 的下游路口）
    ↓ 写入 _board[B], _board[C]，expires_at = step + ceil(25/30) = step+1
    ↓ logger.info 记录广播日志（含来源、目标、事件类型、TTL）

[下一决策步，路口 B 构建 Prompt]
EventBulletin.get_context(jid=B, current_step)
    ↓ 返回未过期通知的文本摘要
PromptBuilder.build_decision_prompt(..., coordination_context=ctx)
    ↓ 若 ctx 非空，在 Prompt 第 6 节插入 "Upstream Coordination Context [ACTIVE]"
    ↓ VLM 看到上游事件描述，在 Duration Selection 和 Selection Logic 中主动调整
```

#### 日志体系

所有协同相关操作均写入 loguru logger，分级如下：

| 日志级别 | 触发场景 | 示例内容 |
| :--- | :--- | :--- |
| `INFO` | 广播成功 | `[Bulletin][广播] J1 → J3 \| 事件类型: emergency_vehicle \| TTL: 1步 \| 描述: Ambulance detected...` |
| `INFO` | Prompt 注入 | `[Bulletin][注入] J3 收到上游协同通知，已注入 Prompt: ...` |
| `INFO` | 过期清理（批量） | `[Bulletin][过期清理] 本步共清除 2 条过期通知` |
| `INFO` | 拓扑汇总（每20步） | `[Bulletin][拓扑] J1 → J3 (票数=15)` |
| `INFO` | 评测结束拓扑汇总 | 全局路口有向图快照 |
| `DEBUG` | 无下游路口时跳过 | `[Bulletin] J1 检测到事件但尚无已知下游路口` |
| `DEBUG` | 单条过期清理 | `[Bulletin][过期清理] J3 清除 1 条过期通知` |

#### 核心文件修改清单（含协同机制）

| 文件 | 修改内容 |
| :--- | :--- |
| `TransSimHub/.../tls_type/choose_next_phase_with_duration.py` | **新增** TLS 动作类型，支持联合相位+时长决策，实现异步决策 |
| `TransSimHub/.../traffic_light_action_type.py` | 新增枚举值 `ChooseNextPhaseWithDuration` |
| `TransSimHub/.../traffic_light.py` | 注册新动作类型；新增 `in_road_upstream_point`；`control_traffic_light()` 支持元组动作 |
| `TransSimHub/.../tls_type/base_tls.py` | 新增 `in_road_upstream_point` 计算（lane shape[0]） |
| `TransSimHub/.../scene_sync.py` | 方向映射；element_id 改为 `{jid}_{dir_short}`；新增 upstream 摄像机 |
| `configs/scenairo_config.py` | sensor_cfg 改为 `{tls_id: cfg}` 格式，新增 `upstream` 配置 |
| `configs/env_config.py` | `tls_action_type` → `choose_next_phase_with_duration`；新增候选集 |
| `configs/prompt_builder.py` | 8图描述；Duration Selection CoT 块；`coordination_context` 参数；协同章节动态注入 |
| `src/utils/tsc_env/tsc_wrapper.py` | `GREEN_DURATION_CANDIDATES`；联合 action_space；`_decode_action()`；`vehicle_next_tls` 透传 |
| `src/utils/tsc_env/tsc_env.py` | sensor_cfg upstream 透传 |
| `src/inference/vlm_agent.py` | 多图输入；`_parse_action()` 返回 `(phase_id, duration)` 元组 |
| `src/evaluation/run_eval.py` | **新增** `EventBulletin` + `EventNotice` 类；协同广播/读取/过期清理/拓扑推断全流程；增强步骤日志 |
| `tests/test_upgrade.py` | 升级验证测试（PromptBuilder / _parse_action / _decode_action / 批量接口 / 候选集一致性） |

---

## 九、关于相关研究文献汇总

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