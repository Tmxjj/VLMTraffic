# E2ELight - 端到端视觉语言模型交通信号控制框架

## 一. 出发点（Introduction）—— 利用 LVLM 的世界知识理解动态交通事件，对常规 / 动态事件进行快 / 慢思考

交通信号控制 (Traffic Signal Control, TSC) 作为城市交通治理的核心环节，直接决定了道路通行效率、车辆排放水平以及紧急救援响应速度。随着城市规模扩张与交通流量剧增，传统的固定配时与感应控制方案已难以应对路口频繁出现的复杂交通事件——救护车接近路口、校车 / 公交车请求优先通行、前方车祸导致单向堵死、散落货物占用车道、施工临时改变车道归属、学校放学时段大量行人涌出等长尾场景。此类事件对决策的影响是决定性的：一次错误的相位选择可能延误紧急车辆数十秒，甚至造成生命财产损失。因此，TSC 的真实难点不在于稳态车流的统计优化，而在于**对动态交通事件的语义理解与即时响应**。

近年来，基于强化学习 (Reinforcement Learning, RL) 的方法 [Wei et al., 2019; Chen et al., 2020] 通过端到端学习路口状态到信号相位的映射，在标准场景下取得显著性能提升。然而，主流 RL 方案依赖卷积神经网络 (CNN) 或目标检测器从原始视觉输入中提取结构化状态向量（如各车道车辆数、平均速度），再交由策略网络输出动作。这种"感知-决策"流水线存在根本性局限：CNN 所输出的状态表征**缺乏语义理解能力**——它无法识别其中一辆是救护车，更无法激活"紧急车辆在场必须立即清空其行进方向"这类交通常识。任何基于固定标注类别训练的判别式模型，其理解边界均被刚性限定于训练分布之内，面对分布外事件时只能将其视作"未知形状"予以忽略。这种语义损失并非工程实现层面的缺陷，而是**判别式架构的根本性局限**——此类方法既无法对事件进行"理解"，更无法推理"事件对未来交通流的影响"。

近年来，已有研究开始尝试将大语言模型 (LLM) 引入 TSC 决策环节，代表性工作包括 LLMLight [Lai et al., 2024]、LA-Light [Wang et al., 2024] 与 VLMLight [Zhang et al., 2025] 等。这些方法普遍采用"感知 → 描述 → 决策"的**多阶段多模块流水线**架构：视觉信号需依次经过 VLM / 检测器（生成文字描述）、LLM 推理器（生成决策）、外部路由器（在常规与事件场景间切换执行分支）等多个异构模块的串行处理后，方可输出最终相位。以 VLMLight 为例，其完整推理流程涉及 VLM 场景摘要、LLM 控制、RL 快分支与 LLM 慢分支等至少 4 个独立子模块的协同运行；LLMLight 虽未引入外部路由，但视觉感知与决策推理仍被分离至视觉前端与 LLM 决策器两个异构环节。此类**非端到端流水线架构**存在三类显著缺陷：① **视觉语义的二次损失**：将高维视觉场景强行压缩为数句文本描述后，紧急车辆的视觉特征、车辆空间分布、队列几何形态等关键结构信息被大幅削弱，下游推理模型所接收的已是退化的观测表征，且文本描述环节本身可能引入幻觉性噪声；② **LVLM 固有的推理能力未被利用**：此类方法并未将 LVLM 作为统一推理主体，而是将其降格为前置判别器或摘要器，真正的决策推理被剥离至下游 LLM 或外部 RL 分支完成——这种设计割裂了视觉感知与语义推理，延长了推理链路，也使 LVLM 本可胜任的多步思考能力完全闲置；③ **评测场景覆盖不足**：LLMLight 等工作仅在常规车流路网上验证其 CoT 决策能力，**完全未涉及紧急车辆、异常拥堵等事件场景**；VLMLight 虽引入紧急车辆测试，却将实验**局限于单一标准 4 叉路口**，没有验证跨拓扑（T 字路口、左行特例、非对称多车道）与跨规模（大规模路网）的泛化能力。现有工作普遍缺乏对"**事件迁移 × 拓扑迁移 × 规模迁移**"三维泛化能力的系统性验证。

此外，我们的核心洞察是：**交通事件的正确响应本质上是一个"慢思考"任务——它需要先识别事件、再推理事件对未来交通流的影响、最后选择适宜的响应动作；而常规车流场景则仅需要"快思考"——基于当前压力即可直接完成决策**。现有双阶段方法将"快思考"与"慢思考"分配给两个不同的模型承担，其隐含假设是单一模型难以同时胜任这两种截然不同的推理模式。LVLM 的出现改变了这一前提：在互联网规模图文语料上预训练的 LVLM 已将"视觉感知 → 语义理解 → 操作知识"内化为联合表征，使其既能在常规场景直接生成短推理链完成快速决策，也能在事件场景展开多步推理链进行深入分析。关键在于，这两种推理模式本就内嵌于同一模型的联合表征之中——**通过适当的任务对齐即可将其显式激活，在单一 LVLM 内部完成快慢思考的自适应切换，无需引入外部路由器或双分支架构**。

基于上述洞察，我们提出 **E2ELight**——一个面向交通信号控制的**端到端** LVLM 框架：从多视角进口道图像输入到最终相位决策，整个决策流程**由一个微调 LVLM 通过单次前向推理直接完成**，系统中不存在 LVLM 之外的任何独立模型、检测器、文本摘要模块、外部路由器或下游决策器。这种极简的端到端设计直接消除了多阶段流水线架构的所有耦合点与失败源。

在此端到端骨架之上，E2ELight 的方法论核心是**单模型自适应的快慢思考 CoT**：常规车流场景下，模型触发**短路径推理** `Scene Understanding → Phase Selection`，以最小开销完成高效决策；事件场景下，模型触发**长路径推理** `Scene Understanding → Event Recognition → Impact Reasoning → Scene Analysis → Selection Logic → Phase`，依次完成**事件类型识别**（识别出当前路口属于**紧急车辆、校车与公交车、交通事故、占道、行人过街**六类典型交通事件中的何种，或不存在事件）、**事件影响推理**（该事件对未来若干仿真步内各方向交通流的影响）、以及最终的**最优相位选择**。与 VLMLight、LLMLight 等非端到端方法相比，E2ELight 的差异是结构性的：**我们没有把"快"和"慢"分配给两个不同的模型，也没有把"视觉"和"决策"分离到两个不同的模块——所有感知、理解、推理与决策行为统一发生在 LVLM 的同一次生成过程之中**，快慢思考的切换完全由 LVLM 自身基于视觉观察在 CoT 生成过程中自适应决定，既充分发挥了 LVLM 固有的多步推理能力，也从根本上规避了异构模块拼接带来的架构冗余与信息失真。

为使上述核心机制在真实部署场景中可行，E2ELight 包含两个关键支撑决策：

**(i) 多视角进口道图像输入。** E2ELight 不采用难以大规模部署的全局 BEV（其获取通常需要专用高架摄像头或多目重建设备）。我们将每个路口的视觉输入定义为**各进口道方向独立俯拍图像**的集合——即一个四进口道路口对应 4 张进口道停止线视图（N/E/S/W 各一张）——这些图像可直接取自国内"天网 / 雪亮工程"已广泛部署的杆式摄像头，无需任何新增硬件。相比全景 BEV，多视角进口道输入具有三重优势：部署门槛显著降低、单视角拍摄距离近且目标密度较低（有效缓解多目标计数幻觉）、事件对象的关键视觉特征——紧急车辆警灯与警示标识、校车 / 公交车的车身涂装与尺寸、交通事故导致的车辆异常停驻与碰撞姿态、散落货物与路面占用物的几何轮廓、行人群体的密度与过街动线、施工围挡与锥桶的几何标识——在近距离视图下均更为清晰可辨，共同为慢思考路径中的事件识别子步骤提供了可靠的视觉证据。

**(ii) 仿真器反馈的闭环对齐。** 我们的对齐流水线采用 **SFT + 在线 RLVR** 两阶段设计：首先通过 SFT 建立基础任务对齐，使模型掌握快慢思考双 CoT 模板的正确格式与基本决策逻辑；进而引入基于仿真器**双路可验证奖励**的在线 RLVR——以 SUMO e2 检测器的真实排队车辆数作为**感知奖励**缓解计数幻觉，以 rollout 若干仿真步后的 ATT/AQL 变化量作为**效率奖励**直接量化决策质量——将物理交通效率作为终极监督信号直接作用于 LVLM 训练过程，实现感知准确性与决策质量在统一梯度下的同步约束。

**本文的主要贡献总结如下：**

- 我们提出了**首个真正端到端的 TSC LVLM 框架 E2ELight**——从 4 张进口道停止线视图到信号相位决策全程由单一微调 LVLM 的**单次前向推理**完成，无需任何前置检测器、中间文本描述、外部路由器或下游决策模块。在此端到端骨架之上，E2ELight 通过**单模型自适应快慢思考 CoT** 实现对常规与事件场景的统一应对：常规场景触发短路径快速决策，事件场景展开 `Event Recognition → Impact Reasoning → Selection Logic` 的长路径慢思考。相较 LLMLight、VLMLight 等多阶段多模块流水线方法，E2ELight 在架构简洁性、推理链路长度、部署可维护性以及 LVLM 固有推理能力的利用率等方面均具显著优势。

- 我们提出了**面向视觉幻觉的双路可验证奖励在线 RLVR 对齐机制**。以 SUMO 仿真器作为终极裁判，对 LVLM 的每一次 CoT 输出即时给出两类硬标签奖励：感知奖励（e2 检测器排队车辆数 vs 模型预测值）与效率奖励（rollout 后的 ATT/AQL 变化量），通过 GRPO 范式完成在线强化学习微调。该机制将物理交通效率作为可验证奖励信号直接作用于 LVLM 训练过程，实现感知准确性与决策质量在统一梯度下的同步优化。这是交通领域首个基于**视觉-仿真闭环**的 RLVR 框架。

- 进行了**覆盖多种动态交通事件的系统化评测**。在 JiNan / Hangzhou 真实监控数据基础上，进一步在 SouthKorea_Songdo（非对称 6 车道大型路口）、France_Massy（T 字路口）、Hongkong_YMT（左行特例）、NewYork（196 路口大规模路网）以及**四类交通事件注入场景**（紧急车辆+校车与公交车、交通事故+占道）上验证模型的零样本泛化能力，同时覆盖**拓扑迁移、规模迁移、事件迁移**三个维度。这一评测体系直接针对 LLMLight 仅在常规路网验证、VLMLight 仅在单一 4 叉路口验证紧急车辆的短板，提供了当前 LLM-TSC 文献中最全面的泛化能力测试。实验结果表明 E2ELight 在所有场景下均显著优于固定配时、MaxPressure 以及现有 LLM 增强方法。


## 二. 项目概览
本项目实现了一个用于交通信号控制 (TSC) 的**真正端到端**视觉语言大模型 (LVLM) 框架。该框架以**单路口各进口道方向的停止线俯拍图像**（一个四进口道路口对应 4 张输入图像，N/E/S/W 各一张）为视觉输入，由单一微调 LVLM 通过**单次前向推理**直接输出相位决策，系统中不存在 LVLM 之外的任何独立检测器、文本摘要模块、外部路由器或下游决策器。方法论核心为**单模型自适应快慢思考 CoT**：常规场景触发短路径 (`Scene Understanding → Phase Selection`)，事件场景触发长路径 (`Scene Understanding → Event Recognition → Impact Reasoning → Scene Analysis → Selection Logic → Phase`)。项目包含完整 Pipeline，涵盖基于 SUMO/TransSimHub 的仿真与多视角进口道图像生成、SFT 数据构建、SFT 训练、基于仿真器双路可验证奖励的在线 RLVR 微调，以及覆盖常规、拓扑迁移、规模迁移与事件迁移的端到端评估。

相位空间选择：相位选择和相位时间的动态生成：基于停止线排队车辆状态决定下一个相位以及该相位绿灯时间（绿灯时间为一个候选合集：15、20、25、30、35、40）（候选空间大小 n*m）

## 三. 核心目录与模块
- `data/`：存放原始仿真数据、生成的多视角进口道图像以及处理后的 SFT 数据集与 RLVR rollout 缓存。
- `models/`：用于存储从 ModelScope 等下载的基础 LVLM 模型和训练后生成的 Checkpoints。
- `src/`：框架核心代码库。
  - `bev_generation/`：负责与仿真环境交互并生成单路口多视角进口道图像。
  - `dataset/`：负责 SFT 训练数据集的构建与处理。
  - `inference/`：LVLM 模型推理与 Prompt 构建逻辑。
  - `training/`：SFT 与基于 GRPO 的在线 RLVR 训练核心模块。
  - `evaluation/`：端到端指标评估 (ATT, AQL, AWT 等)。
- `configs/`：基于 YAML 的统一配置目录，管理仿真环境 (`env_config.yaml`)、模型与 Prompt (`model_config.yaml` / `prompt_builder.py`) 以及训练参数 (`train_config.yaml`)。
- `scripts/`：大量实用的辅助脚本，包括模型下载、仿真运行、评测任务投递 (`run_eval_gpu*.sh`) 等。
- `TransSimHub/`：底层 3D 交通仿真环境基座（基于开源仓库做的二次深度开发集成）。

## 四. 数据集与场景介绍

本研究训练数据基于 4×4 的随机合成路网数据（syn-train）用于 RL 训练，用于评测的数据集涵盖了全球多个城市不同类型的交通路口，包括真实交通流量数据和针对特定场景（如大规模路网、紧急车辆、特殊路口拓扑）的验证数据。

**1 随机合成路网数据**

To simulate the multiintersection and multi-step dynamics of real-world traffic flow, we constructed a 4 × 4 simulated road network with 300-meter roads between each intersection. The 16 positions in the network represent most typical road scenarios encountered at real-world intersections.

**2 测试数据集（真实交通流量数据）**

| 数据集/场景名称 | 简介与特点 | 交叉口数量 | 相位描述 (Action Space for Prompt) |
| :--- | :--- | :--- | :--- |
| **Jinan (China)** | 真实采集数据。包含3个时段采集的数据。基于真实采集数据，模拟了高峰流量数据和全天候数据（包含早晚高峰）。 | 12 | 4相位：NTST (南北直行), NLSL (南北左转), ETWT (东西直行), ELWL (东西左转) |
| **Hangzhou (China)** | 真实采集数据。包含3个时段采集的数据。基于真实采集数据，模拟了高峰流量数据。 | 16 | 同 Jinan 数据 (4相位) |
| **Songdo (South Korea)** | 新开发城区，大型交叉口。每个方向多达5车道，交通流量大。拓扑迁移验证。 | 1 | 4相位：东西/南北 直行/左转，右转无限制 |
| **Yau Ma Tei (Hong Kong)** | 稠密市中心，道路狭窄。只有南进口有右转，东进口有左转。拓扑迁移验证（左行特例）。 | 1 | 3相位：东西/南北直行，南进口右转（东进口左转一直放行） |
| **Massy (France)** | 郊区 T字路口 (T-junction)。车道配置特殊，流量较小。拓扑迁移验证。 | 1 | 2相位：南北进口道放行，西进口道放行（右转无限制） |
| **New York (USA)** | 曼哈顿上东区，基于出租车轨迹数据。超大规模复杂路网。规模迁移验证。 | 196 | 同 Jinan 数据 (4相位) |

相关数据集流量分布见 `results/traffic_flow_stats.csv`。

**3 事件场景验证数据集**

每类场景生成 2 个混合事件路由文件（减少独立评测次数）：

| 事件路由文件 | 包含事件类型 | 核心评测指标 |
| :--- | :--- | :--- |
| `*_emergy_bus.xml` | 紧急车辆（消防车/救护车/警车）+ 校车与公交车 | EATT、EAWT（紧急车辆）；BATT、BAWT（公交车） |
| `*_accident_debris.xml` | 车辆碰撞/异常停驻 + 散落货物/施工占道 | MaxQL（峰值排队长度）；TPT（普通车辆通行数） |

事件路由文件基于 Jinan×2 + Hangzhou×2 + SouthKorea_Songdo×1 + France_Massy×1 + Hongkong_YMT×1 共 7 个场景生成，通过 `scripts/event_scene_generation/` 下批量脚本一键生成。

## 五. 工程实现参考

**⚠️ 注意：执行所有命令前，请务必确保已激活名为 `VLMTraffic` 的虚拟环境！代码风格严格遵循 PEP-8，添加详细的中文注释。模型训练和场景验证部分在远程服务器上进行，涉及训练和场景验证的模块只需提供可运行代码，不需要本地执行。**

### 5.1 动作空间：联合（相位 × 绿灯时长）

动作格式为 `{'phase_id': int, 'duration': int}`。绿灯时长常量定义在 `src/utils/tsc_env/tsc_wrapper.py`：

```python
GREEN_DURATION_CANDIDATES = [15, 20, 25, 30, 35, 40]  # VLM 可选绿灯时长（秒）
FIXED_TIME_GREEN_DURATION = 27  # FixedTime / MaxPressure 专用：27s + 3s 黄灯 = 30s 整步
```

VLM 输出校验：解析 `Action: phase=X, duration=Y` → 吸附到最近合法候选值 → 传入环境。`TSCEnvWrapper._decode_action` 同时兼容字典、元组、int 三种格式。

### 5.2 视觉输入：4张多视角图像

每次决策提供 4 张图像（4 张进口道停止线视图），方向顺序固定为 N/E/S/W（顺时针，N=0）。传感器命名：`{jid}_{dir}`（如 `J1_N`）。方向由 SUMO 车头朝向推算：`approach_bearing = (heading + 180°) % 360°`，映射到最近象限。图像由 `Evaluator._collect_4_images()` 采集，缺失方向返回 None。

图像输出路径：`data/eval/{dataset}/{route_file_name}/{method}/{jid}/{sumo_step}/`，包含 `{jid}_N.png`、`{jid}_E.png`、`{jid}_S.png`、`{jid}_W.png` 四张图像及 `response.txt`。

### 5.3 Prompt 与 CoT 输出格式

CoT 模板：常规场景走 `Scene Understanding → Phase Selection` 短路径；事件场景走包含 `Event Recognition → Impact Reasoning → Scene Analysis → Selection Logic` 的长路径。输出末行固定为：

```
Action: phase=<phase_id>, duration=<seconds>
```

`configs/prompt_builder.py` 负责 4 图描述、Duration Selection CoT 块、`coordination_context` 协同章节动态注入。`src/inference/vlm_agent.py` 的 `_parse_action()` 返回 `(phase_id, duration)` 元组，duration 不在候选集时自动吸附。

### 5.4 异步决策与评测主循环

`src/utils/tsc_env/tsc_wrapper.py` 的 `state_wrapper` 采用 OR 逻辑（任一路口就绪即返回）。`src/evaluation/run_eval.py` 主循环仅对 `can_perform_action=True` 的路口渲染图像并执行 VLM 推理，其余路口维持上一次动作。

终止条件为 `sumo_t >= max_sumo_seconds`（CLI 参数 `--max_sumo_seconds`，单位秒，从路由文件动态计算：`max_depart + 300s`，上下限 [300, 6000]）。

评测模式：`--fixed_time`、`--max_pressure`、VLM 三种。批量评测脚本：`run_batch_eval.sh`（常规）、`run_batch_generalization.sh`（泛化）、`scripts/run_batch_event_eval.sh`（事件，支持 `--event {type}` 单类过滤）。

### 5.5 指标体系

`src/evaluation/metrics.py` 的 `calculate_from_files()` 支持 `event_id_prefixes`、`MaxQL`（峰值排队长度）、`TPT`（普通车辆通行数）、`special_vtypes` 参数。各场景 5 项核心指标：

- 紧急车辆：ATT / AWT / AQL / **EATT** / **EAWT**（按 vType=emergency/police/fire_engine 过滤）
- 校车与公交车：ATT / AWT / AQL / **BATT** / **BAWT**（按 vType=bus/school_bus 过滤）
- 交通事故 / 占道：ATT / AWT / AQL / **MaxQL** / **TPT**（过滤对应事件车辆）

`src/evaluation/collect_metrics.py` 提供统一收集入口：`--type {generalization|emergency|bus|accident|debris|all}`。结果模板位于 `results/` 目录下。

### 5.6 上下游协同机制：EventBulletin

`configs/event_bulletin.py` 实现路口间异步事件广播（纯 Prompt 层，不修改仿真底层）。当 VLM CoT 输出 `Final Condition: Special` 时触发广播，TTL = `green_duration + 3s`（SUMO 秒）。拓扑来源于 `configs/scenairo_config.py` 的 `TOPOLOGY` 静态配置（规则路网由 `_generate_grid_topology` 自动生成）。`configs/prompt_builder.py` 在 `coordination_context` 非空时于 Prompt 末节插入 `Upstream Coordination Context [ACTIVE]`。

### 5.7 3D 渲染双通路

TransSimHub 存在两条独立渲染通路：**通路一（动态车辆）** 由 `scene_sync` → `Vehicle3DElement` 处理真实运动车辆（紧急车辆、Bus/School Bus）；**通路二（静态障碍物）** 由 `EventManager` → `EmergencyManager3D` 处理 trip+stop 占位假车（路障 barrier_A~E、碰撞残骸 crash_vehicle）及无 SUMO 实体的 ClosureZone。`scene_sync._is_event_vehicle()` 过滤器防止假车产生幽灵背景车辆，精确匹配 `crash_vehicle`、`barrier_A~E`、`tree_branch_*`，并兼容带长度后缀的变体（如 `barrier_A_5.00`）。

---

## 六. 实验设计

### 6.1 主实验：常规场景性能对比

**目标**：在真实交通流量场景下，与经典 RL 方法和最新 LLM-TSC 方法进行全面对比，验证 E2ELight 在常规场景下的控制效率。

**数据集**：Jinan（3个时段）× Hangzhou（2个时段），共 5 个评测场景，每个场景对应 1 条路由文件。

**对比方法（13种）**：
- 规则基线：FixedTime、MaxPressure
- 经典 RL 方法：IntelliLight、FRAP、PressLight、MetaLight、MPLight、DynamicLight
- 大规模协同 RL：UniTSA
- LLM/VLM 增强方法：LLMLight、VLMLight（兼做端到端架构消融基线）

**核心指标**：ATT（平均行程时间）、AWT（平均等待时间）、AQL（平均排队长度）

**实验规模**：5 场景 × 13 方法 = 65 次评测

---

### 6.2 泛化性实验

#### 6.2.1 拓扑迁移验证

**目标**：验证 E2ELight 在 Jinan/Hangzhou 之外的异构拓扑路口的零样本适应能力。

**数据集**：SouthKorea_Songdo（非对称 5/6 车道）、France_Massy（T 字路口）、Hongkong_YMT（左行特例）

**对比方法**：FixedTime、MaxPressure、LLMLight、VLMLight、E2ELight（5种）

**指标**：ATT、AWT、AQL

**实验规模**：3 场景 × 5 方法 = 15 次评测

#### 6.2.2 规模迁移验证

**目标**：验证 E2ELight 在 196 路口超大规模路网下的协同控制能力。

**数据集**：New York（196路口，全量）

**对比方法**：FixedTime、MaxPressure、LLMLight、VLMLight、E2ELight（5种）

**指标**：ATT、AWT、AQL

**实验规模**：1 场景 × 5 方法 = 5 次评测

#### 6.2.3 事件迁移验证

**目标**：验证 E2ELight 对动态交通事件的识别与响应能力，覆盖紧急车辆优先通行和占道事故两大类。

**数据集**：基于 Jinan × 3 + Hangzhou × 2 + SouthKorea_Songdo × 1 + France_Massy × 1 + Hongkong_YMT × 1 共 8 个场景，各生成 2 个混合事件路由文件：
- `*_emergy_bus.xml`：紧急车辆 + 校车/公交车
- `*_accident_debris.xml`：交通事故 + 路面占道

**对比方法**：FixedTime、MaxPressure、LLMLight、VLMLight、E2ELight（5种）

**指标**：
- `*_emergy_bus` 场景：ATT / AWT / AQL / EATT（紧急车辆平均行程时间）/ EAWT（紧急车辆平均等待时间）/ BATT / BAWT
- `*_accident_debris` 场景：ATT / AWT / AQL / MaxQL / TPT

**实验规模**：8 场景 × 2 文件 × 5 方法 = 80 次评测

---

### 6.3 消融实验

#### 6.3.1 训练阶段消融

**目标**：验证 SFT 和 RLVR 各阶段对最终性能的贡献。

**数据集**：Jinan × 3 + Hangzhou × 2（代表性子集）

**消融变体（3种）**：
| 变体 | 描述 |
| :--- | :--- |
| Zero-shot | 预训练 LVLM，无任何微调 |
| SFT only | 仅 SFT，无 RLVR |
| **E2ELight (full)** | SFT + RLVR（完整模型） |

**指标**：ATT、AWT、AQL，以及感知准确率（车辆计数误差）

**实验规模**：5 场景 × 3 变体 = 15 次评测

#### 6.3.2 动作空间消融

**目标**：验证联合动作空间（相位 + 绿灯时长）相比仅选相位的必要性。

**数据集**：Jinan × 3 + Hangzhou × 2

**消融变体（2种）**：
| 变体 | 描述 |
| :--- | :--- |
| Phase-only | 仅选择相位，绿灯时长固定为 27s |
| **Phase + Duration** | 联合选择相位与时长（候选集 [15,20,25,30,35,40]s） |

**指标**：ATT、AWT、AQL

**实验规模**：5 场景 × 2 变体 = 10 次评测

#### 6.3.3 快慢思考 CoT 自适应切换消融

**目标**：验证自适应快慢思考切换机制的有效性，以及慢思考路径各推理步骤的必要性。

**数据集**：事件场景（Jinan × 3 + Hangzhou × 2，`*_emergy_bus.xml`）

**消融变体（3种）**：
| 变体 | 描述 |
| :--- | :--- |
| Fast-only | 所有场景强制使用短路径（无 Event Recognition 等步骤） |
| Slow-only | 所有场景强制使用长路径 |
| **Adaptive (E2ELight)** | LVLM 自适应决定快慢路径（完整模型） |

**指标**：ATT、AWT、AQL、EATT、EAWT，以及事件识别准确率

**实验规模**：5 场景 × 3 变体 = 15 次评测

#### 6.3.4 路口间事件广播机制消融

**目标**：验证 EventBulletin 定向广播机制对路网级协同控制的贡献，量化上下游路口预警信息注入的有效性。

**背景**：E2ELight 实现了纯 Prompt 层的路口间异步事件广播（`src/utils/event_bulletin.py`）——当某路口 VLM CoT 输出 `Condition: Special` 时，系统根据事件类型定向路由：Emergency/Transit 广播给下游出口方向邻居（车辆将驶入），Crash/Obstruction 广播给上游进口方向邻居（拥堵将向上游溢出）；通知以 Prompt 片段形式注入邻居路口的 `Upstream Coordination Context`，TTL 绑定当前绿灯时长。该机制无需修改仿真底层，完全在语言层面实现跨路口协同。

**数据集**：事件场景（Jinan × 3 + Hangzhou × 2，`*_emergy_bus.xml` + `*_accident_debris.xml` 各一）

**消融变体（3种）**：
| 变体 | 描述 |
| :--- | :--- |
| No Bulletin | 关闭广播，所有路口独立决策，无邻居事件通知注入 |
| Broadcast All | 广播给所有邻居（无定向路由，全部邻居均收到通知） |
| **Directed Bulletin (E2ELight)** | 定向广播：Emergency/Transit → 下游邻居；Crash/Obstruction → 上游邻居 |

**指标**：ATT / AWT / AQL / EATT / EAWT（emergy_bus 场景）；ATT / AWT / AQL / MaxQL（accident_debris 场景）

**实验规模**：5 场景 × 2 事件文件 × 3 变体 = 30 次评测

---

### 6.4 补充实验

#### 6.4.1 推理效率分析

**目标**：衡量 E2ELight 的实际部署可行性，与 VLMLight（多模块流水线）进行延迟对比。

**测量指标**：
- 单次推理延迟（ms/决策）：E2ELight vs VLMLight（分模块计时）
- 显存占用（GB）
- 推理延迟与绿灯最短时长（15s）的对比，验证实时可行性

#### 6.4.2 幻觉分析

**目标**：量化 E2ELight 在车辆计数任务上的幻觉程度，以及 RLVR 感知奖励的缓解效果。

**测量指标**：
- 车辆计数 MAE（CoT 中 Lane Analysis 预测值 vs SUMO e2 检测器真值）
- 训练前后的计数误差对比（Zero-shot vs SFT vs Full RLVR）

#### 6.4.3 事件识别准确率分析

**目标**：单独评估 E2ELight 在 CoT Event Recognition 步骤的分类准确性。

**测量指标**：事件类型分类准确率（normal / emergency / bus / accident / debris），基于事件场景的所有决策步骤统计。

#### 6.4.4 Case Study

**目标**：通过典型案例展示 E2ELight 慢思考路径的推理质量。

**案例设计**：
- 紧急车辆场景：展示 Event Recognition → 优先放行决策的完整 CoT
- 交通事故场景：展示 Impact Reasoning 对阻塞方向的绕行推理
- 正常高峰场景：展示短路径快速决策的 CoT 格式

---

## 七. 相关研究文献汇总

### 1. VLLM 幻觉

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

### 2. LLM for Traffic Signal Control

| 论文名称及链接 | 期刊/时间 | 概述 | 论文的详细介绍 | 评论 / 优缺点分析 | 控制策略 (输入与动作) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VLMLight**: Safety-Critical Traffic Signal Control via Vision-Language Meta-Control and Dual-Branch Reasoning Architecture<br>🔗[Arxiv 2505](https://arxiv.org/pdf/2505.19486) | 2025<br>arXiv | 提出了融合 VLM、LLM 和 RL 的交通信号控制框架。构建了多视角图像仿真平台，VLM 提取结构化文本，LLM 作元控制器，实现"常规用 RL，紧急用 LLM"的双分支策略，显著降低紧急车辆等待时间。 | **1. 感知层**：利用 Qwen2.5-VL 将路口原始多视角图像转化为结构化文本（如拥堵度、特种车辆）。<br>**2. 元控制层**：LLM 充当"路由器"，根据文本描述判断当前是 Normal 还是 Special 场景。<br>**3. 执行层**：正常场景直接调用 PPO 模型（快分支）；紧急场景激活 Agent 多步推理（慢分支）让出特权绿灯。<br>整体思路是用 LLM/VLM 解决长尾紧急场景，用 RL 保证常规效率。 | **缺点**：<br>1. **推理延迟/功能冗余**：VLM 提取信息 + LLM 分发意图导致链路过长。<br>2. **仿真局限**：自研视觉器缺乏天气、光照变化，影响 sim-to-real 泛化。<br>3. **模型过重无微调**：使用了32B/72B超大模型且未做SFT，极易产生视觉幻觉。建议改用小参数 VLM 结合图像-文本问答对进行 SFT。<br>4. **冗余验证**：验证动作是否合法意义不大，模型可通过约束生成直接控制。 | **感知输入**：多视角 BEV 图像。<br>**控制逻辑**：<br>- 特殊场景：优先放行救护车/消防车。<br>- 常规场景：基于 PPO 历史状态选择动作。<br>- 仿真循环：首步用预训练 PPO 决策并渲染，后续步获取图像 $\rightarrow$ VLM 场景理解 $\rightarrow$ LLM 快慢分支判断 $\rightarrow$ 输出 Phase。 |
| **Traffic-R1**: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control Systems<br>🔗[Arxiv 2508](https://arxiv.org/abs/2508.02344) | 2025<br>arXiv | 提出基于强化学习训练的 3B 参数 LLM（致敬 DeepSeek-R1 范式）。通过两阶段 RL 训练，具备零样本泛化能力，支持边缘设备部署及异步多路口协调，已在真实世界落地管理10个路口。 | **1. 训练范式革命**：放弃传统的 SFT，直接采用"离线专家指导 RL" + "在线开放世界 RL"两阶段训练，赋予模型类人推理（Thinking）能力。<br>**2. 工程落地**：在极小参数量（3B）下实现了边缘设备实时推理，并通过异步通信网络解决了多路口协同的延迟问题。<br>**3. 真实部署**：是少数真正走到线下部署（日均5.5万驾驶员）并验证能降低 9.3% 排队长度的工作。 | **特点**：<br>1. LLM 的输入是纯结构化**文本描述**（提取自雷达或视觉传感器），而非原始图像数据。<br>2. 重点在于"强化学习微调 LLM"在交通领域的落地，证明了小模型+RL的巨大潜力。 | **感知输入**：场景结构化文本描述。<br>**控制逻辑**：模型内部生成长程 CoT 推理，最终输出下一步的最优红绿灯相位（Action）。支持相邻路口的异步状态共享。 |
| **LLM-assisted light**: Leveraging LLM capabilities for human-mimetic TSC in complex urban environments<br>🔗[Arxiv 2403](https://arxiv.org/abs/2403.08337) | 2024<br>arXiv | 提出 LA-Light 框架，将 LLM 作为决策中心"大脑/指挥官"。LLM 通过调用工具（Tools）感知环境，并将现有的 RL 算法作为"顾问"辅助决策，最终输出可解释的方案。 | **典型的 Agent（智能体）架构**：<br>不强求 LLM 自己算出最优解，而是让 LLM 充当调度枢纽。框架包含：<br>- **记忆模块**：记录历史交通状态。<br>- **工具调用**：调用摄像头 API 获取排队长度，调用 RL 算法获取"建议动作"。<br>- **推理决策**：LLM 综合"自己看到的"和"RL 建议的"，做最终拍板。 | **特点**：<br>架构设计非常符合 Agent 哲学，包含了需求理解、工具调用、反馈循环。但同样面临大模型 API 调用带来的高延迟问题，且对 RL "顾问"的依赖度较高。 | **感知输入**：通过调用 Tool 获取的环境状态文本。<br>**控制逻辑**：综合推理后，输出指定的 Phase 控制命令。 |
| **LLMLight**: Large Language Models as Traffic Signal Control Agents<br>🔗[KDD 2025](https://dl.acm.org/doi/abs/10.1145/3690624.3709379) | 2025<br>KDD | 基于 Qwen2-14B 的三阶段对齐训练：GPT-4 收集轨迹 $\rightarrow$ LoRA 模仿微调 $\rightarrow$ Critic 引导的排序损失 (RBC Loss) 后对齐，使小模型具备强大的零样本控制能力。 | **非常经典的对齐（Alignment）管线**：<br>**Stage 1**：利用高级模型（GPT-4）生成含有思维链的决策轨迹，并用预训练的 RL（Advanced-CoLight）作为 Critic 剔除烂数据。<br>**Stage 2**：用筛选后的优质数据对 14B 模型进行 SFT。<br>**Stage 3**：用 Critic 的 Q 值作为奖励信号，通过 RBC Loss 让模型进一步对齐交通效率最大化目标。 | **缺点/反思**：<br>1. **信息压缩过度**：输入给 LLM 的仅是各车道排队长度，丢失了路口几何拓扑、车道数等核心物理信息。<br>2. **为了 LLM 而 LLM**：训练重度依赖 Advanced-CoLight（一个RL模型）来当裁判。这说明 LLM 只是在"拟合"一个性能优异的 RL 模型，但 LLM 的加入确实弥补了纯 RL 在可解释性和跨路口泛化上的短板。 | **感知输入**：文本化排队长度。<br>**控制逻辑**：端到端输出推理过程和动作（下一个绿灯相位）。 |
| **Prompt to Transfer**: Sim-to-Real Transfer for Traffic Signal Control with Prompt Learning<br>🔗[AAAI 2024] | 2024<br>AAAI | 利用 LLM 作为世界模型知识库（World Model KB），通过 Prompt Learning 辅助传统 RL 算法适应系统动力学差异，解决 Sim-to-Real 鸿沟。 | **解决仿真到现实的痛点**：<br>RL在仿真器（如 SUMO）中训练得很好，但到了真实世界（雨雪天、刹车距离变长）就崩溃。本文不让 LLM 直接控灯，而是让 LLM 提供"常识"（例如：下雪天车速会变慢10%），用这些常识去动态修正 RL 模型的输入状态或 Reward 函数。 | **特点**：<br>典型的应用型论文，切入点很巧。针对仿真场景中**缺乏恶劣环境真实数据**的痛点，用 LLM 的内部先验知识来做"软补偿"。 | **感知输入**：环境描述（天气、路况）。<br>**控制逻辑**：LLM 输出环境动态调整参数 $\rightarrow$ 传统 RL 结合参数输出具体 Phase。 |
| **The Crossroads of LLM and Traffic Control**: A Study on LLMs in Adaptive TSC<br>🔗[IEEE TITS 25] | 2025<br>TITS | 提出通用能力智能体（GCA），结合 Zero-Shot CoT 与 Actor-Critic 机制，让 GPT-3.5 像人类调度员一样逻辑推理，并根据文本反馈自我修正，效率超越传统感应控制。 | **免微调范式（Zero-shot + Reflection）**：<br>完全没有训练过程。直接把交通状态塞给 GPT-3.5，利用大模型强大的原生推理能力配时。如果配得不好（比如下一秒某个方向排队更长了），Critic 会生成一段"批评文本"，让 GPT-3.5 在下一轮"反思"并调整策略。 | **缺点**：<br>1. **感知极度理想化**：前提是 LLM 能拿到无噪声的完美文本（如"精确的5辆车"），但在真实世界，传感器和 CV 模型不可能100%准。LLM 对感知误差的容忍度存疑。<br>2. **API 延迟**：严重限制实时毫秒级调度。 | **感知输入**：理想化的结构化交通文本。<br>**控制逻辑**：<br>不仅输出选择的 **相位 (Phase)**，还直接输出该相位的 **具体持续时长 (Duration)**。（相比只选相位的模型进了一大步） |
| **LLM-Driven Urban Traffic Signal Control**<br>🔗[ANZCC 2024] | 2024<br>ANZCC | 提出基于 ACP 方法的框架，将 LLM 定位为人类与底层算法的"翻译官"。设计了自主、反馈、人工接管三种模式，强调可解释性与安全性。 | **概念性系统架构**：<br>系统不信任 LLM 算读秒，而是让 LLM 充当"操作员接口"。人类用自然语言下达宏观指令（如"优先疏散东向拥堵"），LLM 负责把这句话翻译成底层算法的代码或参数；同时，把底层的执行结果翻译成人话汇报给操作员。 | **致命缺点**：<br>1. **空洞无物**：通篇只有流程图，没有任何仿真实验、对比基线和具体的 Case Study。<br>2. **大材小用**：LLM 沦为了纯粹的 NLP 翻译机，其最强大的因果推理和时空规划能力在控制环节完全缺席。 | **感知输入**：人类自然语言指令 / 宏观交通报告。<br>**控制逻辑**：翻译为规则/RL 算法的运行参数。 |

### 3. RLVR（Reinforcement Learning with Verifiable Rewards）

RLVR 是以"无需人工偏好标注、用确定性验证器直接给出奖励"为核心的 RL 范式。以下文献从基础算法 → VLM 视觉推理扩展 → 交通仿真应用三个层次覆盖本项目所需背景。

| 论文简称 / 核心亮点 | 论文完整名称 | arXiv 链接 | GitHub 开源代码 |
| :--- | :--- | :--- | :--- |
| **DeepSeek-R1**<br>*(RLVR+GRPO 范式奠基作)* | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning (2025) | [2501.12948](https://arxiv.org/abs/2501.12948) | [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) |
| **R1-V**<br>*(VLM 视觉计数 RLVR，$3 成本触发计数"aha moment")* | R1-V: Reinforcing Super Generalization Ability in Vision Language Models with Less Than $3 | [GitHub Only](https://github.com/Deep-Agent/R1-V) | [Deep-Agent/R1-V](https://github.com/Deep-Agent/R1-V) |
| **VLM-R1**<br>*(稳定可泛化的 R1 风格 VLM)* | VLM-R1: A Stable and Generalizable R1-style Large Vision-Language Model | [2504.07615](https://arxiv.org/abs/2504.07615) | [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) |
| **CrowdVLM-R1**<br>*(模糊 GRPO 奖励解决人群计数)* | CrowdVLM-R1: Expanding R1 Ability to VLM for Crowd Counting using Fuzzy Group Relative Policy Reward | [2504.03724](https://arxiv.org/abs/2504.03724) | — |
| **PEARL**<br>*(感知证据锚定 RL，防止视觉幻觉与奖励 Hacking)* | Perceptual-Evidence Anchored Reinforced Learning for Multimodal Reasoning | [2511.18437](https://arxiv.org/abs/2511.18437) | — |
| **KAWHI**<br>*(视觉-几何信息注入奖励重加权，兼容 GRPO)* | Bridging Visual Representation and Reinforcement Learning from Verifiable Rewards in Large Vision-Language Models | [2603.27375](https://arxiv.org/abs/2603.27375) | — |
| **SynthRL**<br>*(可验证数据合成扩展 RLVR 训练集)* | SynthRL: Scaling Visual Reasoning with Verifiable Data Synthesis | [2506.02096](https://arxiv.org/abs/2506.02096) | — |
| **R1Sim**<br>*(R1 风格 GRPO + 安全奖励用于交通轨迹仿真)* | Learning Rollout from Sampling: An R1-Style Tokenized Traffic Simulation Model | [2603.24989](https://arxiv.org/abs/2603.24989) | — |
| **LaViPlan**<br>*(RLVR 用于语言引导视觉路径规划，ICCV 2025W)* | LaViPlan: Language-Guided Visual Path Planning with RLVR | [ICCV2025W](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/papers/Oh_LaViPlan__Language-Guided_Visual_Path_Planning_with_RLVR_ICCVW_2025_paper.pdf) | — |
| **Traffic-R1**<br>*(见 §2，RLVR 文本 TSC 落地，含真实部署验证)* | Traffic-R1: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control | [2508.02344](https://arxiv.org/abs/2508.02344) | — |
| **CoLLMLight**: Cooperative LLM Agents for Network-Wide TSC<br>🔗[ICLR 2026在投] | 2026<br>ICLR | 将控制范围扩展至**全路网**。引入结构化时空图谱和**复杂度感知推理**（动态调整推理深度），并通过**自定义物理价值函数**构建数据集，进行两阶段 SFT+自监督微调。 | **路网级协同与微调创新**：<br>**1. 动态算力分配**：基于路口繁忙度，决定使用【无协调、简单协调、复杂协调】。复杂时会预测邻近路口的未来状态。<br>**2. 自定义物理价值标签**：没有用 RL 跑数据，而是直接用公式 `V(动作) = 排队时间成本 + 行驶时间成本` 穷举算出一个最大值。把这个物理最优动作作为标签，让模型进行 SFT 和自监督纠错学习。 | **评论/优缺点**：<br>1. **延迟危机**：多路口时空提示词极其复杂，长文本推理必然导致严重延迟。唯一的解法是：保证"推理时间 < 绿灯相位持续时间"，用滞后一个时间步的状态提前做决策。<br>2. 核心亮点在于**放弃了 RL 框架，直接用物理公式造数据集**，降低了工程门槛。 | **感知输入**：当前路口 + 上下游车道状态（占用率、排队时间）。<br>**控制逻辑**：<br>基于动态推理深度，评估未来所有可行 Phase 的综合成本，选择最优 Phase。 |
