# NCDPO 技术方案：数值距离连续惩罚 DPO

> **方向 3 实现文档** | 基于 RPO（TRL 0.29.1）的逐样本动态 β 偏好对齐

---

## 一、问题动机

### 1.1 标准 DPO 的固定惩罚缺陷

标准 DPO（Direct Preference Optimization）损失为：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta \cdot \Delta_i\right)\right]$$

$$\Delta_i = \left(\log\frac{\pi_\theta(y_w^i|x)}{\pi_{\text{ref}}(y_w^i|x)} - \log\frac{\pi_\theta(y_l^i|x)}{\pi_{\text{ref}}(y_l^i|x)}\right)$$

其中 $\beta$ 是**全局固定标量**，对所有负样本施加相同强度的惩罚。

**这在交通信号控制场景中存在根本性问题：**

| 样本 | 实际车辆数 | 模型预测 | 计数误差 ε | DPO 惩罚 |
|:---:|:---:|:---:|:---:|:---:|
| A | 5 辆 | 4 辆 | ε = 1 | β × Δ |
| B | 5 辆 | 15 辆 | ε = 10 | β × Δ |

样本 A 和 B 在标准 DPO 眼中是**等价的负样本**，惩罚力度完全相同。然而，预测 15 辆（实际 5 辆）的幻觉严重程度是预测 4 辆的 10 倍，应当承受更强的惩罚。

### 1.2 现有数据分析印证了问题的严重性

从当前清洗后的 DPO 数据集（3555 条）分析结果：

- **平均计数 MAE**：每样本每车道相差 **0.341 辆**
- **拥堵判断分歧率**：50.72% 的样本正负侧拥堵评级不一致
- **决策规则差异率**：26.72% 的样本使用了不同的调度规则

这说明负样本质量参差不齐，部分负样本的计数幻觉极为严重，标准 DPO 对其惩罚力度严重不足。

---

## 二、NCDPO 核心算法

### 2.1 动态 β 设计

**Numerically Calibrated DPO（NCDPO）** 将 β 从全局常量升级为**逐样本动态变量**：

$$\beta_i = \beta_{\text{base}} \times \left(1 + \alpha \cdot \tanh\!\left(\frac{\varepsilon_i}{\varepsilon_{\text{scale}}}\right)\right)$$

| 参数 | 含义 | 默认值 |
|:---|:---|:---:|
| $\beta_{\text{base}}$ | 基准 KL 惩罚系数 | 0.1 |
| $\alpha$ | 动态范围幅值系数 | 1.0 |
| $\varepsilon_i$ | 第 i 条样本的计数 MAE | 由数据集预计算 |
| $\varepsilon_{\text{scale}}$ | tanh 归一化尺度 | 2.0 |

**tanh 的三大优良特性**：
1. **有界性**：$\tanh(\cdot) \in (0, 1)$，β 最大为 $\beta_{\text{base}} \times (1 + \alpha)$，防止梯度爆炸
2. **单调性**：计数误差越大，惩罚越强，符合直觉
3. **平滑性**：连续可微，梯度稳定，不引入训练震荡

**β_i 的实际动态范围**（α=1.0, ε_scale=2.0）：

| ε_i（车道计数 MAE） | w = tanh(ε/2) | β_i | 相对增强 |
|:---:|:---:|:---:|:---:|
| 0.00（完全一致） | 0.000 | 0.1000 | ±0% |
| 0.34（数据集均值） | 0.167 | 0.1167 | +16.7% |
| 0.50 | 0.245 | 0.1245 | +24.5% |
| 1.00 | 0.462 | 0.1462 | +46.2% |
| 2.00 | 0.762 | 0.1762 | +76.2% |
| 3.00 | 0.905 | 0.1905 | +90.5% |
| ∞（极端幻觉） | → 1.000 | → 0.2000 | +100%（上界） |

### 2.2 完整损失函数

NCDPO 在 RPO 框架下扩展，完整损失为：

$$\mathcal{L}_{\text{NCDPO}} = \underbrace{-\mathbb{E}\left[\log\sigma\left(\beta_i \cdot \Delta_i\right)\right]}_{\text{动态惩罚 DPO 损失}} + \underbrace{\lambda \cdot \mathcal{L}_{\text{SFT}}(y_w)}_{\text{SFT 正则项（同 RPO）}}$$

- **第一项**：逐样本动态 β 的 DPO 对比损失，惩罚计数幻觉严重的负样本
- **第二项**：在 chosen 回复上的 NLL 损失，防止偏好学习导致正样本质量退化（继承自 RPO）

当 $\varepsilon_i = 0$（全部样本计数误差为零）时，$\beta_i = \beta_{\text{base}}$，NCDPO 退化为标准 RPO，**向下兼容性完美**。

---

## 三、实现架构

### 3.1 数据流全景

```
原始 DPO 数据集
(dpo_dataset_sft_model_gener_cleaned.jsonl)
         │
         ▼
┌─────────────────────────────────────┐
│  augment_ncdpo_weights.py           │
│  ① 解析 chosen/rejected 车道计数      │
│  ② 计算 MAE → count_error_weight     │
│  ③ 写入 dpo_dataset_ncdpo.jsonl      │
└─────────────────────────────────────┘
         │
         ▼
增强数据集 dpo_dataset_ncdpo.jsonl
(含 count_error_weight: float 字段)
         │
         ▼ HuggingFace .map()
format_qwen_dpo_dataset()
(加载 PIL 图像，保留 count_error_weight)
         │
         ▼ 
┌─────────────────────────────────────┐
│  NCDPODataCollator                  │
│  ① pop count_error_weight           │
│  ② 调用父类 VisionPreference         │
│     (tokenize + image encode)       │
│  ③ 将 count_error_weight 作为        │
│     FloatTensor 追加到 batch          │
└─────────────────────────────────────┘
         │  batch["count_error_weight"] = Tensor(B,)
         ▼
┌─────────────────────────────────────┐
│  NCDPOTrainer._compute_loss()       │
│  ① pop count_error_weight           │
│  ② 存入 self._ncdpo_errors           │
│  ③ 调用 super()._compute_loss()      │
│     → TRL 内部访问 self.beta          │
│     → property getter 返回 (B,) 张量 │
│     → -logsigmoid(β_i × Δ_i)        │
│  ④ 清除 _ncdpo_errors                │
└─────────────────────────────────────┘
```

### 3.2 核心设计：Python Property 劫持 beta

TRL 0.29.1 的 sigmoid 损失计算（`dpo_trainer.py:1236`）：

```python
# TRL 原始代码
per_sequence_loss = -F.logsigmoid(self.beta * delta_score)
```

`self.beta` 在 TRL 中是普通实例属性（`self.beta = args.beta`，赋值一次）。NCDPO 将其改写为 **Python 描述符（property）**：

```python
class NCDPOTrainer(DPOTrainer):
    @property
    def beta(self):
        if self._ncdpo_errors is not None:
            w = torch.tanh(self._ncdpo_errors / self._ncdpo_eps_scale)
            return self._base_beta * (1.0 + self._ncdpo_alpha * w)
        return self._base_beta   # 退化为标量，与原 TRL 完全一致

    @beta.setter
    def beta(self, value: float):
        self._base_beta = float(value)  # 父类 __init__ 赋值时触发
```

**为什么这个设计是正确的**：

| 场景 | `self._ncdpo_errors` | `self.beta` 返回类型 | 效果 |
|:---|:---:|:---:|:---|
| 正常训练 step | `Tensor(B,)` | `Tensor(B,)` | 逐样本 β，计数误差大的样本β更高 |
| eval step | `None` | `float` | 标准 DPO，与父类完全一致 |
| SFT 损失部分 | `Tensor(B,)` | `Tensor(B,)` | SFT 不使用 beta，无影响 |
| 父类 `__init__` | 任意 | 触发 setter | 存储为 `_base_beta` |

**关键属性**：TRL 的 `-F.logsigmoid(self.beta * delta_score)` 当 `self.beta` 为 `(B,)` 张量、`delta_score` 亦为 `(B,)` 张量时，Element-wise 乘法自然成立，无需任何修改。

### 3.3 最小侵入性验证

本实现对 TRL 0.29.1 的修改量：

| 修改点 | 说明 |
|:---|:---|
| `NCDPODataCollator.torch_call()` | 仅多一个 `pop` + `tensor` 操作 |
| `NCDPOTrainer._compute_loss()` | 仅多 `pop` + 设置/清除 `_ncdpo_errors` |
| `NCDPOTrainer.beta` property | 替换实例属性为 property，父类行为无改变 |
| `NCDPOTrainer.__init__()` | 增加 3 个实例变量初始化 |

**不涉及任何 TRL 源码的复制**，对未来 TRL 版本更新的抵抗力强。

---

## 四、文件说明

### 4.1 新增文件

```
VLMTraffic/
├── NCDPO_technical_plan.md                  ← 本文档
├── src/
│   ├── dataset/DPO_data_construct/
│   │   └── augment_ncdpo_weights.py         ← 数据集增强脚本（计算 count_error_weight）
│   └── training/
│       ├── trainer.py                       ← 新增 NCDPODataCollator + NCDPOTrainer + ncdpo 分支
│       └── ncdpo_trainer.sh                 ← NCDPO 训练启动脚本
```

### 4.2 数据集字段说明

增强后的 `dpo_dataset_ncdpo.jsonl` 在原 DPO 格式基础上新增：

```json
{
  "id": "JiNan_...",
  "prompt": [...],
  "chosen": [...],
  "rejected": [...],
  "count_error_weight": 0.417   ← 新增：正负样本车辆计数 MAE（跨所有车道均值）
}
```

`count_error_weight` 的含义：值越大，表示负样本的计数幻觉越严重，训练时该样本将获得更高的 β 惩罚。

---

## 五、使用流程

### Step 1：生成增强数据集（本地执行）

```bash
conda activate VLMTraffic
python src/dataset/DPO_data_construct/augment_ncdpo_weights.py
```

输出文件：`src/dataset/DPO_data_construct/dpo_dataset_ncdpo.jsonl`

脚本会打印计数误差分布统计和 β 预览，确认数值合理后执行 Step 2。

### Step 2：上传数据集到远程服务器

```bash
scp src/dataset/DPO_data_construct/dpo_dataset_ncdpo.jsonl \
    <user>@<server>:/root/autodl-tmp/dpo_dataset_ncdpo.jsonl
```

### Step 3：在远程服务器启动 NCDPO 训练

```bash
cd /root/code/VLMTraffic
conda activate VLMTraffic
bash src/training/ncdpo_trainer.sh
```

---

## 六、超参数调优指南

### 6.1 核心超参数敏感性分析

**`--ncdpo_alpha`（幅值系数 α）**

| α 值 | β 最大倍率 | 适用场景 |
|:---:|:---:|:---|
| 0.5 | 1.5× | 数据集噪声较大，计数误差标注不可靠 |
| 1.0 | 2.0× | **推荐默认值**，平衡惩罚强度 |
| 2.0 | 3.0× | 计数幻觉严重，需要强力纠正 |

**`--ncdpo_eps_scale`（归一化尺度 ε_scale）**

建议将 ε_scale 设置为数据集 MAE 均值的约 **6 倍**（使均值样本获得约 16% 的 β 增强）：

| ε_scale | 均值样本 (ε≈0.34) 的 β 增强 | 满分饱和 (ε=ε_scale×2) |
|:---:|:---:|:---:|
| 1.0 | +33% | ε ≈ 2.0 时饱和 |
| 2.0 | +17% | ε ≈ 4.0 时饱和 |
| 3.0 | +11% | ε ≈ 6.0 时饱和 |

### 6.2 推荐实验配置

| 实验组 | 目的 | `alpha` | `eps_scale` | `beta` |
|:---|:---|:---:|:---:|:---:|
| NCDPO-Base | 与 RPO 对比基线 | 1.0 | 2.0 | 0.1 |
| NCDPO-Strong | 更强幻觉惩罚 | 2.0 | 2.0 | 0.1 |
| NCDPO-Conservative | 更保守幻觉惩罚 | 0.5 | 3.0 | 0.1 |
| NCDPO-LowBeta | 更小基础 KL 约束 | 1.0 | 2.0 | 0.05 |

---

## 七、预期效果与评估

### 7.1 理论预期改进

1. **计数幻觉降低**：对"数错 10 辆"的负样本惩罚力度是"数错 1 辆"的约 5.7 倍（tanh 饱和区 vs 线性区），模型权重将更强烈地规避大幅计数错误。

2. **拥堵等级判断改善**：计数误差大的样本往往伴随拥堵等级误判（从数据分析可见 50.72% 的判断分歧），NCDPO 强化了对这类样本的惩罚，间接提升拥堵感知精度。

3. **训练稳定性保持**（继承 RPO）：SFT 正则项防止正样本质量退化，tanh 函数保证 β 有界，避免梯度爆炸。

### 7.2 评估指标

在现有消融实验框架（`scripts/run_eval_gpu*.sh`）中，与 `SFT+DPO+CoT` 和 `SFT+RPO+CoT` 基线对比：

| 指标 | 含义 | 目标 |
|:---|:---|:---|
| ATT ↓ | 平均行程时间 | NCDPO < RPO |
| AWT ↓ | 平均等待时间 | NCDPO < RPO |
| AQL ↓ | 平均队列长度 | NCDPO < RPO |
| 计数 MAE ↓ | 模型车辆计数误差（需专项评估） | 显著改善 |

### 7.3 与现有消融结果对比基准

```
Model Variant             JiNan_real ATT↓  AWT↓   AQL↓
─────────────────────────────────────────────────────
Zero-Shot (Qwen3-8b-VL)   393.88    133.67  159.50
SFT+CoT                   331.44     71.48   76.46
SFT+DPO+CoT               329.01     70.48   75.12
SFT+RPO+CoT               [待补充]
SFT+NCDPO+CoT             [待实验]   ← 目标：优于 RPO
```

---

## 八、与其他优化方向的关系

| 方向 | 名称 | 当前 NCDPO 关系 |
|:---|:---|:---|
| 方向 1 | 逻辑因果感知 C-DPO (Causal Step-DPO) | 正交，可组合 |
| 方向 2 | 仿真器反馈令牌级环境 DPO | NCDPO 提供计数层面的先验惩罚，方向 2 提供全局奖励信号，两者互补 |
| **方向 3** | **数值距离连续惩罚 DPO（NCDPO）** | **本文档** |

**与方向 1 的组合潜力**：方向 1（C-DPO）处理"感知正确但推理错误"和"感知错误但推理正确"两类样本，方向 3 处理"感知中的计数误差大小"。两者从不同粒度解决幻觉问题，理论上可叠加训练。

---

## 九、技术注意事项

### 9.1 ZeRO-3 兼容性

- `NCDPOTrainer._compute_loss()` 使用 `self.accelerator.device` 确定设备，与 ZeRO-3 参数分片兼容
- `beta` property 在 ZeRO-3 下返回的张量为 `(B,)` 而非分片张量，不涉及跨卡参数

### 9.2 梯度检查点兼容性

- `self.beta * delta_score`：两个张量的 Element-wise 乘积，梯度流不受影响
- `torch.tanh()` 的梯度存在且有界，不会影响梯度检查点的重计算稳定性

### 9.3 数据集空字段处理

- 若某条样本的正/负侧均无法解析出车道计数，`count_error_weight = 0.0`
- 此时 `tanh(0 / ε_scale) = 0`，`β_i = β_base`，退化为标准 RPO，无副作用
- 该 fallback 机制保证数据集的鲁棒性

### 9.4 与 `format_qwen_dpo_dataset` 的兼容性

HuggingFace `datasets.map()` 以"合并"模式工作：
- 函数返回的键会被更新
- 函数未返回的键（如 `count_error_weight`、`id`）会被**原样保留**

因此无需修改 `format_qwen_dpo_dataset` 即可透传 `count_error_weight`。

---

*文档版本：v1.0 | 2026-04-11*
