## 🧠 一、问题定义

UniPixel 旨在解决**像素级视觉推理任务**。其输入输出定义如下：

### 输入：

- 图像或视频 $\mathcal{X}$
- 文本提示 $\mathcal{T}$
```math
 N 个可选的视觉提示 \{\mathcal{P}_i\}_{i=1}^N，每个 \mathcal{P}_i 可以是：
```
  - **点**：(x, y, t)
  - **框**：(x_1, y_1, x_2, y_2, t)
  - **掩码**：
```math
  \mathbf{m}_{ij} \in \{0,1\}
```
### 输出：
- 文本响应
- $K$ 个接地（grounded）的时空掩码 $\{\mathcal{M}_i\}_{i=1}^K$

> **注意**：$N$ 和 $K$ 可以为零，这意味着模型同样能够处理纯视觉理解任务。

---

## 🧩 二、模型架构

UniPixel 基于 **Qwen2.5-VL** 构建，主要包含以下核心组件：

1.  **视觉编码器**（ViT）
2.  **提示编码器**（Prompt Encoder）
3.  **对象记忆库**（Object Memory Bank）
4.  **掩码解码器**（Mask Decoder）
5.  **大语言模型**（LLM）

此外，还扩展了三个特殊 Token：
- `<REF>`：表示视觉提示
- `<MEM>`：表示记忆注入
- `<SEG>`：触发掩码解码

---

## 🧮 三、算法原理详解

## 3.1 提示编码器

**目标**：将稀疏或密集的视觉提示编码为 LLM 可理解的 Token 表示。

**方法**：

#### **稀疏提示（点、框）编码**：
- 使用 **2D 傅里叶位置编码**与**可学习的类型嵌入**。
- 时间信息采用 **1D 傅里叶编码**。
- 拼接后进行 `GELU → Linear` 投影，映射到 LLM 的嵌入空间。

$$
\begin{aligned}
\mathbf{e}_{\text{pos}} &= \text{Fourier2D}(x, y) + \mathbf{e}_{\text{type}} \\
\mathbf{e}_{\text{time}} &= \text{Fourier1D}(t) \\
\mathbf{t}_{\text{prompt}} &= \text{Linear}(\text{GELU}([\mathbf{e}_{\text{pos}}; \mathbf{e}_{\text{time}}]))
\end{aligned}
$$

#### **密集提示（掩码）编码**：
- 将掩码调整尺寸并与视觉特征图对齐。
- 使用**掩码池化**（masked pooling）提取特征。
- 通过 `Linear → GELU → Linear` 投影到 LLM 嵌入空间。

$$
\begin{aligned}
\mathbf{f}_{\text{mask}} &= \text{MaskedPool}(\mathbf{V}, \mathbf{m}) \\
\mathbf{t}_{\text{mask}} &= \text{Linear}(\text{GELU}(\text{Linear}(\mathbf{f}_{\text{mask}})))
\end{aligned}
$$

## 3.2 对象记忆库

**目标**：动态存储和管理用户所指代的对象信息，实现跨帧、跨轮对话的**对象感知推理**。

**结构**：实现为一个哈希表，以对象 ID 为键，值为该对象的**时空掩码序列**。

**两种核心操作**：

1.  **记忆预填充**：
    - 当输入中出现 `<REF>` Token 时触发。
    - 模型预测对应对象的掩码，并将其存储至记忆库。

2.  **记忆注入**：
    - 将记忆库中对象的特征通过 `<MEM>` Token 注入到当前输入。
    - 使用掩码池化提取对象特征，投影后替换 `<MEM>` Token。

$$
\begin{aligned}
\mathbf{f}_{\text{obj}} &= \text{MaskedPool}(\mathbf{V}, \mathcal{M}_{\text{obj}}) \\
\mathbf{t}_{\text{mem}} &= \text{M→L Projector}(\mathbf{f}_{\text{obj}})
\end{aligned}
$$

## 3.3 掩码解码器

采用 **SAM 2.1** 作为掩码解码器，实现从语言表示到像素掩码的转换。

**流程**：
1.  提取 `<SEG>` Token 的最后一层隐藏状态 $\mathbf{h}_{\text{seg}}$。
2.  使用 L→M 投影器（结构与 M→L 对称）进行降维，并重塑为两个 Token。
3.  输入 SAM 2.1 解码器，预测第一帧的掩码。
4.  通过**传播机制**将掩码传播到其他帧。

$$
\begin{aligned}
\mathbf{h}_{\text{seg}}' &= \text{L→M Projector}(\mathbf{h}_{\text{seg}}) \\
\mathcal{M} &= \text{SAM2}(\mathbf{h}_{\text{seg}}', \mathbf{V})
\end{aligned}
$$

---

## 🧪 四、训练策略与损失函数

## 4.1 总损失函数

总损失 $\mathcal{L}$ 是多个损失项的加权和：

$$
\mathcal{L} = \mathcal{L}_{\text{lm}} + 100 \cdot \mathcal{L}_{\text{focal}} + 5 \cdot \mathcal{L}_{\text{dice}} + 5 \cdot \mathcal{L}_{\text{iou}} + 5 \cdot \mathcal{L}_{\text{obj}}
$$

- $\mathcal{L}_{\text{lm}}$：语言建模损失（交叉熵）
- $\mathcal{L}_{\text{focal}}$：用于掩码预测的 Focal Loss
- $\mathcal{L}_{\text{dice}}$：用于掩码预测的 Dice Loss
- $\mathcal{L}_{\text{iou}}$：IoU 预测的 MAE 损失
- $\mathcal{L}_{\text{obj}}$：对象性预测的交叉熵损失

## 4.2 三阶段训练策略

1.  **阶段一**：预训练提示编码器（使用 851K 区域标注数据）。
2.  **阶段二**：对齐 LLM 与掩码解码器（使用 87K 指代分割数据）。
3.  **阶段三**：联合多任务训练（使用约 2M 样本，采用 LoRA 微调视觉编码器和 LLM）。

---

## 📊 五、实验与评估

## 5.1 主要任务评估

UniPixel 在以下基准任务上进行了全面评估：
- **推理视频对象分割**（ReVOS）
- **指代视频对象分割**（RVOS）
- **运动基础视频推理**（GroundMoRe）
- **指代表达分割**（RES）
- **推理分割**（ReasonSeg）
- **指代表达理解**（REC）
- **指代视频描述与问答**
- **PixelQA**（本文提出的新任务，联合指代、分割与问答）

## 5.2 主要结论

- UniPixel 在 10 个基准测试中达到了 **SOTA** 性能。
- 3B 参数量模型在 ReVOS 任务上取得了 **62.1 $\mathcal{J}\mathcal{F}$** 的分数，超越多个 7B~13B 模型。
- 实验展示了模型中**指代与分割任务间的相互促进效应**。

---

## 🧾 六、总结与贡献

## 主要贡献：
1.  **提出 UniPixel**：首个能够统一处理图像和视频中对象指代与分割的端到端大型多模态模型。
2.  **引入对象记忆库**：实现了跨帧、跨轮对话的对象感知推理能力。
3.  **提出 PixelQA 新任务**：用于验证模型在像素级别进行复杂推理的能力。
4.  **全面的性能验证**：在多项任务上达到 SOTA，并展示了任务统一带来的性能增益。

---

# 数学公式解析

## 🧮 一、损失函数详细推导

### 1.1 总损失函数构成

论文中的总损失函数为：
```math
$$ \mathcal{L} = \mathcal{L}_{\text{lm}} + 100 \cdot \mathcal{L}_{\text{focal}} + 5 \cdot \mathcal{L}_{\text{dice}} + 5 \cdot \mathcal{L}_{\text{iou}} + 5 \cdot \mathcal{L}_{\text{obj}} $$
```
### 1.2 语言建模损失 (Language Modeling Loss)

对于自回归语言模型，语言建模损失是标准的下一个 Token 预测的交叉熵损失：

$$ \mathcal{L}_{\text{lm}} = -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_{1:t-1}, \mathcal{X}, \{\mathcal{P}_i\}) $$

其中：
- $T$ 是序列长度。
- $w_t$ 是第 $t$ 个 Token。
- $w_{1:t-1}$ 是前 $t-1$ 个 Token。
- $\mathcal{X}$ 是视觉输入。
- $\{\mathcal{P}_i\}$ 是视觉提示。

### 1.3 Focal Loss 用于掩码预测

Focal Loss 是针对类别不平衡问题的改进交叉熵损失：

$$ \mathcal{L}_{\text{focal}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \alpha (1-p_i)^\gamma y_i \log(p_i) + (1-\alpha) p_i^\gamma (1-y_i) \log(1-p_i) \right] $$

其中：
- $N$ 是像素总数。
- $y_i \in \{0,1\}$ 是像素 $i$ 的真实标签。
- $p_i = \sigma(z_i)$ 是模型预测的概率，$\sigma$ 是 Sigmoid 函数。
- $\alpha$ 是平衡正负样本的权重（通常设为 0.25）。
- $\gamma$ 是聚焦参数（通常设为 2），用于降低易分类样本的权重。

### 1.4 Dice Loss

Dice Loss 基于 Dice 系数，用于衡量两个集合的相似度：

$$ \mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum_{i=1}^{N} y_i p_i + \epsilon}{\sum_{i=1}^{N} y_i + \sum_{i=1}^{N} p_i + \epsilon} $$

其中 $\epsilon$ 是平滑项，用于防止除零错误（通常为 $10^{-6}$）。

Dice Loss 的梯度推导：

$$ \frac{\partial \mathcal{L}_{\text{dice}}}{\partial p_i} = -\frac{2y_i(\sum y + \sum p) - 2(\sum yp)(1)}{(\sum y + \sum p)^2} $$

### 1.5 IoU 预测的 MAE 损失

模型预测每个掩码的 IoU 分数 $\hat{i}$，与真实 IoU $i$ 计算平均绝对误差：

$$ \mathcal{L}_{\text{iou}} = \frac{1}{K} \sum_{k=1}^{K} | \hat{i}_k - i_k | $$

其中真实 IoU 计算为：

$$ i_k = \frac{\text{Area}(\mathcal{M}_k \cap \mathcal{M}_k^{\text{gt}})}{\text{Area}(\mathcal{M}_k \cup \mathcal{M}_k^{\text{gt}})} $$

### 1.6 对象性预测的交叉熵损失

这是一个二分类问题，判断每个候选是否包含有效对象：

$$ \mathcal{L}_{\text{obj}} = -\frac{1}{K} \sum_{k=1}^{K} \left[ o_k \log(\hat{o}_k) + (1-o_k) \log(1-\hat{o}_k) \right] $$

其中 $o_k \in \{0,1\}$ 表示第 $k$ 个候选是否包含对象。

---

## 🧬 二、编码原理详细推导

### 2.1 Fourier 位置编码

#### 2.1.1 理论基础

Fourier 特征映射允许神经网络学习高频函数。给定标量输入 $v$（坐标或时间），映射为：

$$ \gamma(v) = [\cos(2\pi \mathbf{B}v), \sin(2\pi \mathbf{B}v)]^T $$

其中 $\mathbf{B} \in \mathbb{R}^{m \times d}$ 是随机高斯矩阵，$m$ 是特征维度。

#### 2.1.2 空间位置编码

对于 2D 坐标 $(x,y)$，分别编码后拼接：

$$ \mathbf{e}_x = [\cos(2\pi \mathbf{B}_x x), \sin(2\pi \mathbf{B}_x x)] $$

$$ \mathbf{e}_y = [\cos(2\pi \mathbf{B}_y y), \sin(2\pi \mathbf{B}_y y)] $$

$$ \mathbf{e}_{\text{pos}} = \mathbf{e}_x + \mathbf{e}_y + \mathbf{e}_{\text{type}} $$

其中 $\mathbf{e}_{\text{type}}$ 是可学习的类型嵌入，用于区分：
- 单点
- 框的左上角
- 框的右下角

#### 2.1.3 时间编码

对于帧索引 $t$：

$$ \mathbf{e}_t = [\cos(2\pi \mathbf{B}_t t), \sin(2\pi \mathbf{B}_t t)] $$

#### 2.1.4 最终投影

拼接空间和时间编码后投影：

$$ \mathbf{t}_{\text{prompt}} = \text{Linear}(\text{GELU}([\mathbf{e}_{\text{pos}}; \mathbf{e}_t])) $$

其中 GELU 激活函数：

$$ \text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right] $$

### 2.2 掩码池化 (Masked Pooling)

给定视觉特征 $\mathbf{V} \in \mathbb{R}^{H \times W \times C}$ 和二进制掩码 $\mathbf{m} \in \{0,1\}^{H \times W}$：

$$ \mathbf{f}_{\text{mask}} = \frac{\sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{V}_{i,j} \cdot \mathbf{m}_{i,j}}{\sum_{i=1}^{H} \sum_{j=1}^{W} \mathbf{m}_{i,j} + \epsilon} $$

这实际上是对掩码区域的特征进行平均池化。

### 2.3 M→L 投影器

架构：`Linear → GELU → Linear`

数学形式：

$$ \mathbf{t}_{\text{mask}} = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{f}_{\text{mask}} + \mathbf{b}_1) + \mathbf{b}_2 $$

其中：
```math
\mathbf{W}_1 \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{visual}}}
```
```math
\mathbf{W}_2 \in \mathbb{R}^{d_{\text{llm}} \times d_{\text{hidden}}}
```
---

## 🔄 三、对象记忆库的数学形式化

### 3.1 记忆库定义

对象记忆库是一个哈希映射：

$$ \mathcal{M} = \{(id_1, \{\mathcal{M}_{1,t}\}_{t=1}^T), (id_2, \{\mathcal{M}_{2,t}\}_{t=1}^T), \ldots\} $$

其中每个对象有跨帧的掩码序列。

### 3.2 记忆预填充

当检测到 `<REF>` Token 时，模型预测掩码：

$$ \mathcal{M}_{\text{new}} = \text{MaskDecoder}(\mathbf{h}_{\text{ref}}, \mathbf{V}) $$

然后更新记忆库：

$$ \mathcal{M} \leftarrow \mathcal{M} \cup \{(id_{\text{new}}, \mathcal{M}_{\text{new}})\} $$

### 3.3 记忆注入

对于记忆库中的每个对象，提取特征并注入：

$$ \mathbf{f}_{\text{obj}}^{(k)} = \text{MaskedPool}(\mathbf{V}, \mathcal{M}_{k,t}) $$

$$ \mathbf{t}_{\text{mem}}^{(k)} = \text{M→L}(\mathbf{f}_{\text{obj}}^{(k)}) $$

输入中的 `<MEM>_k` 被替换为 $\mathbf{t}_{\text{mem}}^{(k)}$。

---

## 🎯 四、训练策略的数学描述

### 4.1 三阶段训练

**阶段一**（预训练提示编码器）：

$$ \mathcal{L}_1 = \mathcal{L}_{\text{lm}} + \lambda_{\text{reg}} \|\Theta_{\text{prompt}}\|^2 $$

**阶段二**（对齐 LLM 与掩码解码器）：

$$ \mathcal{L}_2 = \mathcal{L}_{\text{lm}} + \beta \mathcal{L}_{\text{mask}} $$

其中 $\mathcal{L}_{\text{mask}} = \mathcal{L}_{\text{focal}} + \mathcal{L}_{\text{dice}}$

**阶段三**（联合训练）：
```math
$$ \mathcal{L}_3 = \mathcal{L}_{\text{lm}} + 100\mathcal{L}_{\text{focal}} + 5\mathcal{L}_{\text{dice}} + 5\mathcal{L}_{\text{iou}} + 5\mathcal{L}_{\text{obj}} $$
```
### 4.2 LoRA 适配器
```math
对于线性层 $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$，LoRA 分解为：
```

$$ \mathbf{W}' = \mathbf{W} + \mathbf{B} \mathbf{A} $$

其中：
- $\mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$, $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}$
- $r = 128$ 是秩，$\alpha = 256$ 是缩放因子

前向传播变为：
```math
$$ \mathbf{y} = \mathbf{W}' \mathbf{x} = \mathbf{W} \mathbf{x} + \mathbf{B} \mathbf{A} \mathbf{x} $$
```
---

## 📊 五、关键数学洞察

### 5.1 损失权重设计原理

- **Focal Loss 权重 100**：由于掩码预测是像素级任务，样本数量极大，需要高权重进行平衡。
- **其他掩码损失权重 5**：作为辅助损失，提供额外的监督信号。
- **语言损失权重 1**：保持原始语言生成能力。

### 5.2 编码设计的数学优势

1.  **Fourier 编码**：使模型能够学习高频的空间位置信息。
2.  **类型嵌入**：让模型区分不同提示类型的语义含义。
3.  **掩码池化**：实现从密集掩码到紧凑表示的压缩。

### 5.3 记忆库的数学意义

通过记忆库，模型实现了：
```math
$$ P(\text{Response} | \mathcal{X}, \mathcal{T}) \rightarrow P(\text{Response} | \mathcal{X}, \mathcal{T}, \mathcal{M}) $$
```
即从条件概率到增强条件概率的转变，显著提升了对象感知能力。

这些数学设计共同构成了 UniPixel 强大的像素级推理能力的基础。
