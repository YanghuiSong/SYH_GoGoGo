这篇论文提出的 **PWOOD（Partial Weakly-Supervised Oriented Object Detection）** 框架，旨在解决遥感图像和复杂场景下，获取大量且高质量的定向边界框（OBB）标注极其昂贵的问题 。

PWOOD 的核心突破在于：它不仅能在“部分弱标注”（例如只有 10%-30% 的数据带有水平框或单点标注，其余全是无标注数据）的极端条件下完成训练，还能达到甚至超越传统半监督方法（使用昂贵旋转框标注）的检测精度 。

以下是对 PWOOD 框架整个算法流程、数学原理及网络架构的极其详尽的解析：

---

### 一、 整体网络架构与范式

PWOOD 建立在经典的师生模型（Teacher-Student Paradigm）之上 。

* 
**基础检测器**：算法采用了基于无锚框（Anchor-free）机制的 FCOS 架构 。


* 
**特征提取网络**：模型使用 **ViT（Vision Transformer）** 视觉编码器作为主干网络（Backbone），提取全局和局部特征，随后接入特征金字塔网络（FPN）作为 Neck，以处理多尺度目标 。


* 
**双分支结构**：包括一个**方向与尺度感知学生模型（OS-Student）**和一个**弱增强教师模型（Teacher）** 。两者在结构上完全一致，但在训练过程中扮演不同的角色，通过指数移动平均（EMA）进行参数传递 。



---

### 二、 核心组件 1：OS-Student (方向与尺度感知学生模型)

在仅有水平框（HBox）或单点（Point）标注的情况下，模型是没有任何“旋转角度”和“精确边界框尺度”信息的 。OS-Student 通过两个极其巧妙的自监督学习模块，实现了从无到有的信息提取。

#### 1. 方向学习 (Orientation Learning)

该模块基于**对称性原则（Symmetry-aware）**。如果输入图像发生了几何变换（如翻转、旋转），网络对目标的预测也应产生严格对应的数学映射 。
网络分别对原始视图（Original View）和变换视图（Augmentation View）进行预测，并强制其角度预测满足一致性。其角度损失函数 $\mathcal{L}_{Ang}^{s}$ 定义为：


$$\mathcal{L}_{Ang}^{s}=\begin{cases}L_{Ang}^{s}(\theta_{flip}+\theta,0)&trans=flip\\ L_{Ang}^{s}(\theta_{rot}-\theta,\mathcal{R})&trans=rotate\end{cases}$$

其中，$L_{Ang}^{s}$ 为 Smooth-L1 损失，$trans$ 代表图像变换方式（垂直翻转或随机旋转角度 $\theta$）。

#### 2. 尺度学习 (Scale Learning)

这是针对“单点标注”这种极端弱监督情况设计的 。单点只提供了位置，没有大小。OS-Student 通过约束目标尺度的“上限”和“下限”来逼近真实尺度 ：

* 
**尺度上限（Overlap Upper Bound）**：为了防止预测框无限变大，模型将定向边界框建模为二维高斯分布 $\mathcal{N}$ 。通过巴氏系数（Bhattacharyya coefficient）$B$ 计算不同预测框之间的重叠度，并对其进行最小化，从而形成排斥力，限制目标过度膨胀 。高斯重叠损失 $\mathcal{L}_{O}^{s}$ 为：



$$\mathcal{L}_{O}^{s}=\frac{1}{N}\sum_{i,j=1,i\ne j}^{N}B(\mathcal{N}_{i},\mathcal{N}_{j})$$





* 
**尺度下限（Watershed Lower Bound）**：利用 Voronoi 图和分水岭算法（Watershed algorithm）。以单点标注作为前景标记，Voronoi 图的脊线作为背景标记，对图像进行分割，得到每个目标的初始“盆地（basin）”区域 。将其旋转对齐后，作为目标的伪宽高回归目标（$w_t, h_t$）。随后通过高斯 Wasserstein 距离（GWD）损失进行回归 ：



$$\mathcal{L}_{W}^{s}=L_{GWD}\left(\begin{bmatrix}w/2&0\\ 0&h/2\end{bmatrix}^{2},\begin{bmatrix}w_{t}/2&0\\ 0&h_{t}/2\end{bmatrix}^{2}\right)$$






综合以上，OS-Student 在弱标注数据上的总监督损失 $\mathcal{L}^{s}$ 包含分类、中心度、框回归以及新引入的方向和尺度损失：


$$\mathcal{L}^{s}=\alpha_{1}\mathcal{L}_{cls}^{s}+\alpha_{2}\mathcal{L}_{cen}^{s}+\alpha_{3}\mathcal{L}_{box}^{s}+\alpha_{4}\mathcal{L}_{Ang}^{s}+\alpha_{5}\mathcal{L}_{O}^{s}+\alpha_{6}\mathcal{L}_{W}^{s}$$



---

### 三、 核心组件 2：CPF (类别无关的伪标签过滤)

在半监督学习中，Teacher 模型需要为无标注数据生成伪标签供 Student 学习 。传统方法通常设定一个静态阈值（如 Confidence > 0.5），但这在训练初期（Teacher 能力弱，低分居多）和后期（Teacher 能力强，高分居多）会导致严重的阈值不一致性，极大地影响鲁棒性 。

为了消除对静态阈值的依赖，论文提出了 **CPF（Class-Agnostic Pseudo-Label Filtering）**，这是一种完全数据驱动的动态过滤策略 。

1. 
**高斯混合模型（GMM）建模**：将 Teacher 输出的所有预测框的置信度得分 $s$，建模为两个一维高斯分布（正样本分布 $\mathcal{N}_{p}$ 和负样本分布 $\mathcal{N}_{n}$）的混合分布 $\mathcal{P}(s)$ ：



$$\mathcal{P}(s)=w_{p}\mathcal{N}_{p}(\mu_{p},(\sigma_{p})^{2})+w_{n}\mathcal{N}_{n}(\mu_{n},(\sigma_{n})^{2})$$





2. 
**动态初始化**：正样本均值 $\mu_{p}^{(0)}$ 初始化为当前得分矩阵的最大值，负样本均值 $\mu_{n}^{(0)}$ 初始化为最小值，两者权重各为 0.5 。


3. 
**EM 算法求解**：通过期望最大化（Expectation-Maximization）算法迭代，推导出某个检测框属于“正样本”（即应当作为高质量伪标签）的后验概率 $\mathcal{P}_{p}$ 。


4. **自适应阈值**：最终的动态过滤阈值 $T_d$ 定义为使正样本后验概率最大化的点：


$$T_{d}=\text{argmax }\mathcal{P}_{p}(s,\mu_{p},(\sigma_{p})^{2})$$


得分大于等于 $T_d$ 的预测框才会被保留为有效伪标签 。



---

### 四、 全流程训练与优化机制

PWOOD 的整体训练流程被精心设计为一个相互促进的正反馈循环 ：

**阶段 1：Burn-in（预训练阶段）**
首先，仅使用极少量的部分弱标注数据（Partial Weakly Labeled Data）对 OS-Student 进行预训练 。在这一阶段，OS-Student 依靠方向学习和尺度学习模块，初步掌握从弱标注中推断目标完整姿态的能力 。达到一定步数后，将 OS-Student 的参数镜像复制给 Teacher 模型，宣告预训练结束 。

**阶段 2：半监督相互学习（联合训练阶段）**
引入大量的无标注数据（Unlabeled Data）。

1. 
**Teacher 预测**：无标注数据经过弱数据增强（Weak Aug.）输入 Teacher 模型，生成初始预测框 。


2. 
**CPF 提纯**：利用上述的 GMM 和 EM 算法，动态计算出当前特征分布下的最优阈值 $T_d$，过滤掉低质量框，生成高质量的伪标签（Pseudo-labels）。


3. 
**Student 学习**：无标注数据经过强数据增强（Strong Aug.，如大角度旋转、颜色抖动等）输入 OS-Student 模型 。Student 的预测结果与 Teacher 提供的伪标签计算无监督损失 $\mathcal{L}^{u}$ 。



$$\mathcal{L}^{u}=\omega(\mathcal{L}_{cls}^{u}+\mathcal{L}_{cen}^{u}+\mathcal{L}_{box}^{u})$$


这里的权重 $\omega$ 与该点的置信度得分相关，确保高置信度的点在优化中占据主导地位 。


4. 
**EMA 更新**：OS-Student 通过反向传播更新自身参数后，采用指数移动平均（EMA）的方式平滑地更新 Teacher 模型的参数 。这使得 OS-Student 学到的方向和尺度能力反哺给 Teacher，从而在下一轮迭代中生成更精准的伪标签 。



整个 PWOOD 框架的最终优化目标为：


$$\mathcal{L}=\alpha\mathcal{L}^{s}+\beta\mathcal{L}^{u}$$

其中 $\alpha$ 和 $\beta$ 在实验中均默认设置为 1 。
