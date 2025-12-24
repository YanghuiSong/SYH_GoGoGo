# **《SAM辅助的遥感图像语义分割：结合目标与边界约束》**- 网络层面详细解读

这篇论文题为**《SAM辅助的遥感图像语义分割：结合目标与边界约束》**，提出了一种利用**Segment Anything Model (SAM)** 的原始输出来增强遥感图像语义分割性能的简单而有效的框架。本文将从网络层面详细分析输入到输出的全过程变化，并对原笔记进行润色。

---

## 一、研究背景与问题

### 1. **遥感图像语义分割的重要性**
- 遥感图像语义分割旨在为每个像素赋予语义标签（如建筑、道路、植被等），广泛应用于**环境监测、土地覆盖制图、灾害评估**等地理信息任务。
- 传统方法依赖深度学习模型（如CNN、Transformer），但**标注数据获取成本高**，且遥感图像与自然图像在分辨率、视角、物体尺度等方面差异显著。

### 2. **SAM 的优势与局限**
- **Segment Anything Model (SAM)** 是Meta AI推出的**通用图像分割基础模型**，具备强大的零样本分割能力。
- 但SAM存在两个主要问题：
  1. **生成的掩码没有语义标签**（只能分割，不能分类）。
  2. **在遥感图像上表现不佳**，因为其训练数据主要为自然图像，遥感图像中的物体尺度、纹理、背景复杂度差异大。

### 3. **现有方法的不足**
- 现有方法多通过**微调SAM、设计复杂提示词、或构建多阶段网络**来适应遥感任务，但这些方法往往：
  - 依赖特定数据集设计，**通用性差**。
  - 需要**额外模块或训练策略**，复杂度高。
  - 大多只适用于**二分类任务**，难以扩展到多类别语义分割。

---

## 二、核心思想：利用SAM的原始输出

作者提出直接利用SAM生成的两种中间结果：

1. **SAM生成的对象 (SGO, SAM-Generated Object)**
   - SAM通过网格提示自动分割图像中的**潜在物体区域**。
   - 每个区域被视为一个"对象"，即使没有语义标签，也保留了**物体的空间结构信息**。

2. **SAM生成的边界 (SGB, SAM-Generated Boundary)**
   - 从SGO中提取每个物体的**外轮廓**，形成边界图。
   - 边界信息可用于**增强分割结果的边缘精度**。

> 这两种信息都是**无需额外训练或标注**的，直接来自SAM的原始输出。

---

## 三、方法框架：网络层面的输入到输出变化

### 1. **整体流程**

```
输入图像 → 语义分割模型 → 预测分割图
         ↓
      SAM生成 SGO 和 SGB (预处理阶段)
         ↓
   计算三种损失函数：
   - 分割损失（交叉熵）
   - 对象一致性损失
   - 边界保留损失
         ↓
     反向传播更新模型
```

### 2. **网络结构分析**

以UNetFormer为例，详细分析网络层面的输入到输出变化：

#### a) **输入处理**
- 输入图像形状为 `(B, 3, H, W)`，其中B为批次大小，H、W为图像高度和宽度
- 图像首先被归一化到[0,1]区间，然后送入预训练的ResNet编码器

#### b) **编码器阶段**
- 使用预训练的ResNet作为骨干网络（如`swsl_resnet18`）
- 通过`features_only=True`参数提取多尺度特征图
- 输出4个不同分辨率的特征图：
  - `res1`: `(B, C1, H/4, W/4)` - 最细粒度特征
  - `res2`: `(B, C2, H/8, W/8)` - 中等粒度特征
  - `res3`: `(B, C3, H/16, W/16)` - 粗粒度特征
  - `res4`: `(B, C4, H/32, W/32)` - 最粗粒度特征

#### c) **解码器阶段** - 网络层面的详细变化

**i. 初始处理阶段**：
- `self.pre_conv`: 将res4特征从C4通道转换为decode_channels通道
- `self.b4`: 通过Block（包含GlobalLocalAttention）处理最粗粒度特征

**ii. 多级融合阶段**：
- 第1级融合（p3）：将上采样的x与res3进行融合
  - `x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)`
  - 使用加权融合：`x = fuse_weights[0] * self.pre_conv(res3) + fuse_weights[1] * x`
  - 经过卷积处理：`x = self.post_conv(x)`
- 第2级融合（p2）：将上采样的x与res2进行类似融合
- 第3级融合（p1）：使用FeatureRefinementHead处理x与res1的融合

**iii. 精细化处理阶段**：
- `self.segmentation_head`: 通过卷积序列将特征转换为类别数
  - [ConvBNReLU](file://d:\CodeReading\SSRS\SAM_RS\model\UNetFormer.py#L9-L16): 特征变换和归一化
  - `nn.Dropout2d`: 防止过拟合
  - [Conv](file://d:\CodeReading\SSRS\SAM_RS\model\UNetFormer.py#L28-L33): 最终将特征映射到类别数维度

#### d) **输出处理**
- 通过双线性插值将特征图恢复到原始输入尺寸：`F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)`
- 最终输出形状为 `(B, num_classes, H, W)`

### 3. **SAM预处理阶段**
- 使用SAM的**网格提示模式**自动生成物体掩码（SGO）：
  - 设置`pred_iou_thresh=0.96`，`crop_nms_thresh=0.5`，`box_nms_thresh=0.5`
  - 过滤面积小于50的区域，保留前50个最大物体
  - 从SGO中提取边界，形成SGB

### 4. **两种新提出的损失函数**

#### a) 对象一致性损失（Object Consistency Loss）

```python
class ObjectLoss(nn.Module):
  def __init__(self, max_object=50):
        super().__init__()
        self.max_object = max_object

  def forward(self, pred, gt):
    num_object = int(torch.max(gt)) + 1
    num_object = min(num_object, self.max_object)
    total_object_loss = 0

    for object_index in range(1,num_object):
        mask = torch.where(gt == object_index, 1, 0).unsqueeze(1).to('cuda')
        num_point = mask.sum(2).sum(2).unsqueeze(2).unsqueeze(2).to('cuda')
        avg_pool = mask / (num_point + 1)

        object_feature = pred.mul(avg_pool)

        avg_feature = object_feature.sum(2).sum(2).unsqueeze(2).unsqueeze(2).repeat(1,1,gt.shape[1],gt.shape[2])
        avg_feature = avg_feature.mul(mask)

        object_loss = torch.nn.functional.mse_loss(num_point * object_feature, avg_feature, reduction='mean')
        total_object_loss = total_object_loss + object_loss
      
    return total_object_loss
```

- **核心思想**：同一个物体内部的像素预测应该尽量一致。
- **计算过程**：
  1. 从SGO中提取第 i 个物体的掩码 M^i
  2. 获取该物体在模型预测中的区域 P^i = P ⊙ M^i
  3. 计算该区域的平均预测值 P_avg^i
  4. 计算该物体内所有像素预测值与平均值的均方误差（MSE）
- **作用**：鼓励模型在同一个物体内输出**平滑、一致的预测**，减少内部碎片化。

#### b) 边界保留损失（Boundary Preservation Loss）

```python
class BoundaryLoss(nn.Module):
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred, gt):
        n, _, _, _ = pred.shape
        pred = torch.softmax(pred, dim=1)
        class_map = pred.argmax(dim=1).cpu()

        # boundary map
        gt_b = F.max_pool2d(
            1 - gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - gt

        pred_b = F.max_pool2d(
            1 - class_map, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - class_map

        # extended boundary map
        gt_b_ext = F.max_pool2d(
            gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(
            pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, 2, -1)
        pred_b = pred_b.view(n, 2, -1)
        gt_b_ext = gt_b_ext.view(n, 2, -1)
        pred_b_ext = pred_b_ext.view(n, 2, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss
```

- **核心思想**：利用SAM提供的**高质量边界先验**，引导模型学习更准确的物体边界。
- 通过最大池化操作提取边界区域，并计算预测边界与SGB的匹配程度。

### 5. **总损失函数**

$$
L_{\text{total}} = L_{\text{seg}} + \lambda_o L_{\text{obj}} + \lambda_b L_{\text{bdy}}
$$

其中：
- $L_{\text{seg}}$ 是传统的语义分割损失（交叉熵）
- $\lambda_o = 1.0$, $\lambda_b = 0.1$ 是权重超参数

在训练过程中，总损失函数的计算方式为：

```python
if LOSS == 'SEG':
    loss = loss_ce
elif LOSS == 'SEG+BDY':
    loss = loss_ce + loss_boundary * LBABDA_BDY
elif LOSS == 'SEG+OBJ':
    loss = loss_ce + loss_object * LBABDA_OBJ
elif LOSS == 'SEG+BDY+OBJ':
    loss = loss_ce + loss_boundary * LBABDA_BDY + loss_object * LBABDA_OBJ
```

---

## 四、实验设计与结果

### 1. **数据集**
| 数据集 | 分辨率 | 类别数 | 场景类型 |
|--------|--------|--------|----------|
| ISPRS Vaihingen | 9 cm | 6类（5前景+背景） | 城镇 |
| LoveDA Urban | 30 cm | 7类 | 城市 |

### 2. **评估指标**
- **mF1（平均F1分数）**
- **mIoU（平均交并比）**

### 3. **实验结果**
- 在**四个主流语义分割模型**上测试（ABCNet、CMTFNet、UNetformer、FTUNetformer）。
- **均取得显著提升**，尤其在**建筑、车辆**等结构清晰的类别上。
- 在**边界复杂的类别**（如植被、裸地）上也有稳定改善。

### 4. **可视化对比**
- 论文提供了多组对比图，显示加入SAM辅助后：
  - 物体内部更一致
  - 边界更清晰
  - 错误分类减少

---

## 五、方法优势与创新点

1. **无需修改网络结构**：直接作为损失函数使用，兼容现有模型。
2. **无需语义标签**：利用SAM的**无标签对象和边界信息**。
3. **通用性强**：在两个差异显著的遥感数据集上均有效。
4. **计算代价低**：仅在训练阶段使用SAM预处理，推理阶段不增加计算负担。
5. **损失函数设计巧妙**：通过对象一致性损失和边界保留损失，有效利用SAM的先验知识。

---

## 六、未来展望

作者提出未来可探索：
- 针对**不同分辨率、不同地物类型**优化SGO/SGB生成策略。
- 将框架扩展至**无监督/半监督语义分割**。
- 结合其他视觉基础模型（如DINO、CLIP）进一步融合多模态信息。

---

## 七、总结

这篇论文提出了一种**简洁而有效**的方法，通过直接利用SAM的原始输出（对象掩码和边界）构建两个辅助损失函数，显著提升了遥感图像语义分割的性能。该方法**无需复杂网络设计或额外标注**，具有良好的通用性和可扩展性，为SAM在遥感领域的应用提供了新思路。

通过从网络层面的详细分析，我们可以看到该方法在各个网络模块上的具体实现，从输入图像到最终分割结果的完整处理流程，以及如何通过损失函数的约束实现性能提升，是一种实用且有效的技术方案。
