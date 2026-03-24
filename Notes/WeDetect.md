

# WeDetect: Fast Open-Vocabulary Object Detection as Retrieval 详细解析

## 1. 核心思想与创新点

WeDetect提出了一种全新的开放词汇对象检测（Open-Vocabulary Object Detection）方法，其核心思想是将对象检测问题视为**检索问题**，而非传统的跨模态融合方法。这一思想带来了三个关键优势：

1. **高效推理**：避免了计算密集的跨模态融合层，显著提高推理速度
2. **任务扩展**：统一了检测、提案生成、对象检索和参考表达理解（REC）等任务
3. **性能优越**：在多个基准测试上达到SOTA性能

与传统方法（如Grounding-DINO、GLIP、LLMDet）相比，WeDetect不使用任何跨模态融合层，而是采用双塔（dual-tower）架构，将视觉特征和文本特征映射到共享嵌入空间，通过简单的点积进行匹配。

## 2. 模型家族架构

WeDetect是一个模型家族，包含三个主要模型：

### 2.1 WeDetect（开放词汇对象检测器）



- **文本编码器**：从XLM-RoBERTa初始化
- **视觉编码器**：YOLO-like架构，包含ConvNeXt骨干网络、CSPRepBiFPAN颈部和YOLO-World对比头部
- **分类**：通过图像网格特征和类别文本嵌入之间的点积进行，无任何融合层

**关键创新**：
- 采用ConvNeXt骨干网络替代ViT，提供自然的多尺度特征
- 通过高质量数据集和训练策略实现卓越性能

### 2.2 WeDetect-Uni（通用提案生成器）



- **架构**：与WeDetect共享参数，仅微调一个可学习的对象性提示（objectness prompt）
- **功能**：生成类特定的提案，用于对象检索任务

### 2.3 WeDetect-Ref（基于LLM的REC模型）



- **架构**：基于Qwen3-VL，添加区域投影器
- **功能**：将REC任务建模为检索任务，避免序列预测

## 3. 关键算法原理详解

### 3.1 双塔架构与检索机制

WeDetect的核心是将对象检测视为检索问题，其数学表达为：

$$S_{ij} = V_i \cdot T_j$$

其中：
- $V_i$：图像网格特征（视觉特征）
- $T_j$：类别文本嵌入（文本特征）
- $S_{ij}$：区域 $i$ 与类别 $j$ 的匹配分数

**与传统融合方法的区别**：
- 传统方法：
  $$V_i \rightarrow FusionLayer \rightarrow F(V_i, T_j) \rightarrow Classification$$
- WeDetect：
  $$V_i \cdot T_j \rightarrow Classification$$

这种设计避免了融合层，使推理速度显著提高。

### 3.2 数据集构建：高质量数据引擎

作者提出了一套自动化的数据引擎，构建了包含1500万张图像和3.3亿个边界框的高质量数据集：

**数据构建流程**：

1. **源图像采样**：
   - 从SAM-1B、LAION、CC12M、Zero和自爬取的许可网站图像中采样
   - 使用图像配对的原始标题选择稀有名词，平衡概念

2. **边界框注释管道**：
   - 使用物体性检测器检测图像中所有区域（确保注释完整性）
   - 使用Qwen2.5-VL 7B生成实例特定的层次标签（如"动物, 狗, 黄色的狗"）
   - 对Qwen2.5-VL进行微调，使其：
     - 输出结构化输出（先粗粒度，后细粒度）
     - 拒绝生成错误框的标签（"拒绝识别"能力）

**多粒度标签采样**：
- 每个对象有多个标签（如"动物, 狗, 黄色的狗"）
- 训练时从候选列表中独立采样一个标签
- 提供丰富的监督和多样化的负样本

### 3.3 WeDetect-Uni：类特定提案

WeDetect-Uni的核心创新是：**即使冻结整个检测器，其框嵌入仍然是类特定的**。

**工作原理**：
1. 冻结WeDetect的整个模型
2. 仅训练一个对象性提示（objectness prompt）用于分类
3. 由于检测器是冻结的，最高得分提案对应的框嵌入保持类特定性

**对象检索（Object Retrieval）任务**：
- 与图像-文本检索不同，关注局部语义（如小对象"香烟头"）
- 评估方式：类别名称作为查询，验证集作为数据库，包含特定类别的图像作为真实标签
- 评估指标：精确度、召回率和F1分数

**应用流程**：
1. 使用WeDetect-Uni提取所有感兴趣区域
2. 将对应于最高得分提案的框嵌入缓存起来表示图像
3. 新查询到达时，只需简单的点积进行快速检索

### 3.4 WeDetect-Ref：检索式REC

WeDetect-Ref将参考表达理解（REC）任务建模为检索任务，避免了序列预测的瓶颈。

**关键公式**：

$$
\{h_i\}_{i=1}^n = LLM(I, q, \{o_i\}_{i=1}^n)
$$

$$
\{s_i\}_{i=1}^n = Sigmoid(Classifier(\{h_i\}_{i=1}^n)) \in [0, 1]
$$

其中：
- $I$：完整图像token
- $q$：用户查询
- $\{o_i\}_{i=1}^n$：对象token列表
- $\{s_i\}_{i=1}^n$：每个对象的分类分数

**三阶段训练方案**：

1. **区域投影器训练**：
   - 引入特殊token "⟨object⟩"作为占位符
   - 使用RoIAlign从视觉特征图中提取多尺度特征
   - 通过线性层将RoI特征压缩为单个token
   - 仅微调区域投影器（70万图像级和区域级字幕数据集）

2. **区域感知微调**：
   - 进一步微调LLM和投影器
   - 包含更多图像级和区域级指令微调数据（约170万数据）
   - 使LLM能准确感知特定对象

3. **区域分类微调**：
   - 将LLM微调为分类器
   - 丢弃原始语言建模头
   - 仅在对象token的隐藏嵌入上训练二元分类头
   - 数据格式：`user: ⟨image⟩ 请在图像中检测"类别名称"。assistant: ⟨object⟩⟨object⟩...`

## 4. 实现细节与技术优势

### 4.1 模型架构细节

- **视觉编码器**：ConvNeXt骨干网络提供多尺度特征
  - 与YOLO-World类似，但不包含融合层
  - 使用CSPRepBiFPAN颈部和YOLO-World对比头部

- **文本编码器**：从XLM-RoBERTa初始化
  - 每个类别名称单独编码为单个嵌入

- **损失函数**：与YOLO-World相同
  - 区域-文本对比损失（用于分类）
  - 边界框回归损失

### 4.2 训练策略

**分阶段训练**：
1. **预训练阶段**：在大规模图像-文本数据集上进行CLIP式的图像级对比学习
   - 初始化视觉骨干和语言编码器

2. **颈部/头部训练**：冻结视觉骨干和语言编码器，仅训练颈部和头部
   - 20个epoch，学习率5e-4

3. **端到端训练**：训练所有参数
   - 30个epoch，学习率1e-5
   - 总batch size为320

**多粒度标签采样**：
- 为每个对象从候选列表中独立采样一个标签
- 丰富了监督并提供了多样化的负样本

### 4.3 推理效率对比

| 模型 | Backbone | 分辨率 | #Params | FPS | LVIS AP |
|------|----------|--------|---------|-----|---------|
| WeDetect-Tiny | ConvNext-T | 640×640 | 33M | 62.5 | 31.4 |
| YOLO-World-L | YOLOv8-L | 640×640 | 48M | 54.6 | 29.4 |
| Grounding-DINO-L | Swin-T | 800×1333 | 172M | 3.1 | 33.9 |
| LLMDet | Swin-T | 800×1333 | 172M | 6.0 | 40.1 |

**关键观察**：
- WeDetect-Tiny比YOLO-World-L快8 FPS，且性能更高（+2.0 AP on LVIS）
- WeDetect-Large比Grounding-DINO-L快20倍，性能更高（+9.3 AP on LVIS）

### 4.4 WeDetect-Uni的性能

| 模型 | COCO AR50 | LVIS AR50 | PACO-LVIS AR50 |
|------|-----------|-----------|---------------|
| WeDetect-Base-Uni | 66.7 | 69.3 | 69.3 |
| WeDetect-Large-Uni | 66.7 | 69.3 | 69.3 |

**关键观察**：
- WeDetect-Uni在所有数据集上获得最高召回率
- 300个提案时，COCO AR50达到73.2
- 与传统方法相比，召回率显著提高

### 4.5 WeDetect-Ref的性能

| 模型 | FPS | RefCOCOg Avg. |
|------|-----|--------------|
| Qwen3-VL 4B | 0.4 | 86.6 |
| WeDetect-Ref 4B | 5.3 | 93.2 |

**关键观察**：
- WeDetect-Ref 4B比Qwen3-VL 4B高6.6分
- 推理速度提高13倍（5.3 vs 0.4 FPS）
- 在COCO上达到50.0 AP，首次超过传统检测器

## 5. 与现有方法的对比分析

### 5.1 与传统融合方法的对比

| 方法 | 融合层 | 推理速度 | LVIS AP |
|------|--------|----------|---------|
| GLIP | 有 | 5.4 FPS | 35.9 |
| Grounding-DINO | 有 | 3.1 FPS | 33.9 |
| LLMDet | 有 | 6.0 FPS | 40.1 |
| WeDetect | 无 | 62.5 FPS | 31.4 |

**关键发现**：
- WeDetect避免了融合层，推理速度提高了10-20倍
- 性能不逊于融合方法，甚至在某些指标上更高

### 5.2 与基于LLM的方法对比

| 方法 | FPS | RefCOCOg Avg. |
|------|-----|--------------|
| Qwen3-VL 4B | 0.4 | 86.6 |
| WeDetect-Ref 4B | 5.3 | 93.2 |

**关键发现**：
- WeDetect-Ref通过避免序列预测，推理速度提高13倍
- 性能提升6.6分，同时保持高推理速度

## 6. 新任务：对象检索（Object Retrieval）

WeDetect提出了一项新任务：**对象检索**，旨在检索包含用户指定对象类别的图像，特别是小对象（如"香烟头"）。

**与传统图像-文本检索的区别**：
- 传统：关注全局图像语义
- 对象检索：关注局部语义，适合小对象

**评估设置**：
- 查询：类别名称
- 数据库：验证集
- 真实标签：包含特定类别的图像
- 评估指标：精确度、召回率和F1分数

**结果**：
- WeDetect-Large-Uni在COCO上达到82.6%的精确度，57.5%的召回率，F1分数为83.6
- 比CLIP高出37.2 F1分数

## 7. 详细实验分析

### 7.1 WeDetect在LVIS上的性能

| 模型 | LVIS minival AP | LVIS AP | COCO AP |
|------|----------------|----------|----------|
| WeDetect-Tiny | 37.4 | 31.4 | 44.9 |
| YOLO-World-L | 35.4 | 29.4 | 42.8 |
| LLMDet | 41.4 | 35.9 | 45.8 |

**关键发现**：
- WeDetect-Tiny在LVIS minival上比YOLO-World-L高2.0 AP
- 在LVIS上比YOLO-World-L高2.0 AP
- COCO AP上高2.1 AP

### 7.2 WeDetect-Ref在REC任务上的性能

| 模型 | RefCOCOg Avg. |
|------|--------------|
| Qwen3-VL 4B | 86.6 |
| WeDetect-Ref 4B | 93.2 |

**关键发现**：
- WeDetect-Ref 4B在RefCOCOg上达到93.2的平均分数
- 比Qwen3-VL 4B高6.6分
- 速度比Qwen3-VL 4B快13倍

### 7.3 WeDetect-Ref在COCO上的性能

| 模型 | COCO AP | COCO APs | COCO APm | COCO APl |
|------|----------|-----------|-----------|-----------|
| Grounding-DINO-T | 48.4 | 51.4 | - | - |
| Qwen2.5-VL 7B | 17.7 | 37.3 | - | - |
| WeDetect-Ref 4B | 50.0 | 34.7 | 57.6 | 69.2 |

**关键发现**：
- WeDetect-Ref 4B是第一个在COCO上达到50+ AP的LLM
- COCO AP达到50.0，比Qwen2.5-VL 7B高32.3 AP
- 在COCO APl上达到69.2，远超其他LLM

## 8. 优势与局限性

### 8.1 优势

1. **高效推理**：无融合层设计使推理速度显著提高
2. **任务扩展**：统一了检测、提案生成、对象检索和REC任务
3. **性能卓越**：在多个基准测试上达到SOTA
4. **数据效率**：通过多粒度标签采样，利用有限数据获得高质量性能

### 8.2 局限性

1. **多查询处理**：WeDetect-Ref不能在单次前向传播中处理多个查询
   - 解决方案：对每个查询单独运行，然后合并结果
2. **训练数据需求**：需要高质量的多粒度标注数据
   - 解决方案：作者开发了自动数据引擎

## 9. 未来方向

1. **扩展到更多任务**：将检索框架扩展到其他视觉任务
2. **优化多查询处理**：改进WeDetect-Ref以支持多查询处理
3. **降低训练成本**：进一步优化数据收集和训练流程
4. **跨语言支持**：增强对多语言查询的支持

## 10. 总结

WeDetect提出了一种基于检索的开放词汇对象检测方法，通过以下创新实现了性能和效率的双重提升：

1. **双塔架构**：避免了计算密集的跨模态融合层
2. **检索机制**：将对象检测视为检索问题，实现快速推理
3. **高质量数据**：开发了数据引擎，构建了包含1500万张图像和3.3亿个边界框的高质量数据集
4. **任务扩展**：提出了对象检索任务，并将REC建模为检索任务

WeDetect在多个基准测试上达到了SOTA性能，同时推理速度比现有方法快10-20倍。特别是WeDetect-Ref通过避免序列预测，将REC任务的推理速度提高了13倍，同时性能也显著提升。

这一工作展示了检索范式在开放词汇视觉任务中的强大潜力，为未来的研究提供了新的方向。


## 📊 WeDetect 代码仓库深度技术解析
基于对整个代码仓库的全面分析，我将为您详细解读这个开放词汇目标检测系统的设计哲学和实现细节。

---

## 🎯 一、核心设计理念
### 1.1 研究动机与创新点
**WeDetect** 是一个**实时开放词汇目标检测器**，其核心创新在于：

1. **双塔架构（Dual-Tower Architecture）**：
    - 摒弃传统跨模态融合层，将识别转化为检索问题
    - 图像编码器和文本编码器独立工作，在共享嵌入空间进行匹配
    - 显著提升推理速度，实现实时检测
2. **多功能统一框架**：
    - **WeDetect**: 基础检测器
    - **WeDetect-Uni**: 通用候选框生成器（支持物体检索）
    - **WeDetect-Ref**: 基于 MLLM 的指代表达式理解（REC）
3. **技术优势**：
    - 无需跨模态融合 → 推理速度快
    - Prompt Tuning → 快速适配新类别
    - 单次前向传播分类 → 高效 REC

---

## 🏗️ 二、系统架构详解
### 2.1 整体架构图
```plain
输入图像 → [Image Encoder] → Image Features ↘
                                              [对比头] → 检测分数 + BBox
文本提示 → [Text Encoder]  → Text Features ↗
```

### 2.2 核心组件分析
#### **A. Backbone 模块** ([mm_backbone.py](file://d:\SYH\CodeReading\WeDetect\wedetect\models\backbones\mm_backbone.py))
**多模态 backbone 设计**：

```python
class MultiModalYOLOBackbone(BaseModule):
    def __init__(self, image_model, text_model, frozen_stages=-1):
        self.image_model = MODELS.build(image_model)  # ConvNeXt
        self.text_model = MODELS.build(text_model)    # XLM-Roberta
        
    def forward(self, image, text):
        img_feats = self.image_model(image)  # 多尺度特征
        txt_feats = self.text_model(text)    # 文本嵌入
        return img_feats, txt_feats
```

**关键实现细节**：

1. **视觉编码器** (`ConvNextVisionBackbone`):
    - 基于 ConvNeXt 架构
    - 输出 4 个尺度特征图 (c1, c2, c3, c4)
    - 支持 tiny/base/large/xlarge 多种配置
    - 通过 [frozen_modules](file://d:\SYH\CodeReading\WeDetect\generate_class_embedding.py#L0-L0) 控制参数冻结
2. **文本编码器** ([XLMRobertaLanguageBackbone](file://d:\SYH\CodeReading\WeDetect\generate_class_embedding.py#L17-L70)):
    - 使用 XLM-RoBERTa (Base: 768dim, Large: 1024dim)
    - 投影头将文本特征映射到 768 维共享空间
    - L2 归一化确保特征可比性

```python
txt_feats = self.model(**text)["last_hidden_state"][:, 0]  # [CLS] token
txt_feats = self.head(txt_feats)  # 投影到 768 维
txt_feats = F.normalize(txt_feats, dim=-1)  # L2 归一化
```

3. **冻结策略** (`_freeze_modules`):
    - 支持部分冻结或全部冻结
    - 微调时可选择性更新参数

---

#### **B. Neck 模块** ([yolo_world_pafpn.py](file://d:\SYH\CodeReading\WeDetect\wedetect\models\necks\yolo_world_pafpn.py))
**特征金字塔融合**：

```python
class CSPRepBiFPANNeck(nn.Module):
    """双向特征金字塔网络"""
    def forward(self, inputs):
        # Bottom-up 路径
        p5, p4, p3 = self.top_down_path(inputs)
        
        # Top-down 路径
        n3, n4, n5 = self.bottom_up_path(p3, p4, p5)
        
        return n3, n4, n5
```

**关键技术点**：

1. **CSP (Cross Stage Partial)**: 减少计算量同时保持性能
2. **RepVGG Block**: 重参数化卷积，训练时多分支，推理时单路
3. **BiFPN**: 双向特征融合，增强多尺度表达能力

**RepVGG 块实现**：

```python
class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        self.rbr_dense = ConvModule(in_ch, out_ch, 3x3)
        self.rbr_1x1 = ConvModule(in_ch, out_ch, 1x1)
        self.rbr_identity = BatchNorm() if in_ch==out_ch else None
    
    def forward(self, x):
        # 训练时：三路并行
        return self.nonlinearity(
            self.rbr_dense(x) + self.rbr_1x1(x) + self.rbr_identity(x)
        )
    
    def switch_to_deploy(self):
        # 推理时：融合为单路卷积
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(...)
```

---

#### **C. Head 模块** ([yolo_world_head.py](file://d:\SYH\CodeReading\WeDetect\wedetect\models\dense_heads\yolo_world_head.py))
**对比学习检测头**：

```python
class YOLOWorldHeadModule(YOLOv8HeadModule):
    def _init_layers(self):
        # 分类预测器（解耦头）
        self.cls_preds = nn.Sequential(
            ConvModule(in_ch, cls_ch, 3x3),
            ConvModule(cls_ch, cls_ch, 3x3),
            nn.Conv2d(cls_ch, embed_dims, 1x1)
        )
        
        # 对比头（计算区域 - 文本相似度）
        self.cls_contrasts = ContrastiveHead(embed_dims)
        
        # 回归预测器
        self.reg_preds = nn.Sequential(...)
```

**对比头核心逻辑**：

```python
class ContrastiveHead(BaseModule):
    def __init__(self, embed_dims):
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
        self.bias = nn.Parameter(torch.zeros([]))
    
    def forward(self, x, w):
        # L2 归一化
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        
        # 余弦相似度计算（使用 einsum 优化）
        x = torch.einsum('bchw,bkc->bkhw', x, w)
        x = x * self.logit_scale.exp() + self.bias
        return x
```

**损失函数设计**：

```python
def loss_by_feat(self, cls_scores, bbox_preds, batch_gt_instances):
    # 1. Task Aligned Assigner 分配样本
    assigned_result = self.assigner(
        flatten_pred_bboxes.detach(),
        flatten_cls_preds.detach().sigmoid(),
        gt_labels, gt_bboxes
    )
    
    # 2. 分类损失（交叉熵）
    loss_cls = self.loss_cls(flatten_cls_preds, assigned_scores)
    
    # 3. 定位损失（CIoU）
    loss_bbox = self.loss_bbox(pred_bboxes_pos, assigned_bboxes_pos)
    
    # 4. 分布焦点损失（DFL）
    loss_dfl = self.loss_dfl(pred_dist_pos, assigned_ltrb_pos)
    
    return dict(loss_cls, loss_bbox, loss_dfl)
```

---

### 2.3 WeDetect-Uni 扩展
**通用候选框生成器**：

```python
class SimpleYOLOWorldDetector(YOLODetector):
    def __init__(self, prompt_dim=512, num_prompts=256):
        # 可学习的 objectness prompts
        self.embeddings = nn.Parameter(torch.randn(num_prompts, prompt_dim))
        
        # 可选的 MLP adapter
        self.adapter = nn.Sequential(
            nn.Linear(prompt_dim, prompt_dim*2),
            nn.ReLU(),
            nn.Linear(prompt_dim*2, prompt_dim)
        )
    
    def extract_feat(self, batch_inputs, batch_data_samples):
        # 仅提取图像特征
        img_feats, _ = self.backbone(batch_inputs, None)
        
        # 使用 learnable prompts
        txt_feats = self.embeddings[None].repeat(img_feats[0].shape[0], 1, 1)
        if self.adapter:
            txt_feats = self.adapter(txt_feats) + txt_feats
            txt_feats = F.normalize(txt_feats, dim=-1)
        
        return img_feats, txt_feats
```

**关键创新**：

+ 冻结整个检测器，仅微调 objectness prompt
+ 生成的 proposals 具有类别特定嵌入
+ 支持物体检索应用

---

### 2.4 WeDetect-Ref 扩展
**MLLM-based 分类器**：

```python
class Qwen3VLGroundingForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def forward(self, input_ids, pixel_values, bboxes, ori_shapes):
        # 1. 提取多尺度图像特征
        image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values)
        
        # 2. RoI Align 提取候选框特征
        for i, bbox in enumerate(bboxes):
            roi_feats1 = torchvision.ops.roi_align(scale1_feat, [bbox], 7)
            roi_feats2 = torchvision.ops.roi_align(scale2_feat, [bbox], 7)
            roi_feats3 = torchvision.ops.roi_align(scale3_feat, [bbox], 7)
            
            # 3. 多尺度特征融合
            roi_feats = torch.cat([roi_feats1, roi_feats2, roi_feats3], dim=1)
            roi_feats = self.merge(roi_feats)
            
            # 4. 位置编码
            box_coor = box_xyxy_to_cxcywh(bbox) / img_size
            box_pos = self.object_pos_projector(gen_sineembed(box_coor))
            
            object_features.append(roi_feats + box_pos)
        
        # 5. 替换<object> token 的嵌入
        inputs_embeds = inputs_embeds.masked_scatter(
            object_id_mask, 
            torch.cat(object_features, dim=0)
        )
        
        # 6. 标准 LLM 前向传播
        outputs = self.language_model(inputs_embeds=inputs_embeds)
        logits = self.lm_head(outputs.last_hidden_state)
        
        return logits
```

**设计亮点**：

1. **单次前向传播分类**：摒弃 next-token prediction
2. **多尺度 RoI 特征**：融合 3 个尺度特征（8x, 16x, 32x）
3. **位置感知**：添加正弦位置编码
4. **支持复杂查询**：处理长文本指代描述

---

## 🔧 三、训练与推理流程
### 3.1 训练流程
**配置文件结构** ([wedetect_base.py](file://d:\SYH\CodeReading\WeDetect\config\wedetect_base.py)):

```python
model = dict(
    type="YOLOWorldDetector",
    backbone=dict(
        type="MultiModalYOLOBackbone",
        image_model=dict(type="ConvNextVisionBackbone", model_name="base"),
        text_model=dict(type="XLMRobertaLanguageBackbone", model_name="./xlm-roberta-base/")
    ),
    neck=dict(type="CSPRepBiFPANNeck", scale_factor=1.0),
    bbox_head=dict(
        type="YOLOWorldHead",
        head_module=dict(
            type="YOLOWorldHeadModule",
            use_bn_head=True,  # 使用 BNContrastiveHead
            embed_dims=768,
            num_classes=80
        )
    )
)
```

**训练入口** ([train.py](file://d:\SYH\CodeReading\WeDetect\train.py)):

```python
# 1. 加载配置
cfg = Config.fromfile(args.config)

# 2. 构建 Runner
runner = Runner.from_cfg(cfg)

# 3. 开始训练
runner.train()
```

**数据流**：

```plain
图像 → LoadImageFromFile → Resize → LetterBox → Normalize
                                        ↓
文本 → LoadText → Tokenize → PackDetInputs
                                        ↓
                             DataBatch → Model
```

### 3.2 推理流程
**WeDetect 推理** ([infer_wedetect.py](file://d:\SYH\CodeReading\WeDetect\infer_wedetect.py)):

```python
# 1. 初始化模型
model = init_detector(cfg, checkpoint=args.checkpoint)

# 2. 文本编码（reparameterize）
model.reparameterize(texts)
# 内部调用：text_feats = backbone.forward_text(texts)

# 3. 前向传播
with torch.no_grad():
    output = model.test_step(data_batch)[0]
    
# 4. 后处理
pred_instances = output.pred_instances
pred_instances = pred_instances[pred_instances.scores > score_thr]

# 5. NMS 和可视化
```

**WeDetect-Ref 推理** ([infer_wedetect_ref.py](file://d:\SYH\CodeReading\WeDetect\infer_wedetect_ref.py)):

```python
# 1. 加载 Uni 模型生成 proposals
outputs = det_model([args.image])
proposals = outputs[0]['bboxes']

# 2. 构造 Ref 输入
proposal_str = "<object>" * len(proposals)
messages = [
    {"role": "user", "content": [{"type": "image", "image": image}, 
                                  {"type": "text", "text": query}]},
    {"role": "assistant", "content": [{"type": "text", "text": proposal_str}]}
]

# 3. Ref 模型分类
with torch.no_grad():
    pred = model(**model_inputs, bboxes=proposals)
    pred_scores = pred.logits.sigmoid()[proposal_positions]

# 4. 选择 top-k 或阈值过滤
```

---

## 💡 四、关键技术要点
### 4.1 重参数化技巧
**训练时多分支，推理时融合**：

```python
# 训练阶段
output = conv3x3(x) + conv1x1(x) + identity(x)

# 推理阶段（融合后）
kernel_fused = kernel3x3 + kernel1x1_padded
bias_fused = bias3x3 + bias1x1
output = conv_fused(x)
```

**优势**：

+ 训练时增强表达能力
+ 推理时无额外计算开销

### 4.2 Task Aligned Assigner
**动态样本分配策略**：

```python
class BatchTaskAlignedAssigner:
    def forward(self, pred_bboxes, pred_scores, anchors, gt_labels, gt_bboxes):
        # 1. 计算对齐度 metric = s^alpha * t^beta
        alignment_metric = pred_scores ** alpha * iou ** beta
        
        # 2. 每个 GT 选择 top-k 个 anchor
        topk_indices = alignment_metric.topk(tal_topk)
        
        # 3. 筛选正样本（IoU 阈值）
        fg_mask = iou > iou_threshold
        
        return dict(assigned_bboxes, assigned_scores, fg_mask)
```

### 4.3 分布式训练配置
**DeepSpeed 配置** ([zero3.json](file://d:\SYH\CodeReading\WeDetect\wedetect_ref\scripts\zero3.json)):

```json
{
  "fp16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "overlap_comm": true
  },
  "train_batch_size": 32,
  "gradient_accumulation_steps": 4
}
```

---

## 📈 五、性能优化策略
### 5.1 混合精度训练
```python
cfg.optim_wrapper.type = 'AmpOptimWrapper'
cfg.optim_wrapper.loss_scale = 'dynamic'
```

### 5.2 梯度累积
```python
# 当显存不足时使用
train_batch_size = per_gpu_bs * num_gpus * gradient_accumulation_steps
```

### 5.3 数据预取
```python
train_dataloader = dict(
    batch_size=16,
    num_workers=8,  # 多进程加载
    persistent_workers=True,  # 复用 worker
    pin_memory=True  # 加速 CPU→GPU 传输
)
```

---

## 🎓 六、使用示例
### 6.1 开放词汇检测
```bash
python infer_wedetect.py \
  --config config/wedetect_large.py \
  --checkpoint checkpoints/wedetect_large.pth \
  --image assets/demo.jpeg \
  --text '鞋，床，桌子' \
  --threshold 0.3
```

### 6.2 物体检索
```bash
cd eval_retrieval
# 提取嵌入
torchrun extract_embedding.py \
  --model wedetect \
  --dataset coco

# 检索评估
python retrieval_metric.py --thre 0.2
```

### 6.3 指代理解
```bash
python infer_wedetect_ref.py \
  --wedetect_ref_checkpoint checkpoints/wedetect_ref_4b.pth \
  --wedetect_uni_checkpoint checkpoints/wedetect_uni_base.pth \
  --image demo.jpg \
  --query "穿着红色衣服的人" \
  --visualize
```

---

## 🔚 七、总结与展望
### 7.1 技术贡献
1. **高效的非融合架构**：证明了无跨模态融合的可行性
2. **统一的检索框架**：检测、提议、检索、REC 的统一表述
3. **实用的工程实现**：完整的训练、推理、评估工具链

### 7.2 未来方向
1. **更强的骨干网络**：集成 VLM 特征（如 CLIP、SAM）
2. **端到端 REC**：联合训练检测和分类模块
3. **视频扩展**：时序建模能力

---
