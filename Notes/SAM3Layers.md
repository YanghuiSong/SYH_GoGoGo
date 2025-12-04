# SAM3 完整工作流程详解（结合仓库说明）

## 一、模型概述

SAM3（Segment Anything Model 3）是Meta发布的第三代通用分割模型，支持图像和视频的多模态分割。相比于前代，SAM3的主要创新在于：

1. **开放式词汇能力**：能够处理超过27万个独特概念的文本查询
2. **双重架构设计**：解耦的检测器-跟踪器设计，最小化任务干扰
3. **Presence Token机制**：改进对相似文本提示的区分能力
4. **大规模数据引擎**：自动标注超过400万个独特概念

## 二、完整输入流程

### 2.1 输入规格

**图像输入**：
- 支持多种尺寸，推荐1024×1024
- 模型内部会resize到1008×1008进行处理
- 格式：PIL图像、numpy数组或torch tensor

**文本输入**：
- 自然语言描述，如"a dog"、"a player in white"
- 支持中英文（通过分词器处理）
- 最大长度：32个token

**几何提示（可选）**：
- 点提示（point prompts）：坐标+正负标签
- 框提示（box prompts）：边界框坐标
- 掩码提示（mask prompts）：参考掩码

### 2.2 输入预处理流程

```python
# 用户调用示例
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model()
processor = Sam3Processor(model)

# 设置图像
image = Image.open("image.jpg")
inference_state = processor.set_image(image)  # 步骤2.2

# 设置文本提示
output = processor.set_text_prompt(
    state=inference_state, 
    prompt="a dog"  # 步骤4.1
)
```

## 三、各层详细变换

### 3.1 ViT视觉编码器

**输入**：`[1, 3, 1008, 1008]`
**输出**：`[1, 1024, 72, 72]`

**详细处理**：
1. **Patch划分**：1008×1008 → 72×72个14×14的patch
2. **位置编码**：使用RoPE（Rotary Position Embedding）2D编码
3. **32层Transformer处理**：
   - 全局注意力层：第2、5、8、11层
   - 窗口注意力：24×24窗口
   - MLP扩展比：4.625

### 3.2 FPN特征金字塔

**输入**：`[1, 1024, 72, 72]`
**输出**：4个尺度的特征图：
- Level 0: `[1, 256, 288, 288]` (4×上采样)
- Level 1: `[1, 256, 144, 144]` (2×上采样)
- Level 2: `[1, 256, 72, 72]`   (保持)
- Level 3: `[1, 256, 36, 36]`   (2×下采样)

### 3.3 文本编码器

**输入**：文本字符串"a dog"
**处理流程**：
```
"a dog" 
→ 分词: ["<|startoftext|>", "a", "dog", "<|endoftext|>", "<pad>", ...]
→ Token IDs: [1, 32] 形状
→ Token嵌入: [1, 32, 1024]
→ 24层Transformer: [1, 32, 1024]
→ 维度调整: [32, 1, 256]
```

### 3.4 编码器融合层

**输入**：
- 图像特征：`[5184, 1, 256]` (72×72=5184)
- 文本特征：`[32, 1, 256]`
- 几何特征：`[1, 1, 256]` (虚拟提示)

**融合过程**：
1. **文本池化**：对文本特征求平均 → `[1, 256]`
2. **添加到图像特征**：每个图像token添加文本信息
3. **6层交叉注意力**：
   - Query：图像特征 + 位置编码
   - Key/Value：文本+几何提示 (33个token)
4. **输出**：增强的图像特征 `[5184, 1, 256]`

### 3.5 解码器层

**输入**：可学习的查询嵌入 `[200, 256]`
**处理流程**：

```python
# 解码器的6层处理
for layer in range(6):
    # 1. 参考框更新
    reference_boxes = self.bbox_embed(hs[layer])  # [200, 1, 4]
    
    # 2. Box RPB计算（相对位置偏置）
    # 计算每个查询框与特征位置的相对距离
    box_rpb = compute_box_rpb(reference_boxes, feature_positions)
    
    # 3. 三层注意力
    # 自注意力：查询之间
    # 文本交叉注意力：查询←文本特征
    # 图像交叉注意力：查询←图像特征（使用Box RPB）
    
    # 4. 预测生成
    outputs_class = self.dot_product_scoring(hs[layer], text_features)
    outputs_boxes = reference_boxes
```

**解码器输出**：
- 查询特征：`[6, 200, 1, 256]` (6层，每层200个查询)
- 参考框：`[6, 200, 1, 4]`
- Presence logits：`[6, 1, 1]`

### 3.6 分割头

**输入**：
- 查询特征：`[200, 1, 256]` (最后一层)
- 图像特征金字塔：4个尺度特征
- 像素嵌入：`[1, 256, 288, 288]`

**处理**：
1. **像素解码器**：FPN风格上采样，融合多尺度特征
2. **掩码预测**：查询特征 × 像素嵌入
   ```python
   # 计算掩码logits
   mask_preds = torch.einsum("bqc,bchw->bqhw", 
                           mask_embed(queries), 
                           instance_embeds)
   # 输出: [1, 200, 288, 288]
   ```

## 四、各类Token详解

### 4.1 图像Token（Patch Tokens）

| 属性 | 说明 |
|------|------|
| **数量** | 5184个 (72×72 patches) |
| **维度** | 256维 |
| **位置编码** | 正弦位置编码 + RoPE |
| **作用** | 表示图像的局部特征 |

### 4.2 文本Token

| 属性 | 说明 |
|------|------|
| **数量** | 32个 (包括padding) |
| **维度** | 256维 |
| **特殊Token** | `<|startoftext|>`, `<|endoftext|>`, `<pad>` |
| **处理** | 24层Transformer编码 |

### 4.3 几何提示Token

**类型1：点提示Token**
- 编码方式：坐标投影(2→256) + 标签嵌入(正/负)
- 数量：每个点1个token

**类型2：框提示Token**
- 编码方式：ROI池化 + 中心宽高编码
- 数量：每个框1个token

**类型3：虚拟提示Token**
- 当无几何提示时使用
- 形状：`[1, 1, 256]` (CLS token)

### 4.4 查询Token（Object Queries）

| 属性 | 说明 |
|------|------|
| **数量** | 200个 (可学习参数) |
| **维度** | 256维 |
| **类型** | O2O（一对一）和O2M（一对多） |
| **作用** | 表示潜在的物体实例 |

**O2O vs O2M**：
- **O2O查询**：200个，用于推理，每个查询对应一个实例
- **O2M查询**：额外200个，用于训练时的one-to-many监督

### 4.5 Presence Token

**设计目的**：区分"存在/不存在"目标，改进对相似文本的区分

**流程**：
```python
# Presence Token处理
presence_token = self.presence_token.weight  # [1, 256]
presence_token = presence_token.unsqueeze(1)  # [1, 1, 256]

# 在解码器中，presence token与查询token一起处理
combined_tokens = torch.cat([presence_token, queries], dim=0)  # [1+200, 1, 256]

# 预测存在概率
presence_logits = self.presence_head(presence_token)  # [1, 1]
presence_prob = presence_logits.sigmoid()  # 存在概率
```

**作用**：
1. 增强对负样本（无目标）的识别
2. 改进对相似概念（如"白衣服球员" vs "红衣服球员"）的区分

## 五、最终输出详解

### 5.1 输出格式

```python
{
    "masks": Tensor[N, 1, 1024, 1024],      # 二值掩码
    "boxes": Tensor[N, 4],                  # 边界框 (xyxy格式)
    "scores": Tensor[N],                    # 置信度分数
    "original_height": int,                 # 原始图像高度
    "original_width": int,                  # 原始图像宽度
}
```

### 5.2 输出处理流程

1. **置信度计算**：
   ```python
   # 分类分数
   class_probs = outputs_class.sigmoid()  # [1, 200, 1]
   
   # Presence概率
   presence_prob = presence_logits.sigmoid()  # [1, 1]
   
   # 联合概率
   joint_probs = class_probs * presence_prob  # [1, 200]
   ```

2. **阈值过滤**：
   ```python
   keep = joint_probs > confidence_threshold  # 默认0.5
   # 假设保留10个检测
   ```

3. **坐标转换**：
   ```python
   # 归一化框 → 像素坐标
   boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)  # [10, 4]
   scale_fct = torch.tensor([1024, 1024, 1024, 1024])
   boxes = boxes * scale_fct  # [10, 4]
   ```

4. **掩码上采样**：
   ```python
   # 从288×288上采样到1024×1024
   masks = F.interpolate(
       masks_logits, 
       size=(1024, 1024), 
       mode='bilinear'
   )
   masks = masks.sigmoid() > 0.5  # 二值化
   ```

### 5.3 多尺度输出

SAM3支持多尺度输出：
- **掩码分辨率**：288×288 → 上采样到原始尺寸
- **框精度**：高精度边界框，支持迭代细化
- **分数校准**：联合分数（分类分数 × presence概率）

## 六、关键参数总结

| 参数 | 值 | 说明 |
|------|-----|------|
| 图像输入尺寸 | 1008×1008 | 内部处理尺寸 |
| Patch大小 | 14×14 | ViT patch尺寸 |
| 文本长度 | 32 tokens | 包括特殊token |
| 查询数量 | 200 | 可学习的object queries |
| 解码器层数 | 6 | 每层细化预测 |
| 输出实例数 | ≤200 | 根据置信度过滤 |
| 掩码分辨率 | 288×288 | 分割头输出分辨率 |

## 七、视频扩展说明

SAM3的视频模型在图像模型基础上添加：
1. **时间注意力机制**：跨帧特征传播
2. **跟踪查询**：帧间对象关联
3. **时序一致性**：平滑的掩码传播

视频处理流程：
```python
# 视频示例
video_predictor = build_sam3_video_predictor()
response = video_predictor.handle_request({
    "type": "start_session",
    "resource_path": "video.mp4",
})
```

## 八、性能特点

1. **高效性**：单次前向传播处理图像+文本
2. **灵活性**：支持文本、点、框、掩码多种提示
3. **准确性**：在SA-CO基准上达到75-80%的人类性能
4. **可扩展性**：支持批量推理和视频处理

这个完整的流程展示了SAM3如何将多模态输入（图像+文本）转换为精确的分割掩码和检测框，体现了其在通用分割任务上的强大能力。


# SAM3中的 `src token` 详解

在SAM3模型中，**`src token`**（源token）指的是**编码器输入的主要图像特征token**，它们在Transformer编码器中扮演着至关重要的角色。

## 一、`src token` 的定义和来源

### 1.1 定义
**`src token`** 是图像特征经过视觉编码器和FPN处理后，展平为序列形式的特征表示。它们是编码器的主要输入来源（"source"），用于与文本提示进行交叉注意力计算。

### 1.2 生成过程
```python
# 从FPN获取图像特征（多尺度）
backbone_features = backbone_out["backbone_fpn"]
# 选择特定尺度的特征（通常是最后一层）
selected_feature = backbone_features[-1]  # [1, 256, 72, 72]

# 展平为序列格式
batch_size, channels, height, width = selected_feature.shape
src_tokens = selected_feature.flatten(2).permute(2, 0, 1)  # [72*72, 1, 256]
# 结果：src_tokens 形状为 [5184, 1, 256]
```

## 二、`src token` 的作用和功能

### 2.1 主要作用

#### 作用1：作为图像信息的载体
- **存储视觉信息**：每个`src token`对应图像的一个局部区域（14×14 patch）
- **多尺度表示**：虽然主要使用最后一层特征，但编码器可以处理多尺度`src token`

#### 作用2：与文本提示交互的桥梁
- **交叉注意力**：`src token`作为Query，文本token作为Key/Value
- **信息融合**：将文本语义信息注入到视觉特征中

#### 作用3：为解码器提供上下文
- **解码器输入**：编码器输出的`src token`（变为`memory`）作为解码器的Key/Value
- **提供空间上下文**：帮助解码器理解图像的空间结构

### 2.2 在编码器中的处理流程

```python
# Transformer编码器内部处理src token的伪代码
class TransformerEncoderFusion:
    def forward(self, src, prompt, prompt_mask):
        """
        src: [N, B, C] = [5184, 1, 256] (src tokens)
        prompt: [T, B, C] = [33, 1, 256] (文本+几何提示)
        """
        
        # 步骤1：文本池化并添加到src token
        pooled_text = pool_text_features(prompt, prompt_mask)  # [1, 256]
        src = src + pooled_text  # 文本信息注入
        
        # 步骤2：逐层处理
        for layer in self.layers:
            # 自注意力：src token之间的交互
            src = self_attention(src, src, src)
            
            # 交叉注意力：src token ← 文本提示
            src = cross_attention(
                query=src,          # [5184, 1, 256]
                key=prompt,         # [33, 1, 256]
                value=prompt,       # [33, 1, 256]
                key_padding_mask=prompt_mask
            )
            
            # FFN处理
            src = feed_forward_network(src)
        
        return src  # 输出：enhanced src tokens
```

## 三、`src token` 与相关概念的对比

### 3.1 `src token` vs `query token`

| 特征 | `src token` (编码器) | `query token` (解码器) |
|------|---------------------|----------------------|
| **来源** | 图像特征展平 | 可学习的参数 |
| **数量** | 5184 (72×72) | 200 (固定) |
| **作用** | 图像信息载体 | 对象实例表示 |
| **更新方式** | 与文本交叉注意力 | 自注意力+交叉注意力 |
| **输出** | 增强的图像特征 | 对象预测（框、分数、掩码） |

### 3.2 `src token` vs `prompt token`

| 特征 | `src token` | `prompt token` |
|------|------------|----------------|
| **类型** | 图像token | 文本/几何token |
| **数量** | 5184 | 33 (文本32 + 几何1) |
| **角色** | Query (在交叉注意力中) | Key/Value (在交叉注意力中) |
| **信息流** | 接收文本信息 | 提供文本/几何信息 |

## 四、`src token` 在多尺度处理中的变化

### 4.1 多尺度`src token`的配置
```python
# 在编码器中处理多尺度src token
spatial_shapes = [
    [288, 288],  # Level 0: 82944 tokens (288*288)
    [144, 144],  # Level 1: 20736 tokens
    [72, 72],    # Level 2: 5184 tokens (主要使用的)
    [36, 36]     # Level 3: 1296 tokens
]

# 总token数 = 82944 + 20736 + 5184 + 1296 = 110,160
# 但SAM3通常只使用最后一层（5184个token）以降低计算成本
```

### 4.2 位置编码与`src token`
每个`src token`都有对应的**位置编码**，帮助模型理解空间关系：
```python
# 生成位置编码
pos_embed = generate_position_embedding(spatial_shapes[-1])  # [72, 72, 256]
pos_embed_flatten = pos_embed.flatten(0, 1).unsqueeze(1)     # [5184, 1, 256]

# 在注意力计算中添加位置信息
query = src + pos_embed_flatten  # 查询位置编码
key = src + pos_embed_flatten    # 键位置编码
```

## 五、`src token` 在完整流程中的演变

### 5.1 前向传播中的变化
```
原始图像: [1024, 1024, 3]
↓ 预处理
Patch嵌入: [1, 72, 72, 1024]
↓ ViT处理
ViT输出: [1, 1024, 72, 72]
↓ FPN处理
FPN特征: [1, 256, 72, 72] (最后一层)
↓ 展平
src token: [5184, 1, 256]
↓ 编码器处理 (与文本交互)
enhanced src token: [5184, 1, 256] (变为memory)
↓ 解码器交叉注意力
作为Key/Value供解码器使用
```

### 5.2 信息流动示意图
```
图像输入
    ↓
视觉编码器 (ViT+FPN)
    ↓
src token [5184×256] ←─────┐
    ↓                      │
编码器自注意力            │
    ↓                      │
编码器交叉注意力 ──────→ prompt token [33×256]
    ↓                      (文本+几何)
enhanced src token         │
    ↓                      │
解码器交叉注意力 ←─────────┘
    ↓
对象查询 → 预测输出
```

## 六、`src token` 的技术细节

### 6.1 维度设计
- **通道维度**：256维，与文本特征维度对齐
- **序列长度**：5184，平衡表达能力和计算效率
- **batch维度**：支持批量处理，形状为[5184, B, 256]

### 6.2 注意力机制中的角色
```python
# 编码器中的交叉注意力计算
def encoder_cross_attention(src, prompt, prompt_mask):
    """
    src: [N_src, B, C] = [5184, 1, 256]
    prompt: [N_prompt, B, C] = [33, 1, 256]
    """
    
    # 计算注意力权重
    # Query = src + pos_embed (图像特征作为查询)
    # Key = prompt (文本特征作为键)
    # Value = prompt (文本特征作为值)
    
    attn_weights = (src @ prompt.transpose(-2, -1)) / sqrt(dim)
    attn_weights = attn_weights.masked_fill(
        prompt_mask, 
        float('-inf')
    )
    attn_output = attn_weights @ prompt
    
    return attn_output
```

### 6.3 与分割头的连接
```python
# enhanced src token 在分割头中的使用
def segmentation_head_forward(enhanced_src, backbone_feats):
    # 将序列格式的src token恢复为空间格式
    memory_spatial = enhanced_src.permute(1, 2, 0)  # [1, 256, 5184]
    memory_spatial = memory_spatial.view(1, 256, 72, 72)  # [1, 256, 72, 72]
    
    # 替换FPN的最后一层特征
    backbone_feats[-1] = memory_spatial  # 用编码器输出增强图像特征
    
    # 通过像素解码器和掩码预测头
    pixel_embed = pixel_decoder(backbone_feats)  # [1, 256, 288, 288]
    masks = mask_predictor(queries, pixel_embed)  # [1, 200, 288, 288]
    
    return masks
```

## 七、`src token` 的重要性总结

1. **核心表示单元**：`src token`是图像在Transformer中的主要表示形式
2. **跨模态桥梁**：通过交叉注意力连接视觉和语言模态
3. **信息聚合点**：聚集了多尺度视觉信息和文本语义信息
4. **计算效率关键**：合理数量的`src token`（5184）平衡了表达能力和计算成本
5. **可扩展基础**：支持添加更多模态（如音频、深度）的token进行融合

## 八、与其他模型的对比

| 模型 | 类似概念 | 主要差异 |
|------|----------|----------|
| **DETR** | encoder tokens | SAM3添加了文本交叉注意力和多模态融合 |
| **ViT** | patch tokens | SAM3的src token包含多尺度信息和文本增强 |
| **CLIP** | image tokens | SAM3的src token专门用于分割任务，与文本交互更深入 |
| **SAM-1/2** | image embeddings | SAM3的src token加入了更强的文本条件机制 |

**`src token`** 在SAM3中不仅是简单的图像特征表示，更是**多模态信息融合的核心枢纽**。它通过Transformer的注意力机制，将文本语义信息注入到视觉特征中，使得模型能够根据文本提示准确地定位和分割目标对象。这种设计是SAM3实现开放式词汇分割能力的关键所在。
