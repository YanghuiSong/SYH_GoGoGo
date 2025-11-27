# SAM系列掩码生成算法原理详解

## 1. SAM掩码生成算法

### 核心公式框架
```
掩码 = MaskDecoder(ImageEncoder(I), PromptEncoder(P))
```

### 详细算法流程

#### 1.1 图像编码
```python
# 输入：图像 I ∈ ℝ^(3×1024×1024)
# 输出：图像嵌入 E_img ∈ ℝ^(256×64×64)

E_img = ViT_MAE(I)                    # Vision Transformer编码
E_img = Conv1x1(E_img)               # 降维到256通道
E_img = Conv3x3(E_img)               # 特征融合
```

#### 1.2 提示编码
```python
# 点提示编码
def encode_point(point, is_foreground):
    pos_enc = positional_encoding(point.coords)  # 位置编码
    type_emb = learnable_embedding(is_foreground) # 类型嵌入
    return pos_enc + type_emb

# 框提示编码  
def encode_box(box):
    tl_emb = positional_encoding(box.top_left) + learnable_embedding("top_left")
    br_emb = positional_encoding(box.bottom_right) + learnable_embedding("bottom_right")
    return [tl_emb, br_emb]

# 掩码提示编码
def encode_mask(mask):
    mask_low_res = downsample(mask, 4)           # 4倍下采样
    conv1 = Conv2x2_stride2(mask_low_res)       # 输出4通道
    conv2 = Conv2x2_stride2(conv1)              # 输出16通道  
    final = Conv1x1(conv2)                      # 输出256通道
    return E_img + final                        # 与图像嵌入相加
```

#### 1.3 掩码解码器（核心算法）

**初始化**：
```python
tokens = concat(prompt_tokens, output_token)    # 提示令牌 + 输出令牌
image_embedding = E_img                         # 64×64×256
```

**两层Transformer解码**：
```python
for layer in range(2):
    # 1. 令牌自注意力
    tokens_self = MultiHeadAttention(
        Q=tokens, K=tokens, V=tokens,
        dim=256, heads=8, dim_head=32
    )
    
    # 2. 令牌→图像交叉注意力
    image_cross1 = MultiHeadAttention(
        Q=image_embedding, K=tokens_self, V=tokens_self,
        dim=256, heads=8, dim_head=32
    )
    
    # 3. MLP更新令牌
    tokens_mlp = MLP(tokens_self, hidden_dim=2048)
    
    # 4. 图像→令牌交叉注意力
    tokens_final = MultiHeadAttention(
        Q=tokens_mlp, K=image_cross1, V=image_cross1, 
        dim=256, heads=8, dim_head=32
    )
    
    tokens = tokens_final
    image_embedding = image_cross1
```

**上采样与掩码预测**：
```python
# 上采样图像嵌入
upsampled_embed = TransposeConv2x2_stride2(image_embedding)  # 128×128×64
upsampled_embed = TransposeConv2x2_stride2(upsampled_embed)  # 256×256×32

# 动态掩码预测
mask_weights = MLP(output_token, [256, 32])     # 输出令牌→线性权重
mask_logits = einsum('c h w, c -> h w', upsampled_embed, mask_weights)
mask = sigmoid(mask_logits)                     # 最终概率图
```

#### 1.4 模糊性处理
```python
# 预测3个掩码对应不同层次
masks = [mask_decoder(E_img, P) for _ in range(3)]
confidences = [iou_head(mask) for mask in masks]

# 训练时选择最佳掩码
loss = min([focal_dice_loss(mask, gt_mask) for mask in masks])
```

---

## 2. SAM2掩码生成算法（视频扩展）

### 核心公式框架
```
对于每个帧t:
    E_cond = MemoryAttention(ImageEncoder(I_t), MemoryBank)
    掩码_t = MaskDecoder(E_cond, PromptEncoder(P_t))
    MemoryBank.update(MemoryEncoder(掩码_t, E_img_t))
```

### 详细算法流程

#### 2.1 流式处理框架
```python
class SAM2:
    def process_frame(self, frame_t, prompt_t=None):
        # 1. 图像编码
        E_img_t = Hiera_Encoder(frame_t)        # 分层编码器
        
        # 2. 内存注意力条件化
        E_cond_t = self.memory_attention(E_img_t, self.memory_bank)
        
        # 3. 掩码解码
        if prompt_t is not None:
            prompt_emb = self.prompt_encoder(prompt_t)
            mask_t = self.mask_decoder(E_cond_t, prompt_emb)
        else:
            mask_t = self.mask_decoder(E_cond_t, None)
            
        # 4. 更新内存
        memory_t = self.memory_encoder(mask_t, E_img_t)
        self.memory_bank.push(memory_t, frame_idx=t)
        
        return mask_t
```

#### 2.2 内存注意力机制
```python
def memory_attention(current_embed, memory_bank):
    # 内存银行包含：
    # - 最近N帧的空间记忆: M_spatial ∈ ℝ^(N×H×W×C)  
    # - 提示帧记忆: M_prompt ∈ ℝ^(M×H×W×C)
    # - 对象指针: O_pointers ∈ ℝ^(K×D)
    
    memories = concat(M_spatial, M_prompt, O_pointers)
    
    # L层Transformer块
    x = current_embed
    for layer in range(L):
        # 自注意力
        x_self = MultiHeadAttention(Q=x, K=x, V=x)
        
        # 内存交叉注意力
        x_mem = MultiHeadAttention(Q=x_self, K=memories, V=memories)
        
        # MLP
        x = MLP(x_mem)
        
    return x
```

#### 2.3 内存编码器
```python
def memory_encoder(mask_pred, image_embed):
    # 下采样掩码
    mask_down = Conv2x2_stride2(mask_pred)      # 32×32
    mask_down = Conv2x2_stride2(mask_down)      # 16×16
    
    # 与图像嵌入融合
    memory = image_embed + mask_down
    
    # 轻量卷积融合
    memory = Conv3x3(memory)
    
    return memory
```

#### 2.4 时序一致性处理
```python
# 时间位置编码
def add_temporal_encoding(memories, frame_indices):
    for i, idx in enumerate(frame_indices):
        pos_enc = sinusoidal_encoding(idx)      # 正弦位置编码
        memories[i] = memories[i] + pos_enc
    return memories
```

---

## 3. SAM3掩码生成算法（概念扩展）

### 核心公式框架
```
# 检测阶段
检测结果 = ConceptDetector(I, text_prompt, exemplars)
# 跟踪阶段（视频）
掩码序列 = VideoTracker(视频, 检测结果, MemoryBank)
```

### 详细算法流程

#### 3.1 概念检测器

**架构**：
```python
class ConceptDetector:
    def __init__(self):
        self.image_encoder = PerceptionEncoder()  # 对齐的视觉编码器
        self.text_encoder = CLIP_TextEncoder()    # 文本编码器
        self.exemplar_encoder = ExemplarEncoder() # 范例编码器
        self.fusion_encoder = TransformerFusion() # 融合编码器
        self.detr_decoder = DETR_Decoder()        # DETR风格解码器
```

**检测流程**：
```python
def detect(self, image, text_prompt, exemplars=None):
    # 1. 编码输入
    E_img = self.image_encoder(image)
    E_text = self.text_encoder(text_prompt)
    
    if exemplars:
        E_exemplar = self.exemplar_encoder(exemplars)
        prompt_tokens = concat(E_text, E_exemplar)
    else:
        prompt_tokens = E_text
        
    # 2. 融合编码
    E_fused = self.fusion_encoder(E_img, prompt_tokens)
    
    # 3. DETR解码
    object_queries = learnable_queries(N=100)    # 100个对象查询
    detections = self.detr_decoder(object_queries, E_fused)
    
    # 4. 存在令牌解耦
    presence_score = self.presence_head(E_fused)  # 全局存在概率
    detection_scores = self.detection_heads(detections)  # 检测分数
    
    final_scores = presence_score * detection_scores  # 最终分数
    
    return final_scores, detections
```

#### 3.2 存在令牌机制（关键创新）

```python
# 数学公式表示
p(检测有效) = p(概念存在|图像,文本) × p(查询匹配|概念存在)

# 代码实现
class PresenceHead(nn.Module):
    def __init__(self, dim):
        self.presence_token = nn.Parameter(torch.randn(1, dim))
        self.mlp = MLP(dim, [dim//2, 1])
        
    def forward(self, fused_embed):
        # 全局存在性判断
        presence_feat = cross_attention(self.presence_token, fused_embed)
        presence_logit = self.mlp(presence_feat)
        return sigmoid(presence_logit)
```

#### 3.3 视频跟踪算法

```python
def video_tracking(video_frames, initial_detections, text_prompt):
    memory_bank = MemoryBank()
    masklets = []
    
    for t, frame in enumerate(video_frames):
        # 1. 检测器预测新对象
        new_detections = detector(frame, text_prompt)
        
        # 2. 跟踪器传播已有掩码
        if t > 0:
            propagated_masks = tracker.propagate(masklets[t-1], memory_bank)
        else:
            propagated_masks = []
            
        # 3. 匹配与更新
        current_masks = match_and_update(propagated_masks, new_detections)
        masklets.append(current_masks)
        
        # 4. 更新内存
        memory_bank.update(current_masks, frame_features[t])
        
    return masklets
```

#### 3.4 匹配与更新策略

```python
def match_and_update(propagated, detections):
    # IoU匹配
    iou_matrix = pairwise_iou(propagated, detections)
    matches = hungarian_matching(iou_matrix)
    
    updated_masks = []
    
    # 处理匹配对
    for prop_idx, det_idx in matches:
        if iou_matrix[prop_idx, det_idx] > threshold:
            # 高质量匹配：使用检测结果
            updated_masks.append(detections[det_idx])
        else:
            # 低质量匹配：使用传播结果
            updated_masks.append(propagated[prop_idx])
            
    # 处理未匹配的检测（新对象）
    for i, det in enumerate(detections):
        if i not in [pair[1] for pair in matches]:
            updated_masks.append(det)
            
    return updated_masks
```

---

## 4. 三代算法对比总结

| 特性 | SAM | SAM2 | SAM3 |
|------|-----|------|------|
| **输入类型** | 几何提示 | 几何提示+时序 | 概念提示+几何提示 |
| **输出规模** | 单实例 | 单实例视频 | 多实例图像/视频 |
| **核心机制** | 交叉注意力 | 内存注意力 | 存在令牌+检测跟踪 |
| **处理模糊性** | 多掩码输出 | 时序消歧 | 概念消歧 |
| **数学复杂度** | O(HW) | O(T×HW) | O(N×HW) |

其中：
- H,W: 空间维度
- T: 时间帧数  
- N: 实例数量

三代算法的演进体现了从特定提示到通用概念的理解，从静态图像到动态视频的扩展，从单实例到多实例的规模化处理能力提升。

---

# SAM系列掩码生成算法公式详解

## 1. SAM掩码生成算法

### 1.1 图像编码过程
```math
$$
\mathbf{E}_{\text{img}} = \text{ViT-MAE}(\mathbf{I}) \in \mathbb{R}^{256 \times 64 \times 64}
$$
```

其中：
```math

- $\mathbf{I} \in \mathbb{R}^{3 \times 1024 \times 1024}$ 为输入图像
```
```math
- $\mathbf{E}_{\text{img}}$ 为图像嵌入特征图
```
### 1.2 提示编码公式

#### 点提示编码：
```math
$$
\mathbf{E}_{\text{point}} = \text{PE}(x, y) + \mathbf{E}_{\text{type}}
$$
```
其中：
```math
- $\text{PE}(x, y)$ 为正弦位置编码函数
```
```math
- $\mathbf{E}_{\text{type}} \in \{\mathbf{E}_{\text{fg}}, \mathbf{E}_{\text{bg}}\}$ 为前景/背景类型嵌入
```
#### 框提示编码：
```math
$$
\mathbf{E}_{\text{box}} = [\text{PE}(x_1, y_1) + \mathbf{E}_{\text{tl}},\ \text{PE}(x_2, y_2) + \mathbf{E}_{\text{br}}]
$$
```
#### 掩码提示编码：
```math
$$
\mathbf{E}_{\text{mask}} = \text{Conv}(\mathbf{E}_{\text{img}} \oplus \text{Downsample}(\mathbf{M}, 4))
$$
```
### 1.3 掩码解码器核心公式

#### 令牌序列初始化：
```math
$$
\mathbf{T}^{(0)} = [\mathbf{P}_1, \mathbf{P}_2, \dots, \mathbf{P}_k, \mathbf{O}] \in \mathbb{R}^{(k+1) \times 256}
$$
```
#### 两层Transformer解码过程：

对于 $l = 1, 2$：

**自注意力层**：
```math
$$
\mathbf{T}_{\text{self}}^{(l)} = \text{MHA}\left(\mathbf{Q} = \mathbf{T}^{(l-1)},\ \mathbf{K} = \mathbf{T}^{(l-1)},\ \mathbf{V} = \mathbf{T}^{(l-1)}\right)
$$
```
**交叉注意力（提示→图像）**：
```math
$$
\mathbf{E}_{\text{cross}}^{(l)} = \text{MHA}\left(\mathbf{Q} = \mathbf{E}_{\text{img}},\ \mathbf{K} = \mathbf{T}_{\text{self}}^{(l)},\ \mathbf{V} = \mathbf{T}_{\text{self}}^{(l)}\right)
$$
```
**MLP更新**：
```math
$$
\mathbf{T}_{\text{mlp}}^{(l)} = \text{MLP}(\mathbf{T}_{\text{self}}^{(l)})
$$
```
**交叉注意力（图像→提示）**：
```math
$$
\mathbf{T}^{(l)} = \text{MHA}\left(\mathbf{Q} = \mathbf{T}_{\text{mlp}}^{(l)},\ \mathbf{K} = \mathbf{E}_{\text{cross}}^{(l)},\ \mathbf{V} = \mathbf{E}_{\text{cross}}^{(l)}\right)
$$
```
#### 掩码预测公式：

**上采样过程**：
```math
$$
\mathbf{E}_{\text{up}} = \text{TransposeConv}^2(\mathbf{E}_{\text{img}}^{(2)}) \in \mathbb{R}^{256 \times 256 \times 32}
$$
```
**动态权重生成**：
```math
$$
\mathbf{W}_{\text{mask}} = \text{MLP}(\mathbf{O}) \in \mathbb{R}^{32}
$$
```
**掩码logits计算**：
```math
$$
\mathbf{M}_{\text{logits}} = \sum_{c=1}^{32} \mathbf{E}_{\text{up}}[c] \cdot \mathbf{W}_{\text{mask}}[c]
$$
```
**最终掩码概率**：
```math
$$
\mathbf{M} = \sigma(\mathbf{M}_{\text{logits}}) \in [0, 1]^{256 \times 256}
$$
```
### 1.4 模糊性处理机制

#### 多掩码预测：
```math
$$
\mathbf{M}_i = \text{MaskDecoder}(\mathbf{E}_{\text{img}}, \mathbf{P}),\quad i = 1, 2, 3
$$
```
#### 置信度预测：
```math
$$
c_i = \text{IoU-Head}(\mathbf{M}_i)
$$
```
#### 训练损失：
```math
$$
\mathcal{L} = \min_{i=1,2,3} \left[\mathcal{L}_{\text{focal}}(\mathbf{M}_i, \mathbf{M}_{\text{gt}}) + \mathcal{L}_{\text{dice}}(\mathbf{M}_i, \mathbf{M}_{\text{gt}})\right]
$$
```
---

## 2. SAM2视频掩码生成算法

### 2.1 流式处理框架

对于 $t = 1, \dots, T$：
```math
$$
\begin{aligned}
\mathbf{E}_{\text{img}}^t &= \text{ImageEncoder}(\mathbf{I}_t) \\
\mathbf{E}_{\text{cond}}^t &= \text{MemoryAttention}(\mathbf{E}_{\text{img}}^t, \mathcal{M}_t) \\
\mathbf{M}_t &= \text{MaskDecoder}(\mathbf{E}_{\text{cond}}^t, \mathbf{P}_t) \\
\mathcal{M}_{t+1} &= \text{MemoryEncoder}(\mathbf{M}_t, \mathbf{E}_{\text{img}}^t)
\end{aligned}
$$
```
### 2.2 内存注意力机制

#### 内存银行构成：
```math
$$
\mathcal{M}_t = \{\mathcal{M}_{\text{spatial}},\ \mathcal{M}_{\text{prompt}},\ \mathcal{O}_{\text{pointers}}\}
$$
```
其中：
```math
- $\mathcal{M}_{\text{spatial}} \in \mathbb{R}^{N \times H \times W \times C}$：最近N帧空间记忆
```
```math
- $\mathcal{M}_{\text{prompt}} \in \mathbb{R}^{M \times H \times W \times C}$：提示帧记忆
```
```math
- $\mathcal{O}_{\text{pointers}} \in \mathbb{R}^{K \times D}$：对象指针
```
#### 内存注意力公式：

对于 $l = 1, \dots, L$：
```math
$$
\begin{aligned}
\mathbf{E}_{\text{self}}^{(l)} &= \text{MHA}\left(\mathbf{Q} = \mathbf{E}^{(l-1)},\ \mathbf{K} = \mathbf{E}^{(l-1)},\ \mathbf{V} = \mathbf{E}^{(l-1)}\right) \\
\mathbf{E}_{\text{mem}}^{(l)} &= \text{MHA}\left(\mathbf{Q} = \mathbf{E}_{\text{self}}^{(l)},\ \mathbf{K} = \mathcal{M}_t,\ \mathbf{V} = \mathcal{M}_t\right) \\
\mathbf{E}^{(l)} &= \text{MLP}(\mathbf{E}_{\text{mem}}^{(l)})
\end{aligned}
$$
```
### 2.3 时间位置编码

#### 正弦位置编码：
```math
$$
\begin{aligned}
\text{PE}(pos, 2i) &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
\text{PE}(pos, 2i+1) &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$
```
#### 内存时间编码：
```math
$$
\mathcal{M}_{\text{spatial}}[i] = \mathcal{M}_{\text{spatial}}[i] + \text{PE}(t_i)
$$
```
---

## 3. SAM3概念掩码生成算法

### 3.1 检测器架构公式

#### 输入编码：
```math
$$
\begin{aligned}
\mathbf{E}_{\text{img}} &= \text{PerceptionEncoder}(\mathbf{I}) \\
\mathbf{E}_{\text{text}} &= \text{CLIP-Encoder}(\mathbf{T}) \\
\mathbf{E}_{\text{exemplar}} &= \text{ExemplarEncoder}(\mathbf{B})
\end{aligned}
$$
```
#### 融合编码：
```math
$$
\mathbf{E}_{\text{fused}} = \text{FusionEncoder}(\mathbf{E}_{\text{img}},\ [\mathbf{E}_{\text{text}};\ \mathbf{E}_{\text{exemplar}}])
$$
```
#### DETR风格解码：
```math
设 $\mathbf{Q} = \{\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_N\}$ 为可学习对象查询

对于 $l = 1, \dots, L$：
$$
\mathbf{Q}^{(l)} = \text{MHA}\left(\mathbf{Q}^{(l-1)},\ \mathbf{E}_{\text{fused}},\ \mathbf{E}_{\text{fused}}\right)
$$
```
### 3.2 存在令牌机制（核心创新）

#### 解耦识别与定位：
```math
$$
p(\text{detection}) = p(\text{presence} \mid \mathbf{I}, \mathbf{T}) \times p(\text{query\_match} \mid \text{presence})
$$
```
#### 数学表达：
```math
$$
\begin{aligned}
s_{\text{presence}} &= \sigma\left(\text{MLP}\left(\text{CrossAttention}(\mathbf{p}_{\text{token}}, \mathbf{E}_{\text{fused}})\right)\right) \\
s_{\text{detection}}^i &= \text{DetectionHead}(\mathbf{q}_i, \mathbf{E}_{\text{fused}}) \\
s_{\text{final}}^i &= s_{\text{presence}} \times s_{\text{detection}}^i
\end{aligned}
$$
```
### 3.3 视频跟踪公式

#### 检测-跟踪框架：

对于 $t = 1, \dots, T$：

**检测新对象**：
```math
$$
\mathcal{D}_t = \text{Detector}(\mathbf{I}_t, \mathbf{T})
$$
```
**传播已有掩码**：
```math
$$
\mathcal{M}_{\text{prop}}^t = 
\begin{cases} 
\text{Tracker}(\mathcal{M}^{t-1}, \mathcal{M}^{t-1}) & \text{if } t > 1 \\
\emptyset & \text{if } t = 1
\end{cases}
$$
```
**匹配与更新**：
```math
$$
\mathcal{M}^t = \text{Match\_Update}(\mathcal{M}_{\text{prop}}^t, \mathcal{D}_t)
$$
```
#### 匹配函数：

**IoU矩阵计算**：
```math
$$
\text{IoU}_{ij} = \frac{\mathcal{M}_{\text{prop}}^i \cap \mathcal{D}^j}{\mathcal{M}_{\text{prop}}^i \cup \mathcal{D}^j}
$$
```
**匈牙利匹配**：
```math
$$
\text{matches} = \text{Hungarian}(\text{IoU}_{\text{matrix}})
$$
```
**掩码更新**：
```math
$$
\mathcal{M}^t = \{\text{Merge}(\mathcal{M}_{\text{prop}}^i, \mathcal{D}^j) \mid (i,j) \in \text{matches}\} \cup \{\text{unmatched detections}\}
$$
```
### 3.4 损失函数设计

#### 检测损失：
```math
$$
\mathcal{L}_{\text{det}} = \mathcal{L}_{\text{presence}} + \lambda_{\text{cls}}\mathcal{L}_{\text{cls}} + \lambda_{\text{box}}\mathcal{L}_{\text{box}} + \lambda_{\text{mask}}\mathcal{L}_{\text{mask}}
$$
```
#### 跟踪损失：
```math
$$
\mathcal{L}_{\text{track}} = \sum_{t=1}^T \left[\mathcal{L}_{\text{mask}}(\mathcal{M}^t, \mathcal{M}_{\text{gt}}^t) + \lambda_{\text{assoc}}\mathcal{L}_{\text{association}}\right]
$$
```
---

## 4. 关键算法原理对比

### 4.1 注意力机制演进

#### SAM（图像级交叉注意力）：
```math
$$
\text{Attention} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{img}}\mathbf{K}_{\text{prompt}}^\top}{\sqrt{d}}\right)\mathbf{V}_{\text{prompt}}
$$
```
#### SAM2（时空内存注意力）：
```math
$$
\text{Attention} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{current}}[\mathbf{K}_{\text{memory}};\mathbf{K}_{\text{prompt}}]^\top}{\sqrt{d}}\right)[\mathbf{V}_{\text{memory}};\mathbf{V}_{\text{prompt}}]
$$
```
#### SAM3（概念条件注意力）：
```math
$$
\text{Attention} = \text{softmax}\left(\frac{\mathbf{Q}_{\text{detector}}[\mathbf{K}_{\text{text}};\mathbf{K}_{\text{exemplar}}]^\top}{\sqrt{d}}\right)[\mathbf{V}_{\text{text}};\mathbf{V}_{\text{exemplar}}]
$$
```
### 4.2 掩码生成复杂度分析

| 模型 | 空间复杂度 | 时间复杂度 | 实例维度 |
|------|------------|------------|----------|
| SAM | $\mathcal{O}(HWC)$ | $\mathcal{O}(1)$ | 单实例 |
| SAM2 | $\mathcal{O}(HWC)$ | $\mathcal{O}(T)$ | 单实例 |
| SAM3 | $\mathcal{O}(HWC)$ | $\mathcal{O}(T)$ | $\mathcal{O}(N)$ |

其中：
```math
- $H,W$：空间维度（如64×64）
```
```math
- $C$：通道维度（如256）
```
```math
- $T$：时间帧数
```
```math
- $N$：实例数量
```
### 4.3 核心数学创新总结

#### SAM：
```math
$$
\mathbf{M} = \text{Decoder}(\text{Encoder}(\mathbf{I}) \oplus \text{Encoder}(\mathbf{P}))
$$
```
**创新点**：多尺度特征融合 + 模糊性建模

#### SAM2：
```math
$$
\mathbf{M}_t = f(\mathbf{M}_{t-1}, \mathbf{I}_t, \mathbf{P}_t, \mathcal{M})
$$
```
**创新点**：流式记忆传播 + 时空一致性

#### SAM3：
```math
$$
\{\mathbf{M}_i\} = \text{Detector}(\mathbf{I}, \text{Concept}) + \text{Tracker}(\{\mathbf{M}_i\}, \text{Video})
$$
```
**创新点**：概念条件检测 + 存在令牌解耦 + 多实例跟踪

这些公式完整地描述了SAM系列从几何提示到概念理解，从静态图像到动态视频，从单实例到多实例的技术演进路径。
