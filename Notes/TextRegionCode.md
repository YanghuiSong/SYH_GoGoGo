# TextRegion：文本对齐区域令牌生成框架详解

## 1. 项目概述

TextRegion是一个无需训练（training-free）的框架，它通过结合冻结的图像-文本模型（如CLIP、SigLIP2、Perception Encoder）与来自SAM2的分割掩码，生成**文本对齐的区域令牌**（text-aligned region tokens）。这种创新方法使模型在没有任何专门训练的情况下，就能在开放世界语义分割、指代表达理解和多对象定位等任务上取得优异表现。

## 2. 核心思想与动机

传统图像-文本模型（如CLIP）擅长理解整张图像与文本之间的关系，但对于图像中特定区域的理解能力有限。另一方面，分割模型（如SAM2）能很好地识别图像中的不同区域，但缺乏语义理解能力。TextRegion的核心思想是将这两种能力结合起来：

- 利用SAM2提供精确的空间区域划分
- 利用CLIP等模型提供丰富的语义理解
- 通过掩码引导的注意力池化机制，使每个区域都具有与文本空间对齐的特征表示

## 3. 详细实现流程

### 步骤1：生成区域掩码（Mask Generation）

#### 3.1 输入预处理
在[TextRegionSegmenter.py](file:///d:/SYH/CodeReading/TextRegion/TextRegionSegmenter.py)中，输入图像经过预处理：

```python
# 加载并调整图像尺寸
img_arr = Image.open(args.image_dir).convert("RGB")
img_arr = np.array(img_arr)

if self.resize_method == 'multi_resolution':
    img_arr = imrescale(img_arr, (args.scale[0], args.scale[1]), return_scale=False, interpolation='bilinear')
else:
    img_arr = cv2.resize(img_arr, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)

# 转换为tensor格式
img_tensor = torch.from_numpy(img_arr).to(device="cuda", dtype=torch.float32)
image_tensor_for_sam2 = torch.stack([img_tensor])
image_tensor_for_sam2 = self.sam_transform(image_tensor_for_sam2)
```

#### 3.2 SAM2区域分割
使用定制版的SAM2掩码生成器：

```python
# 使用CustomAutomaticMaskGenerator生成掩码
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32):
    sam2_masks = self.sam2_generator.generate_for_batch(image_tensor_for_sam2, [ori_shape], None)
    
# 提取分割掩码
unique_masks = torch.stack([mask['segmentations'] for mask in sam2_masks[0]])
unique_masks = unique_masks.to(self.device, dtype=self.dtype)
```

#### 3.3 掩码尺寸调整
为了与图像特征图对齐，需要调整掩码尺寸：

```python
# 调整掩码到特征图尺寸
unique_low_res_masks = F.interpolate(unique_masks.unsqueeze(0), [self.points_per_h, self.points_per_w], mode="bilinear")
unique_low_res_masks = unique_low_res_masks.reshape(-1, self.points_per_h * self.points_per_w)
unique_low_res_masks = torch.clamp(unique_low_res_masks, min=0, max=1)  # 确保值在[0,1]范围内
```

**重要说明**：这里的掩码是"软掩码"，即每个像素的值在0到1之间，表示该像素属于某个区域的置信度，而不是二进制的硬分割。

### 步骤2：提取图像特征（Patch Encoding）

#### 2.1 多模型支持
TextRegion支持多种图像-文本模型，每种模型的特征提取方式略有不同：

```python
# 根据模型类型选择相应的特征提取方法
if self.clip_pretrained == 'meta':  # Perception Encoder
    clip_inputs = clip_inputs.to(self.device, dtype=self.clip.visual.proj.dtype)
    pe_last_blk_value, pe_last_blk = self.clip.encode_image(clip_inputs, return_value=True, region_attn_mask=None)
elif self.clip_pretrained == 'siglip2':  # SigLIP2
    siglip_last_blk_value, intermediates = self.clip.visual.trunk.forward_intermediates(clip_inputs)
    siglip_last_blk = self.clip.visual.trunk.attn_pool
else:  # 标准CLIP
    clip_inputs = clip_inputs.to(self.device, dtype=self.clip.visual.proj.dtype)
    clip_last_blk_value, clip_last_blk = self.clip.encode_image(clip_inputs, return_value=True)
```

#### 2.2 特征提取细节
对于不同模型，特征提取的具体实现：

- **CLIP模型**：提取图像编码器最后一层的value特征
- **SigLIP2模型**：通过中间层前向传播获取特征
- **Perception Encoder**：直接获取编码器输出

这些特征具有良好的语义信息，可以直接与文本嵌入进行对齐。

### 步骤3：掩码引导的注意力池化（Mask-based Attention Pooling）

这是TextRegion的核心创新，下面详细介绍三种模型的实现方式：

#### 3.1 SigLIP2模型的实现

```python
def siglip_value_with_sam2_attn(self, args, low_res_mask_with_pad, last_blk_value, attn_blk):
    bsz, _, embed_dim = last_blk_value.shape
    
    # 如果使用多分辨率方法，需要调整特征尺寸
    if self.resize_method == 'multi_resolution':
        patch_num = self.crop_size // self.patch_size
        x_ori = last_blk_value.permute(0, 2, 1).contiguous().view(bsz, embed_dim, patch_num, patch_num)
        
        # 将多个裁剪后的图像拼接成一个多分辨率特征图
        # ... 多分辨率处理逻辑 ...
        
        x_input = x_multi_reso.contiguous().view(1, embed_dim, self.crop_num_h * self.crop_num_w * patch_num ** 2).permute(0, 2, 1)
    else:
        x_input = last_blk_value

    # 全局补丁过滤：移除与任何区域都不相关的补丁
    if args.remove_global_patch:
        keep_masks = torch.sum(low_res_mask_with_pad, dim=1) > 0
        low_res_mask = low_res_mask_with_pad[keep_masks]
        
        # 计算补丁间的相似性，判断哪些补丁应该被移除
        patch_norm = x_input.norm(dim=-1, keepdim=True)
        patch_features = (x_input / patch_norm)[0]
        patch_similarity = (patch_features @ patch_features.T).float()
        
        # 计算补丁与区域的相似性
        patch_2_region = patch_similarity @ (low_res_mask > 0).float().T
        patch_2_region_avg = patch_2_region / (low_res_mask > 0).sum(dim=-1)
        
        # 计算补丁在区域内与区域外的平均相似性差异
        blong_score = patch_2_region_avg * (low_res_mask > 0).float().T
        blong_score_avg = blong_score.sum(dim=-1) / ((low_res_mask > 0).sum(dim=0) + 1e-9)
        
        outside_score = patch_2_region_avg * (low_res_mask == 0).float().T
        outside_score_avg = outside_score.sum(dim=-1) / ((low_res_mask == 0).sum(dim=0) + 1e-9)
        
        difference_score = (blong_score_avg - outside_score_avg).cpu().float().numpy()
        
        # 根据阈值过滤补丁
        low_res_mask_with_pad[:, difference_score < self.global_patch_threshold] = 0

    keep_masks = torch.sum(low_res_mask_with_pad, dim=1) > 0
    low_res_mask_with_pad = low_res_mask_with_pad[keep_masks]
    low_res_mask_with_pad = torch.clamp(low_res_mask_with_pad, min=0, max=1)
    
    region_num = low_res_mask_with_pad.shape[0]

    # 执行掩码引导的注意力池化
    _, N, C = x_input.shape
    q_latent = attn_blk.latent.expand(region_num, -1, -1)
    q = attn_blk.q(q_latent).reshape(region_num, attn_blk.latent_len, attn_blk.num_heads, attn_blk.head_dim).transpose(1, 2)

    x = x_input.expand(region_num, -1, -1)
    kv = attn_blk.kv(x).reshape(region_num, N, 2, attn_blk.num_heads, attn_blk.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
    q, k = attn_blk.q_norm(q), attn_blk.k_norm(k)

    # 关键：使用掩码约束注意力权重
    attn_mask = low_res_mask_with_pad.unsqueeze(1).unsqueeze(1).repeat(1, attn_blk.num_heads, 1, 1)
    
    # 对键进行平均池化
    k = attn_blk.k_norm(k.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True))
    k = k.repeat(1, 1, v.shape[-2], v.shape[-1])
    
    # 计算带掩码的注意力
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask > 0)

    # 后处理
    x = x.transpose(1, 2).reshape(region_num, attn_blk.latent_len, C)
    x = attn_blk.proj(x)
    x = attn_blk.proj_drop(x)

    x = self.clip.visual.trunk.fc_norm(x)
    x = self.clip.visual.trunk.head_drop(x)

    region_features = x.permute(1, 0, 2)
    region_features /= region_features.norm(dim=-1, keepdim=True)
    return region_features, keep_masks
```

#### 3.2 Perception Encoder模型的实现

```python
def pe_value_with_sam2_attn(self, args, unique_low_res_masks, last_blk_value, blk):
    # 移除CLS标记（如果存在）
    if self.clip.visual.use_cls_token:
        last_blk_value = last_blk_value[:, 1:]
    
    # 多分辨率处理（与SigLIP2类似）
    # ...
    
    # 全局补丁过滤（与SigLIP2类似）
    # ...
    
    # 关键：使用probe机制进行区域特征提取
    q = blk.probe.repeat((batch, 1, 1)).to(x.dtype)  # 查询向量
    k = blk.layernorm(x.mean(dim=-2, keepdim=True))  # 键向量
    k = k.repeat(1, x.shape[-2], 1).to(x.dtype)     # 扩展键向量
    
    # 带掩码的注意力计算
    x = blk.attn(q, k, x, need_weights=False, key_padding_mask=unique_low_res_masks<=0)[0]
    
    # 投影到最终空间
    with torch.no_grad():
        region_features = x @ self.clip.visual.proj
    region_features = F.normalize(region_features, dim=-1)
    return region_features, keep_masks
```

#### 3.3 标准CLIP模型的实现

```python
def clip_value_with_sam2_attn(self, args, unique_low_res_masks, clip_v, blk):
    attn_layer = blk.attn
    num_heads = attn_layer.num_heads
    _, bsz, embed_dim = clip_v.size()
    head_dim = embed_dim // num_heads

    # 标准的多头注意力计算
    x = blk.ln_1(clip_v)
    q, k, v_ori = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)

    # 多分辨率处理（与前面类似）
    # ...

    # 全局补丁过滤（与前面类似）
    # ...

    # 关键：使用掩码约束注意力权重
    attn_weights = unique_low_res_masks.unsqueeze(0).repeat(num_heads, 1, 1)
    attn_weights = attn_weights.to(dtype=v_multi_head.dtype)

    # 应用掩码进行注意力计算
    attn_output = torch.bmm(attn_weights, v_multi_head)
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    attn_output = attn_layer.out_proj(attn_output)
    attn_output += blk.mlp(blk.ln_2(attn_output))
    
    region_features = attn_output.permute(1, 0, 2)  # LND -> NLD

    # 最终投影和归一化
    region_features = self.clip.visual.ln_post(region_features) @ self.clip.visual.proj
    region_features /= region_features.norm(dim=-1, keepdim=True)
    return region_features, keep_masks
```

### 步骤4：生成区域令牌（Region Token）

无论使用哪种模型，最终都会生成归一化的区域特征向量：

```python
# 在各个方法的最后部分
region_features /= region_features.norm(dim=-1, keepdim=True)  # L2归一化
```

这确保了区域特征与文本特征在相同的嵌入空间中，可以直接计算相似度。

### 步骤5：应用于下游任务

#### 5.1 区域分类
```python
# 计算区域令牌与查询词嵌入的相似度
if self.clip_pretrained == 'siglip2':
    logits_per_text = (
            torch.matmul(self.query_features, region_features[0].t()) * self.clip.logit_scale.exp()
            + self.clip.logit_bias
    )
    region_logits = logits_per_text.t()
else:
    region_logits = region_features[0] @ self.query_features.T
```

#### 5.2 像素级分割
```python
def postprocess_result(self, region_logits, unique_masks, ori_shape):
    unique_masks = torch.clamp(unique_masks, min=0, max=1)
    
    # 将区域分类结果广播回原始分辨率
    seg_logits = region_logits.unsqueeze(-1).unsqueeze(-1) * unique_masks.unsqueeze(1)
    seg_logits = seg_logits.sum(0, keepdim=True)

    # 上采样到原始图像尺寸
    seg_logits = F.interpolate(seg_logits, size=ori_shape, mode='bilinear')
    seg_logits = torch.softmax(seg_logits * self.region_logit_scale, dim=1)

    # 获取最终预测结果
    seg_preds = seg_logits.argmax(1)
    seg_logits = seg_logits.max(1)[0]
    return seg_logits, seg_preds
```

#### 5.3 指代表达理解
通过计算查询文本与所有区域令牌的相似度，选择最相似的区域作为目标输出。

## 4. 核心优势与创新

### 4.1 无需训练
TextRegion完全基于预训练模型，不需要任何额外的训练过程，节省了大量计算资源。

### 4.2 高效的区域对齐
通过掩码引导的注意力池化，确保每个区域的特征表示与其空间位置精确对齐。

### 4.3 强大的泛化能力
由于利用了大规模预训练模型，TextRegion能够处理未见过的类别和场景。

### 4.4 模块化设计
支持多种图像-文本模型，易于扩展和替换不同的骨干网络。

## 5. 技术要点总结

TextRegion的关键技术要点包括：

1. **软掩码生成**：使用SAM2生成概率性的区域掩码，而非硬分割
2. **多模型适配**：针对不同模型设计相应的特征提取和池化策略
3. **掩码引导池化**：在注意力机制中引入掩码约束，实现精确的区域特征聚合
4. **全局补丁过滤**：移除与特定区域无关的冗余特征，提高表征质量
5. **多分辨率处理**：支持处理高分辨率图像，提升细节捕捉能力

这种设计使得TextRegion在保持零样本能力的同时，实现了精确的区域级语义理解，为开放世界视觉理解任务提供了强大而高效的解决方案。




---

## 一、先给一句明确结论

> ✅ **可以利用 TextRegion 的“区域级建模 + 掩码引导聚合”思想，显著增强 SAM3**
> ❌ **但不能指望 SAM3 自己完成 open-vocab / text–region 对齐**

换句话说：

> **TextRegion = “语义注入”**
> **SAM3 = “结构与边界专家”**
> 👉 你可以用前者的思想，**补后者的短板**

---

## 二、TextRegion 的“核心思想”到底是什么？（去掉 CLIP）

很多人以为 TextRegion 的核心是 CLIP，其实不是。

**TextRegion 真正的创新是这三点：**

### ① 用“区域”而不是“全图”作为基本语义单位

### ② 用“掩码”而不是 attention 学习区域归属

### ③ 用“区域 token”作为可组合、可比较的中间表示

📌 **这三点，全部和 SAM3 高度契合**

---

## 三、SAM3 能从 TextRegion 学到什么？（逐点对齐）

### 🔹 1️⃣ 区域级建模（Region-centric representation）

#### SAM3 现在的问题是：

* decoder 在 **像素层面**工作
* 区域之间没有显式交互
* mask 是结果，不是中间表征

#### TextRegion 的思想是：

> **mask → region → token → reasoning**

✅ 你可以在 SAM3 中引入：

```text
mask → region embedding → region graph / refinement
```

📌 **这能提升：**

* mask 一致性
* 区域间竞争与抑制
* 复杂场景下的稳定性

---

### 🔹 2️⃣ 掩码引导的特征聚合（Mask-guided pooling）

你已经非常接近这一步了（你做的 pixel/token rectifier 本质就在这）。

**具体可以这样做：**

* 从 SAM3 encoder / pixel_embed 中取特征
* 用 decoder 产生的中间 mask
* 对 encoder 特征做：

[
r_k = \sum_i m_{k,i} \cdot f_i
]

📌 **关键：**

* 这个 region token **不对齐文本**
* 但它可以：

  * 反馈给 decoder
  * 用于 mask refinement
  * 用于区域一致性约束

---

### 🔹 3️⃣ 用“区域 token”反向指导 mask（闭环）

这是 **TextRegion 没有、但 SAM3 特别适合做的**。

你可以构造一个 **闭环结构**：

```
pixel → mask → region token
        ↑           ↓
      refine ← region-aware attention
```

📌 这会让 SAM3 从：

> “一次性预测 mask”
> 升级为
> “区域感知的迭代推理”

这在 **小目标 / 遮挡 / 密集实例** 上非常有潜力。

---

## 四、那“text”在这里还能起什么作用？

这是你问得最深的一层 👇

### ❗ 关键观点：

> **在 SAM3 里，text 不一定是“语义监督”，
> 它可以是“结构约束 / 选择信号”。**

### 三种可行用法（不等价，但都合法）

---

### 🟡 方案 A：Text 作为 *prompt selector*（最稳）

* 用 text encoder：

  * 决定 **激活哪些 region**
  * 决定 **mask 之间的权重**
* 不要求 text embedding 可比较

📌 **作用：**

* 提高 prompt-following 能力
* 减少歧义 mask

---

### 🟠 方案 B：Text 作为 *region gating signal*

* text → gate
* region token → filtered
* decoder 只关注相关区域

📌 **这在 referring expression 上很有用**
而且 **不要求 CLIP-level 语义**

---

### 🔴 方案 C：轻量“语义蒸馏”（最有研究价值）

这是我认为**最有潜力的方向**👇

> **不是把 CLIP 接进来推理，
> 而是用 CLIP *训练* SAM3 的 region token**

具体是：

* 训练时：

  * 用 CLIP region embedding 作为 teacher
  * 用 SAM3 region token 作为 student
* 推理时：

  * **只用 SAM3**

📌 这是：

* 参数小
* 推理快
* 论文友好

---

## 五、一个非常重要的判断（我希望你认真看）

> **TextRegion 的思想 ≠ open-vocab 本身**
> **而是“用区域作为语义计算单元”**

而 SAM3 **正缺这个中间层**。

---

## 六、如果让我替你选一个“最可能成功”的方向

我会建议你做这个 👇

### ⭐ **Region-aware SAM3（不做 open-vocab）**

**核心贡献点可以是：**

* region token 引入
* mask-guided pooling
* region-level refinement
* 显著提升：

  * mask 稳定性
  * 小物体
  * 复杂场景

📌 **不碰 open-vocab，反而更容易打穿审稿人。**

---

## 七、最后一句掏心窝子的判断

你现在这个问题，已经不是“能不能”，而是：

> **“怎么做，才能既不背离 SAM3 的设计哲学，
> 又能引入 TextRegion 的优势？”**

而答案是：

> **把“语义对齐”换成“区域建模”，
> 把“文本监督”换成“结构监督”。**


## SAM3实现方案测试

我将根据代码进行具体详解，分析`RegionAwareSegEarthOV3Segmentor`类的各个组成部分和实现细节。

## 1. 类定义与初始化

```python
@MODELS.register_module()
class RegionAwareSegEarthOV3Segmentor(BaseSegmentor):
```

这个类继承自MMSegmentation的[BaseSegmentor](file:///d:/CodeReading/TextRegion/mmseg/models/segmentors/base.py#L14-L274)，并使用`@MODELS.register_module()`装饰器注册到MMSegmentation的模型注册表中，这样可以通过配置文件创建该模型实例。

### 初始化参数详解：

```python
def __init__(self, 
             classname_path,
             device=torch.device('cuda'),
             prob_thd=0.0,
             bg_idx=0,
             slide_stride=0,
             slide_crop=0,
             confidence_threshold=0.5,
             use_sem_seg=True,
             use_presence_score=True,
             use_transformer_decoder=True,
             region_refinement_iterations=0,  # 修复：默认设为0，因为随机初始化的卷积层会破坏特征
             region_similarity_threshold=0.7, # 区域相似度阈值
             region_pooling_method='masked_average',  # 区域池化方法
             score_balance_factor=0.3,  # 添加评分平衡因子，默认值0.3
             **kwargs):
```

- `classname_path`: 类别名称文件路径，用于读取待分割的类别
- `device`: 计算设备
- `prob_thd`: 概率阈值
- `bg_idx`: 背景类索引
- `slide_stride/slide_crop`: 滑动窗口推理参数
- `confidence_threshold`: 置信度阈值
- `use_sem_seg/use_presence_score/use_transformer_decoder`: 控制是否使用不同模块的布尔参数
- `region_refinement_iterations`: 区域精细化迭代次数（默认为0）
- `region_similarity_threshold`: 区域相似度阈值
- `region_pooling_method`: 区域池化方法（'masked_average'或'masked_max'）
- `score_balance_factor`: 评分平衡因子，平衡置信度和区域大小的影响

## 2. 掩码引导池化实现

```python
def _mask_guided_pooling(self, encoder_features, masks):
    """
    使用mask对encoder特征进行池化，得到区域级别的表示
    """
    pooled_regions = []
    for mask in masks:
        # 确保mask是二维的
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        
        # 将mask调整为与特征图相同的尺寸
        resized_mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=encoder_features.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).squeeze()
        
        # 归一化mask
        mask_sum = resized_mask.sum()
        if mask_sum > 0:
            # 根据选择的池化方法进行池化
            if self.region_pooling_method == 'masked_average':
                # 掩码平均池化
                masked_features = encoder_features * resized_mask.unsqueeze(0)
                region_repr = masked_features.sum(dim=[1, 2]) / mask_sum
            elif self.region_pooling_method == 'masked_max':
                # 掩码最大池化（使用一个技巧来应用掩码）
                masked_features = encoder_features.clone()
                masked_features[:, resized_mask < 0.5] = float('-inf')
                region_repr = F.adaptive_max_pool2d(
                    masked_features.unsqueeze(0),
                    output_size=(1, 1)
                ).squeeze()
            else:
                # 默认使用掩码平均池化
                masked_features = encoder_features * resized_mask.unsqueeze(0)
                region_repr = masked_features.sum(dim=[1, 2]) / mask_sum
        else:
            # 如果mask为空，返回零向量
            region_repr = torch.zeros(encoder_features.shape[0], device=encoder_features.device)
        
        pooled_regions.append(region_repr)
    
    return torch.stack(pooled_regions, dim=0)
```

这个方法实现了**掩码引导特征聚合**，是TextRegion思想的重要体现：

- 将分割掩码与编码器特征相乘，只保留感兴趣区域的特征
- 支持两种池化方法：掩码平均池化和掩码最大池化
- 通过归一化操作消除区域大小对特征表示的影响
- 最终生成区域级别的特征表示

## 3. 类内区域聚合

```python
def merge_regions_by_class(self, regions, image_shape):
    """
    类内region聚合，解决遥感中同一类被分成多个小region的问题
    """
    if not regions:
        return []
    
    merged = []
    
    for class_id in set(r['class_id'] for r in regions):
        cls_regions = [r for r in regions if r['class_id'] == class_id]
        used = [False] * len(cls_regions)

        for i, r in enumerate(cls_regions):
            if used[i]:
                continue

            cur_mask = r['mask'].clone()
            cur_score = r['score']
            used[i] = True

            for j in range(i + 1, len(cls_regions)):
                if used[j]:
                    continue
                other = cls_regions[j]

                # IoU-based merge（遥感非常有效）
                inter = (cur_mask & other['mask']).sum()
                union = (cur_mask | other['mask']).sum()
                iou = inter.float() / (union.float() + 1e-6)

                if iou > 0.3:  # 遥感建议 0.3～0.5
                    cur_mask |= other['mask']
                    cur_score = max(cur_score, other['score'])
                    used[j] = True

            merged.append({
                'mask': cur_mask,
                'class_id': class_id,
                'score': cur_score
            })
    return merged
```

这个方法体现了**类内区域聚合**的设计模式：

- 遍历每种类别，将同类别中的多个小区域合并
- 使用IoU（交并比）判断两个区域是否应该合并
- 设置了适合遥感图像的IoU阈值（0.3），当两个区域的IoU超过此值时进行合并
- 保留合并后区域中的最高得分

## 4. 单视图推理

```python
def _inference_single_view(self, image):
    """在单个PIL图像或裁剪块上进行推理，使用region-wise分配策略."""
    w, h = image.size
    # 返回多个候选区域而不是直接的像素预测
    regions = []  # (mask, class_id, score)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        # 设置图像并获取多尺度特征
        inference_state = self.processor.set_image(image)
        
        # 首先获取完整的semantic logits用于点选择
        semantic_logits = torch.zeros((self.num_cls, h, w), device=self.device, dtype=torch.float16)
        for query_idx, query_word in enumerate(self.query_words):
            class_id = int(self.query_idx[query_idx])
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)
            
            if 'semantic_mask_logits' in inference_state:
                semantic_single = inference_state['semantic_mask_logits']
                if semantic_single.shape != (h, w):
                    # 确保张量为4D格式
                    if semantic_single.dim() == 2:
                        semantic_single = semantic_single.unsqueeze(0).unsqueeze(0)
                    elif semantic_single.dim() == 3:
                        semantic_single = semantic_single.unsqueeze(0)
                    semantic_single = F.interpolate(
                        semantic_single, 
                        size=(h, w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                semantic_logits[class_id] = semantic_single.to(semantic_logits.dtype)
        
        # 然后为每个类别生成实例级别的mask
        for query_idx, query_word in enumerate(self.query_words):
            class_id = int(self.query_idx[query_idx])  # 获取对应的类别ID
            
            self.processor.reset_all_prompts(inference_state)
            inference_state = self.processor.set_text_prompt(state=inference_state, prompt=query_word)

            # 获取初始的分割logits
            initial_logits = torch.zeros((h, w), device=self.device, dtype=torch.float16)
            
            if self.use_transformer_decoder and 'masks_logits' in inference_state:
                if inference_state['masks_logits'].shape[0] > 0:
                    inst_len = inference_state['masks_logits'].shape[0]
                    for inst_id in range(inst_len):
                        instance_logits = inference_state['masks_logits'][inst_id].squeeze()
                        instance_score = inference_state['object_score'][inst_id]
                        
                        # 处理潜在的维度不匹配
                        if instance_logits.shape != (h, w):
                            # 确保张量为4D格式
                            if instance_logits.dim() == 2:
                                instance_logits = instance_logits.unsqueeze(0).unsqueeze(0)
                            elif instance_logits.dim() == 3:
                                instance_logits = instance_logits.unsqueeze(0)
                            instance_logits = F.interpolate(
                                instance_logits, 
                                size=(h, w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                        
                        # 使用加权求和而不是max，避免过度抑制
                        initial_logits.add_(instance_logits.to(initial_logits.dtype), alpha=instance_score)
            
            if self.use_sem_seg and 'semantic_mask_logits' in inference_state:
                semantic_single = inference_state['semantic_mask_logits']
                if semantic_single.shape != (h, w):
                    # 确保张量为4D格式
                    if semantic_single.dim() == 2:
                        semantic_single = semantic_single.unsqueeze(0).unsqueeze(0)
                    elif semantic_single.dim() == 3:
                        semantic_single = semantic_single.unsqueeze(0)
                    semantic_single = F.interpolate(
                        semantic_single, 
                        size=(h, w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                
                # 使用加权融合，避免max抑制问题
                # 将semantic_logits转换为float16以匹配其他张量
                initial_logits.add_(semantic_single.to(initial_logits.dtype))
            
            # 应用存在性分数
            presence_score = 1.0
            if self.use_presence_score and "presence_score" in inference_state:
                # 确保presence_score是标量或与initial_logits兼容的形状
                presence_score = inference_state["presence_score"]
                if torch.is_tensor(presence_score) and presence_score.numel() > 1:
                    # 如果presence_score不是标量，取平均值
                    presence_score = presence_score.mean()
            
            # 从initial_logits中提取高质量的mask proposals
            # 使用多个阈值提取不同质量的mask（保持原有逻辑）
            thresholds = [0.1, 0.3, 0.5]
            for threshold in thresholds:
                mask = initial_logits > threshold
                if mask.sum() > 10:  # 确保mask足够大
                    # 计算综合得分，考虑面积、平均置信度和presence score
                    area = mask.sum().float()
                    avg_conf = initial_logits[mask].mean()
                    
                    # 使用改进的评分函数，平衡置信度和区域大小
                    # 使用新的平衡因子来控制置信度和面积的影响
                    normalized_area = torch.log(area + 1)
                    
                    # 为不同类别使用不同的评分策略
                    # wall/roof/road等细长结构需要更高的置信度权重
                    if class_id in [1, 2, 5]:  # wall, road, roof
                        # 对于细长结构，更重视置信度
                        class_specific_factor = 0.2  # 更偏向置信度
                    else:
                        # 对于其他类别，使用默认平衡
                        class_specific_factor = self.score_balance_factor
                    
                    balanced_score = (
                        (1 - class_specific_factor) * avg_conf + 
                        class_specific_factor * normalized_area
                    ) * presence_score.to(initial_logits.dtype)
                    
                    # 对road/wall/roof类进行形态学闭运算优化
                    mask_np = mask.cpu().numpy().astype(np.uint8)
                    if class_id in [1, 2, 5]:  # wall, road, roof
                        kernel = np.ones((5,5), np.uint8)
                        mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
                        mask = torch.from_numpy(mask_np).bool()
                    
                    # 添加区域到列表中
                    regions.append({
                        'mask': mask.cpu(),  # 移到CPU以节省GPU内存
                        'class_id': class_id,
                        'score': balanced_score.item()
                    })
            
            # 及时释放显存
            del initial_logits
            if 'initial_logits' in locals():
                initial_logits = torch.zeros((h, w), device=self.device, dtype=torch.float16)

    # 按得分排序
    regions.sort(key=lambda x: x['score'], reverse=True)
    
    # 类内region聚合
    regions = self.merge_regions_by_class(regions, (h, w))
    
    # 再次按得分排序
    regions.sort(key=lambda x: x['score'], reverse=True)
    
    # 返回region列表而不是像素级logits
    return regions, (h, w), semantic_logits
```

这是实现**区域级建模**的核心方法：

- 使用SAM3模型生成分割掩码和语义logits
- 对每个类别生成多个mask proposals，使用多个阈值（0.1, 0.3, 0.5）提取不同质量的mask
- 实现了改进的评分函数，平衡置信度和区域大小的影响
- 为不同类别使用不同的评分策略（如对wall/roof/road等细长结构使用不同的平衡因子）
- 对某些类别执行形态学操作（如闭运算）优化分割结果
- 将生成的区域存储在列表中并按得分排序

## 5. 滑动窗口推理

```python
def slide_inference(self, image, stride, crop_size):
    """使用PIL裁剪进行滑动窗口推理，使用region-wise策略."""
    # ... 代码省略，主要是遍历图像块 ...
    
    # 全图级region去重（滑窗NMS）
    all_regions = self.merge_regions_by_class(all_regions, (h_img, w_img))
    
    # Region-aware logits reweighting（替代原来的refinement）
    refined_logits = base_logits.clone()
    
    # 为每个类别只选择一个最高得分的region进行reweighting
    class_regions = {}
    for region in all_regions:
        class_id = region['class_id']
        if class_id not in class_regions or region['score'] > class_regions[class_id]['score']:
            class_regions[class_id] = region
    
    for class_id, region in class_regions.items():
        mask = region['mask'].to(self.device)
        score = region['score']
        
        # 如果region分数太低，跳过reweighting
        if score < 0.2:  # 阈值可以根据需要调整
            continue
            
        # 使用region mask进行logit reweighting
        region_mask = mask.float().to(refined_logits.device)
        
        # 对该类别的logits进行reweighting，增强region内部的置信度
        refined_logits[class_id] = (
            base_logits[class_id] * (1 - 0.1 * region_mask) 
            + base_logits[class_id] * region_mask * 1.1  # 略微提升region内部置信度
        )
```

这个方法实现了**Region-aware Logit Reweighting**：

- 在滑动窗口推理后，将所有区域合并到全图坐标系
- 对跨窗口的同类区域进行去重（滑窗NMS）
- 使用区域mask对基础logits进行重加权，增强区域内置信度，同时略微降低区域外置信度

## 6. 预测方法

```python
def predict(self, inputs, data_samples):
    # ... 代码省略，主要是加载图像和确定推理模式 ...
    
    # 获取region并进行region-aware logit reweighting
    regions, (h, w), semantic_logits = self._inference_single_view(image)
    
    # 为每个类别只选择一个最高得分的region进行reweighting
    class_regions = {}
    for region in regions:
        class_id = region['class_id']
        if class_id not in class_regions or region['score'] > class_regions[class_id]['score']:
            class_regions[class_id] = region
    
    # 对region进行logit reweighting
    refined_logits = base_logits.clone()
    
    for class_id, region in class_regions.items():
        mask = region['mask'].to(self.device)
        score = region['score']
        
        # 如果region分数太低，跳过reweighting
        if score < 0.2:  # 阈值可以根据需要调整
            continue
            
        # 使用region mask进行logit reweighting
        region_mask = mask.float().to(refined_logits.device)
        
        # 对该类别的logits进行reweighting，增强region内部的置信度
        refined_logits[class_id] = (
            base_logits[class_id] * (1 - 0.1 * region_mask) 
            + base_logits[class_id] * region_mask * 1.1  # 略微提升region内部置信度
        )
    
    # 创建最终的分割结果
    seg_logits = refined_logits
```

## 总结

这个实现很好地结合了TextRegion的"区域级建模+掩码引导聚合"思想，具体体现在：

1. **区域级建模**：通过生成region proposals而非直接像素预测
2. **掩码引导特征聚合**：使用分割掩码对特征进行池化
3. **类内区域聚合**：解决同一类被分成多个小区域的问题
4. **Region-aware Logit Reweighting**：使用高质量region mask调整logits
5. **多阈值mask提取**：提高不同质量区域的召回率

此外，还针对遥感图像进行了专门优化，如对细长结构使用特殊的评分策略和形态学操作。
