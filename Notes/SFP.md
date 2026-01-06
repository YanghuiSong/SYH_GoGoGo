# SFP (Self-adaptive Feature Purifier) 学习笔记

## 1. SFP概述

SFP（Self-adaptive Feature Purifier）是一个为解决训练-free开放词汇语义分割中CLIP模型注意力机制引发的异常值传播问题而提出的框架。该方法无需训练即可提升现有视觉语言模型在复杂场景下的分割精度。

### 1.1 核心问题
- CLIP模型中间层注意力易聚焦于图像token中的"异常值"（outliers），导致无关区域过度激活
- 异常值在深层传播，破坏空间感知能力，影响最终分割效果

### 1.2 解决方案
SFP通过三个核心模块解决上述问题：
- SOF (Self-adaptive Outlier Mitigator) - 自适应异常值缓解
- SAE (Semantic-aware Attention Enhancer) - 语义感知注意力增强
- HAI (Hierarchical Attention Integrator) - 分层注意力整合器

## 2. SFP架构详解

### 2.1 输入处理
SFP接收输入图像，首先通过Vision Transformer (ViT)进行编码：

```python
# 图像转换为patch
x = self.conv1(x)  # [B, width, feat_h, feat_w]
feat_w, feat_h = x.shape[-2], x.shape[-1]
x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, feat_h*feat_w]
x = x.permute(0, 2, 1)  # [B, feat_h*feat_w, width]
# 添加class token
x = torch.cat([class_token, x], dim=1)  # [B, 1+feat_h*feat_w, width]
```

### 2.2 Vision Transformer的修改

在原始CLIP的Vision Transformer基础上，SFP做了以下修改：

#### 2.2.1 注意力权重收集
在每一层Transformer块中收集注意力权重：
```python
for idx, blk in enumerate(self.transformer.resblocks[:-1], start=1):
    x, attn_weight = blk(x)  # 获取注意力权重
    attn_weight[:,:,0,1:] *= self.res_cls * self.res_scale  # 调整class-token注意力
    feats_list.append(x)  # 存储特征
    attn_list.append(attn_weight)  # 存储注意力
    attn_maps += attn_weight[:,:,1:,1:]  # 累积patch间注意力
```

#### 2.2.2 多层特征存储
SFP在每一层都存储特征和注意力图，为后续的跨层融合做准备。

## 3. SOF模块 - 自适应异常值缓解

SOF模块是SFP的核心组件之一，负责检测和净化异常patch。

### 3.1 异常值检测策略

SFP实现了多种异常值检测策略：

#### 3.1.1 SOM (Self-Attention Outlier Mitigator) 方法
```python
# 提取注意力权重
attn_weight = attn_weight.squeeze(0)  # 移除batch维度
attn_weight = attn_weight.mean(dim=0)  # 平均多头注意力
cls_attn = attn_weight[0, 1:]  # class token对所有patch的注意力
self_attn = torch.diag(attn_weight[1:, 1:])  # patch的自注意力
important_mask = cls_attn > self_attn  # 判断class注意力是否大于自注意力

# 选择最重要的patches作为潜在异常值
important_scores = cls_attn[important_mask]
important_indices = torch.nonzero(important_mask, as_tuple=False).squeeze(1)
sorted_scores, sort_idx = torch.sort(important_scores, descending=True)
top_patch_indices = important_indices[sort_idx[:10]]  # 选择前10个最显著的patch
```

**SOM检测原理**：如果某个patch的class-token注意力大于其self-attention，则认为该patch是异常值，因为它对分类更重要但内部一致性较低。

#### 3.1.2 L2范数方法
```python
cls_token = x[:1, ...]
norms = torch.norm(x[1:, ...].squeeze(1), dim=1)  # 计算每个patch的L2范数
sorted_norms, sorted_indices = torch.sort(norms, descending=True)
outlier_index = [
    (
        torch.div(index.detach().cpu(), feat_w, rounding_mode='trunc'), 
        index.detach().cpu() % feat_w
    ) 
    for index in sorted_indices[:10]  # 选择L2范数最大的10个patch
]
```

#### 3.1.3 LOF (Local Outlier Factor) 方法
```python
def lof_pytorch(self, x, n_neighbors=30, contamination=0.05):
    # 计算距离矩阵
    distances = torch.norm(x[:, None] - x[None, :], dim=2, p=2) ** 2
    # 获取K近邻
    knn_distances, knn_indices = torch.topk(distances, k=n_neighbors+1, largest=False)
    knn_distances, knn_indices = knn_distances[:, 1:], knn_indices[:, 1:]
    # 计算局部可达密度
    k_distances = knn_distances[:, -1].unsqueeze(1).expand_as(knn_distances)
    reach_distances = torch.max(knn_distances, k_distances)
    LRD = n_neighbors / torch.nan_to_num(reach_distances.mean(dim=1), nan=1e-6)
    # 计算LOF分数
    LRD_ratios = LRD[knn_indices] / LRD.unsqueeze(1)
    LOF_scores = LRD_ratios.mean(dim=1)
    # 确定异常值阈值
    threshold = torch.quantile(LOF_scores.to(torch.float32), 1 - contamination)
    outlier_mask = LOF_scores > threshold
    outlier_indices = torch.where(outlier_mask)[0]
    return outlier_indices, LOF_scores
```

### 3.2 异常值净化

检测到异常patch后，使用均值插值进行净化：

```python
def mean_interpolation(self, fmap, keep_pixels):
    """
    使用邻域均值替换异常patch
    fmap: [B, C, H, W] - 特征图
    keep_pixels: 异常patch的坐标列表
    """
    if len(keep_pixels) == 0:
        return fmap
    
    b, c, h, w = fmap.shape
    dev = fmap.device
    dt = fmap.dtype
    
    # 定义3x3邻域核，中心为0
    neigh_kernel = torch.ones((c, 1, 3, 3), device=dev, dtype=dt)
    neigh_kernel[:, :, 1, 1] = 0   # 去掉中心
    
    xy = torch.tensor(keep_pixels, device=dev, dtype=torch.long)
    # 创建掩码，标记异常patch位置
    spatial_mask = torch.ones((1, 1, h, w), device=dev, dtype=dt)
    spatial_mask[:, :, xy[:, 0], xy[:, 1]] = 0  # 异常位置设为0
    
    # 遮蔽异常patch
    blocked_fmap = fmap * spatial_mask
    pad_fmap = F.pad(blocked_fmap, (1, 1, 1, 1))
    pad_mask = F.pad(spatial_mask, (1, 1, 1, 1))

    # 计算邻域和与邻域计数
    neigh_sum = F.conv2d(pad_fmap, neigh_kernel, groups=c)
    neigh_count = F.conv2d(pad_mask, neigh_kernel[:, :1], groups=1)
    neigh_count_safe = neigh_count.clone()
    neigh_count_safe[neigh_count_safe == 0] = 1  # 避免除零
    neigh_mean = neigh_sum / neigh_count_safe  # 计算邻域均值
    
    # 创建更新掩码
    update_geo = torch.zeros((1, 1, h, w), device=dev, dtype=dt)
    update_geo[:, :, xy[:, 0], xy[:, 1]] = 1  # 异常位置设为1
    
    # 用邻域均值替换异常patch
    updated_fmap = fmap * (1 - update_geo) + neigh_mean * update_geo
    return updated_fmap
```

## 4. SAE模块 - 语义感知注意力增强

SAE模块通过自定义注意力机制增强语义相关区域的注意力：

```python
def custom_attn(self, attn_layer, x: torch.Tensor):
    """
    自定义注意力机制，结合Q和K的注意力
    """
    num_heads = attn_layer.num_heads
    _, bsz, embed_dim = x.size()
    head_dim = embed_dim // num_heads
    scale = head_dim ** -0.5

    # 线性变换得到q, k, v
    q, k, v = F.linear(x, attn_layer.in_proj_weight, attn_layer.in_proj_bias).chunk(3, dim=-1)
    q = q.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    # 计算Q和K的自注意力
    q_attn = torch.bmm(q, q.transpose(1, 2)) * scale
    k_attn = torch.bmm(k, k.transpose(1, 2)) * scale
    # 组合Q和K的注意力
    attn_weights = F.softmax(k_attn, dim=-1) + F.softmax(q_attn, dim=-1)
    attn_weights /= 2  # 平均两种注意力

    # 应用注意力权重
    attn_output = torch.bmm(attn_weights, v)
    attn_output = attn_output.transpose(0, 1).contiguous().view(-1, bsz, embed_dim)
    attn_output = attn_layer.out_proj(attn_output)

    return attn_output
```

## 5. HAI模块 - 分层注意力整合器

HAI模块通过跨层注意力融合多层信息：

```python
# 融合指定层范围内的特征 (cross_s 到 cross_e)
cross_feats = torch.sum(torch.stack(feats_list[self.cross_s:self.cross_e]), dim=0)
cross_clstoken = cross_feats[:1, ...]  # 提取class token
cross_patchtoken = cross_feats[1:, ...]  # 提取patch tokens

# 应用自定义注意力和class token增强
final_block = self.transformer.resblocks[-1]
cross_patchtoken = self.custom_attn(
    final_block.attn, final_block.ln_1(cross_patchtoken)
) + self.res_cls * cross_clstoken  # 增强class token的影响
```

## 6. Patch层面的变化分析

### 6.1 位置表示
- 输入图像被划分为 [feat_w × feat_h] 的patch网格
- 每个patch对应ViT特征图中的一个位置
- 异常patch的坐标通过 `torch.div(idx, feat_w, rounding_mode='trunc')` 和 `idx % feat_w` 转换为空间坐标

### 6.2 变化过程
1. **原始状态**: 每个patch保留其原始特征表示
2. **异常检测**: 通过SOM/L2/LOF方法识别异常patch
3. **净化过程**: 用邻域均值替换异常patch的特征值
4. **注意力调整**: 调整class-token与patch-token之间的注意力权重

### 6.3 可视化方法
```python
# 标记异常patch的位置
outlier_index = [
    (
        torch.div(idx.detach().cpu(), feat_w, rounding_mode='trunc'),
        idx.detach().cpu() % feat_w
    )
    for idx in top_patch_indices
]
```

## 7. 算法流程总结

1. **输入图像** → **ViT编码** → **多层特征提取**
2. **逐层收集注意力权重** → **异常值检测** (SOM/L2/LOF)
3. **异常patch净化** (均值插值) → **语义增强** (自定义注意力)
4. **跨层融合** → **最终输出**

SFP通过在CLIP的Vision Transformer的多个层面进行优化，有效缓解了异常值传播问题，提高了开放词汇语义分割的准确性。整个过程无需重新训练模型，是一种即插即用的解决方案。
