# SegEarth-OV详细工作流程及形状变化分析

## 1. 输入预处理阶段

### 1.1 图像输入
- **原始输入**：[B, 3, H, W]，任意分辨率的遥感图像
- **预处理**：配置文件中统一调整为(448, 448)，保持宽高比
- **处理后**：[B, 3, 448, 448]

### 1.2 文本输入
- **类别文件读取**：从name_path文件读取类别名称
- **类别别名处理**：支持同一类别多个别名（逗号分隔）
- **模板扩展**：使用10个ImageNet模板为每个类别生成变体描述
- **文本编码**：每个类别生成10个文本描述，分别编码为特征向量

## 2. 模型初始化阶段

### 2.1 视觉编码器
- **ViT-B/16**：patch_size=16，image_size=224或448
- **ViT-L/14**：patch_size=14，image_size=224或448
- **patch数量计算**：(448/16)×(448/16) = 28×28 = 784 patches（以ViT-B/16为例）

### 2.2 文本特征构建
- **单个模板编码**：[max_length] → [feature_dim]（如512或1024）
- **模板扩展**：10个模板 → [10, feature_dim]
- **归一化**：每个模板特征L2归一化 → [10, feature_dim]
- **平均融合**：模板维度平均 → [feature_dim]
- **最终归一化**：再次L2归一化 → [feature_dim]
- **类别特征矩阵**：[num_classes, feature_dim] → [query_features](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L169-L169)

## 3. 前向传播阶段

### 3.1 推理方式选择
- **slide_crop > 0**：启用滑动窗口推理
- **slide_crop ≤ 0**：直接推理

### 3.2 滑动窗口处理（以slide_crop=224为例）

#### 3.2.1 网格划分
- **窗口大小**：crop_size=224
- **步长**：stride=112
- **网格数**：h_grids, w_grids取决于原图尺寸

#### 3.2.2 窗口处理流程
- **裁剪图像**：[B, 3, 224, 224]
- **填充处理**：确保尺寸能被patch_size整除
- **调用forward_feature处理**

### 3.3 特征提取与匹配（forward_feature）

#### 3.3.1 视觉特征提取
**BLIP模型**：
- **输入**：[1, 3, H, W] → 插值到[1, 3, 224, 224]
- **ViT编码**：[1, 3, 224, 224] → [1, num_patches+1, feature_dim]
- **去除CLS**：[1, num_patches, feature_dim]，其中num_patches = (224/16)² = 14² = 196
- **投影**：[1, 196, feature_dim] → [1, 196, embed_dim]

**其他模型**：
- **ViT编码**：[1, 3, H, W] → [1, num_patches+1, feature_dim]
- **去除CLS**：[1, num_patches, feature_dim]

#### 3.3.2 CLS Token处理（可选）
- **分离CLS**：[1, feature_dim]
- **归一化**：L2归一化
- **文本匹配**：[1, feature_dim] @ [num_classes, feature_dim].T → [1, num_classes]
- **结果**：[cls_logits](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L320-L320)形状为[1, num_classes]

#### 3.3.3 特征上采样（SimFeatUp）
- **形状变换**：[1, num_patches, feature_dim] → [1, feature_dim, feature_w, feature_h]
  - feature_w = H // patch_size, feature_h = W // patch_size
  - 例如：[1, 196, 768] → [1, 768, 14, 14]（patch_size=16）
- **上采样**：[1, 768, 14, 14] → [1, 768, H, W]
- **恢复形状**：[1, 768, H, W] → [1, H*W, 768]

#### 3.3.4 特征匹配
- **图像特征归一化**：[1, H*W, feature_dim] → L2归一化
- **文本特征**：[num_classes, feature_dim]
- **转置**：[num_classes, feature_dim] → [feature_dim, num_classes]
- **点积匹配**：[1, H*W, feature_dim] @ [feature_dim, num_classes] → [1, H*W, num_classes]
- **结果**：[logits](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L336-L336)形状为[1, H*W, num_classes]

#### 3.3.5 融合CLS Logits（可选）
- **融合**：logits = logits + cls_logits * [cls_token_lambda](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L110-L110)

#### 3.3.6 形状调整
- **维度变换**：[1, H*W, num_classes] → [1, num_classes, H, W]
- **插值**：[1, num_classes, H, W] → [1, num_classes, orig_H, orig_W]（双线性插值）

### 3.4 滑动窗口融合
- **累积预测**：[preds](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L414-L414)形状为[B, num_classes, H, W]
- **计数矩阵**：[count_mat](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L415-L415)形状为[B, 1, H, W]
- **最终融合**：preds / count_mat → [B, num_classes, H, W]

## 4. 后处理阶段

### 4.1 Logit缩放
- **缩放**：[B, num_classes, H, W] * [logit_scale](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L109-L109)（默认50）
- **Softmax归一化**：[B, num_classes, H, W] → 每个像素的类别概率和为1

### 4.2 类别映射（如需要）
- **查询词数量 ≠ 类别数量时**：
  - **one-hot矩阵**：[num_queries, num_classes] → [num_classes, num_queries, 1, 1]
  - **元素乘积**：[num_classes, H, W] * [num_classes, num_queries, 1, 1] → [num_classes, num_queries, H, W]
  - **取最大值**：在查询词维度取最大值 → [num_classes, H, W]

### 4.3 分割预测
- **Argmax**：[num_classes, H, W] → [1, H, W]（预测类别图）
- **置信度过滤**：低于[prob_thd](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L108-L108)的像素设为[bg_idx](file:///d:/CodeReading/SegEarth/SegEarth-OV/segearth_segmentor.py#L112-L112)

## 5. 完整形状变化流程总结

```
图像输入: [B, 3, H, W] 
    ↓ (预处理)
    [B, 3, 448, 448]
    ↓ (ViT编码)
    [B, num_patches+1, feature_dim] 
    ↓ (去除CLS)
    [B, num_patches, feature_dim]
    ↓ (SimFeatUp, if enabled)
    [B, H*W, feature_dim]
    ↓ (L2归一化)
    [B, H*W, feature_dim]

文本输入: [num_classes, class_names] 
    ↓ (模板扩展, 10个模板)
    [num_classes, 10, max_length]
    ↓ (编码)
    [num_classes, 10, feature_dim]
    ↓ (L2归一化)
    [num_classes, 10, feature_dim]
    ↓ (模板平均)
    [num_classes, feature_dim]
    ↓ (最终归一化)
    [num_classes, feature_dim]

特征匹配: [B, H*W, feature_dim] @ [num_classes, feature_dim].T
    ↓ (点积)
    [B, H*W, num_classes]
    ↓ (维度重排)
    [B, num_classes, H, W]
    ↓ (插值到原图尺寸)
    [B, num_classes, orig_H, orig_W]
    ↓ (Softmax归一化)
    [B, num_classes, orig_H, orig_W]
    ↓ (Argmax)
    [B, 1, orig_H, orig_W] ← 分割结果
```

## 6. 关键参数影响

- **patch_size**: 决定特征图尺寸，影响空间分辨率
- **feature_up**: 控制是否使用SimFeatUp恢复空间细节
- **logit_scale**: 控制logit缩放，影响概率分布锐度
- **prob_thd**: 概率阈值，控制低置信度像素的处理
- **slide_crop/stride**: 控制滑动窗口处理的粒度

这个完整流程展示了SegEarth-OV如何将图像和文本信息结合，通过视觉-语言对齐实现开放词汇语义分割。
