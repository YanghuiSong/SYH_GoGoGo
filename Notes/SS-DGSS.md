

# 🌟 DepthForge: 通过几何一致性实现域泛化语义分割的深度感知方法

> **核心思想**：利用深度信息的稳定性，弥补RGB在极端天气下视觉线索缺失的问题，通过几何一致性提升模型在域泛化语义分割中的性能

---

## 🔍 一、问题背景

### 🌧️ 现实场景痛点
- **域泛化语义分割(DGSS)**：让模型在未见过的域（如雨天、雪天、夜间）上也能准确分割
- **现有方法的局限**：
  - 仅依赖RGB图像，当视觉线索缺失（如夜间、雾天）时性能骤降
  - 无法利用几何结构的稳定性（深度信息更鲁棒）

> ✅ **问题**：RGB在极端条件下（夜间、雪天、雾天）视觉线索不足，导致分割性能大幅下降  
> ❌ **传统方法**：仅用RGB，无法处理视觉线索缺失的情况

---

## 🧠 二、核心思想与创新点

### 🌟 为什么深度信息更鲁棒？
| 条件 | RGB表现 | 深度信息表现 |
|------|----------|--------------|
| **夜间** | 严重模糊，线索少 | 依然清晰，几何结构稳定 |
| **雪天** | 雪花干扰，特征混乱 | 深度信息不受雪花影响 |
| **雾天** | 视觉模糊，边界不清 | 深度信息提供结构信息 |

> 💡 **核心洞察**：视觉线索易变，几何结构稳定 → 深度信息比RGB更鲁棒

### 🚀 DepthForge三大创新

1. **深度感知的可学习令牌**（Depth-aware Learnable Tokens）
   - 为模型引入深度信息，学习空间结构不变性
   - 使模型在视觉线索缺失时仍能保持稳定

2. **深度细化解码器**（Depth Refinement Decoder）
   - 动态调整多尺度特征
   - 结合先验和动态特征，建立高质量的特征关系

3. **几何一致性**（Geometric Consistency）
   - 通过深度信息优化视觉特征，增强几何一致性
   - 使模型在极端条件下仍能保持高精度

---

## 🛠 三、方法详解

### 📐 1. 整体框架（如图3所示）


**核心组件**：
1. **深度感知**（Depth Awareness）
2. **注意力优化**（Attention Optimization）
3. **深度细化解码器**（Depth Refinement Decoder）

---

### 🌐 2. 深度感知（Depth Awareness）

#### 🧠 核心思想
- 利用深度VFM（Depth Anything V2）和视觉VFM（DINOv2/EVA02）提取特征
- 将深度信息融入可学习令牌，生成深度感知的特征

#### 🔧 实现细节
```python
# 深度感知的核心公式
Ai = Softmax(QV × (T^v)^T / √c) + λ * Softmax(QD × (T^d)^T / √c)
```
- `QV` = 视觉查询特征
- `QD` = 深度查询特征
- `T^v`, `T^d` = 深度感知的可学习令牌
- `λ` = 平衡系数

> 💡 **关键点**：深度信息作为额外的先验，帮助模型在视觉线索缺失时仍能保持空间结构

---

### 🔍 3. 注意力优化（Attention Optimization）

#### 🧠 核心思想
- 直接使用注意力表示可能引入无关信息
- 通过深度感知令牌学习对齐权重，避免不必要的调整

#### 🔧 实现细节
```python
# 注意力优化核心公式
Δf̂i = Ai × (Ti × W^Ti + b^Ti)
Δfi = φ(Δf̂i)  # 两个残差连接
```
- `W^Ti`, `b^Ti` = 深度感知令牌学习的权重和偏置
- `φ` = 残差连接，防止梯度消失

> 💡 **关键点**：深度感知令牌不仅提供深度信息，还优化了注意力机制，使特征更具判别性

---

### 📐 4. 深度细化解码器（Depth Refinement Decoder）

#### 🧠 核心思想
- 不同层的深度特征不一致
- 需要设计一个解码器，有效集成多尺度特征

#### 🔧 实现细节
```python
# 深度细化解码器
f_out_i = ϕ(ReLU(φ(f_i)))  # 多层特征处理
F_u = γConcat(f_out_1, f_out_2, ..., f_out_N)  # 特征融合
```
- `ϕ`, `φ` = 两个全连接层
- `γ` = 卷积层
- `Concat` = 特征拼接

> 💡 **关键点**：深度细化解码器通过融合多尺度特征，有效捕获不同尺度的空间相关性

---

## 📊 四、实验结果

### 📈 1. 主要性能对比（Cityscapes → ACDC）

| 方法 | Snow | Night | Fog | Rain | Avg. |
|------|------|-------|-----|------|------|
| REIN | 66.0 | 52.1 | 76.0 | 69.5 | 66.2 |
| FADA | 66.3 | 52.2 | 76.2 | 69.8 | 66.3 |
| **DepthForge** | **67.2** | **53.0** | **77.0** | **70.2** | **67.1** |

> ✅ **提升**：比REIN/FADA平均提升**0.9%**，极端条件（雪/夜/雾/雨）提升**1.7%~4.5%**

---

### 🌧️ 2. 极端条件性能（Cityscapes → ACDC）

| 条件 | REIN | FADA | **DepthForge** |
|------|------|------|---------------|
| **Snow** | 66.0 | 66.3 | **67.2** (+1.2) |
| **Night** | 52.1 | 52.2 | **53.0** (+0.9) |
| **Fog** | 76.0 | 76.2 | **77.0** (+1.0) |
| **Rain** | 69.5 | 69.8 | **70.2** (+0.7) |

> 💡 **关键发现**：在**极端条件**（如雪天、夜晚）下，DepthForge性能提升最显著

---

### 🌐 3. 跨数据集泛化能力（GTA5 → Cityscapes + BDD100k + Mapillary）

| 方法 | Cityscapes | BDD100k | Mapillary | Avg. |
|------|------------|----------|------------|------|
| REIN | 65.3 | 60.5 | 64.9 | 63.6 |
| FADA | 66.7 | 61.9 | 66.1 | 64.9 |
| **DepthForge** | **68.2** | **62.0** | **68.1** | **66.1** |

> ✅ **提升**：比FADA平均提升**1.2%**，在所有目标域上均优于现有方法

---

## 📌 五、关键消融实验

### 🔍 1. 消融实验（Cityscapes → ACDC）

| 配置 | Snow | Night | Fog | Rain | Avg. |
|------|------|-------|-----|------|------|
| REIN | 66.0 | 52.1 | 76.0 | 69.5 | 66.2 |
| + Depth Token | 67.2 | 53.0 | 77.0 | 70.2 | 66.9 |
| + Depth Refinement Decoder | 68.2 | 53.5 | 77.6 | 70.7 | 67.3 |
| **DepthForge (Ours)** | **69.5** | **54.5** | **78.5** | **72.0** | **67.8** |

> 💡 **关键发现**：
> - 深度令牌（Depth Token）带来**+0.7%**提升
> - 深度细化解码器（DRD）带来**+0.4%**提升
> - 三者结合（DepthForge）带来**+1.6%**提升

---

### 🌐 2. 不同深度VFM的比较

| 深度VFM | Snow | Night | Fog | Rain | Avg. |
|---------|------|-------|-----|------|------|
| Prompt Depth Anything | 65.8 | 52.4 | 75.5 | 69.3 | 65.8 |
| **Depth Anything V2** | **67.9** | **53.7** | **77.8** | **70.9** | **66.7** |

> 💡 **关键发现**：**Depth Anything V2**（相对空间关系）比**Prompt Depth Anything**（绝对空间信息）效果更好

---

## 📦 六、实现要点

### 🛠 1. 代码结构（简化版）

```python
class DepthForge(nn.Module):
    def __init__(self, visual_vfm, depth_vfm):
        self.depth_aware_tokens = nn.Parameter(torch.randn(num_tokens, embed_dim))
        self.depth_refinement_decoder = DepthRefinementDecoder()
        
    def forward(self, rgb, depth):
        # 1. 提取视觉和深度特征
        visual_features = self.visual_vfm(rgb)
        depth_features = self.depth_vfm(depth)
        
        # 2. 深度感知
        depth_aware_features = self.depth_aware(visual_features, depth_features)
        
        # 3. 注意力优化
        enhanced_features = self.attention_optimization(depth_aware_features)
        
        # 4. 深度细化解码器
        refined_features = self.depth_refinement_decoder(enhanced_features)
        
        return refined_features
```

---

### ⚙ 2. 关键参数设置

| 参数 | 值 | 说明 |
|------|----|------|
| 优化器 | AdamW | lr=1e-4, weight_decay=0.05 |
| 学习率调度 | OneCycleLR | max_lr=1e-4, warmup=10% |
| 数据增强 | 多尺度缩放、随机裁剪、随机翻转 | 保持几何一致性 |
| 训练步数 | 50,000 | 保证充分收敛 |

---

## 🌐 七、应用场景

### 🚗 1. 自动驾驶
- **挑战**：夜间、雨天、雪天等极端条件下的语义分割
- **DepthForge优势**：在视觉线索缺失时仍能保持高精度

### 📱 2. 移动设备
- **挑战**：计算资源有限，无法使用复杂模型
- **DepthForge优势**：仅需添加少量可学习参数（2.99M），适合移动端部署

---

## 📚 八、结论与未来工作

### ✅ 结论
1. **更强**：在极端条件下（雪/夜/雾/雨）性能显著提升
2. **更稳**：深度感知的可学习令牌使视觉-空间注意力更稳定
3. **更优**：在多个数据集和设置下均达到SOTA

###  z未来工作
1. **扩展到其他任务**：目标检测、实例分割
2. **探索更多深度信息**：如3D点云与深度信息的融合
3. **优化计算效率**：进一步减少计算开销，适合实时应用

---

## 📌 九、重要参考

- **Depth Anything V2**：[https://github.com/DepthAnything/DepthAnythingV2](https://github.com/DepthAnything/DepthAnythingV2)
- **DINOv2**：[https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino)
- **EVA02**：[https://github.com/baaivision/EVA](https://github.com/baaivision/EVA)
- **DepthForge代码**：[https://github.com/SY-Ch/DepthForge](https://github.com/SY-Ch/DepthForge)

---

> 💡 **一句话总结**：DepthForge通过利用深度信息的稳定性，弥补RGB在极端条件下的不足，使模型在域泛化语义分割中表现更强、更稳、更优！
