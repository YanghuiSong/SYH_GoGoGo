

# CLIP2RS: 利用预训练视觉-语言模型进行遥感图像语义分割

## 1. 研究背景与问题

遥感图像语义分割是计算机视觉中的一个挑战性任务，主要因为遥感图像具有**多样性、复杂性和大规模**的特点。现有方法主要依赖于视觉上下文信息，忽略了重要的语义细节，导致在处理**类别内变化**（intra-class variations）时效果不佳。

**问题示例**：如图1所示，"ship"类别在不同遥感图像中呈现显著差异（形状、大小、颜色、位置等），这使得单一视觉特征难以表示同一类别中的所有实例。

![图1：不同场景下相同类别的视图比较](https://i.imgur.com/7Rj8q0l.png)

## 2. CLIP2RS核心思想

CLIP2RS利用预训练的**视觉-语言模型（VLM）**，特别是CLIP，来解决遥感图像语义分割问题。其核心思想是：

1. **利用CLIP的语义能力**：CLIP通过对比学习在4亿个图像-文本对上预训练，能将视觉和文本特征对齐在共享空间中。
2. **克服域差距**：自然图像和遥感图像存在显著差异，CLIP2RS采用两阶段训练策略来解决这个问题。
3. **双粒度对齐**：同时对齐像素级局部特征和图像级全局特征，缓解类别样本不平衡问题。
4. **新型提示机制**：设计双粒度提示，充分挖掘CLIP文本描述的潜力。

## 3. 算法原理

### 3.1 两阶段训练策略

CLIP2RS采用两阶段训练策略来克服自然图像和遥感图像之间的域差距：

![图2：CLIP2RS的两阶段训练策略](https://i.imgur.com/5Z5z3bP.png)

**第一阶段**：
- 冻结CLIP的图像编码器和文本编码器
- 优化可学习的上下文向量，生成适合遥感图像类别的文本提示
- 目标：使文本特征与对应的视觉特征在共享语义空间中对齐

**第二阶段**：
- 冻结文本编码器和第一阶段生成的提示
- 优化图像编码器，使其在文本提示的指导下学习更好的视觉表示

### 3.2 双粒度对齐框架

CLIP2RS设计了一个双粒度对齐框架，同时对齐像素级局部特征和图像级全局特征：

![图3：CLIP2RS的双粒度对齐框架](https://i.imgur.com/3j5Zl1k.png)

**像素级局部对齐**：
- 建立像素级视觉特征和文本特征之间的连接
- 计算局部相似度矩阵：$S_L = I_L F_L^T$
  - $I_L$：图像局部特征（来自图像编码器）
  - $F_L$：文本局部特征（来自文本编码器）
- 使用辅助分割损失：$L_{aux} = CrossEntropy(Softmax(S_L/\tau), Y_L)$
- 使用局部分割损失：$L_{local} = CrossEntropy(Softmax(Dec_I([M, S_L])/\tau), Y_L)$

**图像级全局对齐**：
- 通过[CLS]向量整合所有图像块的信息，形成全局语义概念
- 计算全局相似度向量：$S_G = I_G F_G^T$
  - $I_G$：图像全局特征
  - $F_G$：文本全局特征
- 使用二元交叉熵损失：$L_{global} = BinaryCrossEntropy(Sigmoid(S_G/\tau), Y_G)$

**总损失函数**：
$$L_{task} = \alpha L_{global} + \beta L_{aux} + \gamma L_{local}$$
- $\alpha, \beta, \gamma$：损失权重系数（论文中设置为0.3, 0.3, 0.4）

### 3.3 双粒度提示设计

CLIP2RS设计了双粒度提示，分别描述粗粒度和细粒度的类别信息：

**全局提示（粗粒度）**：
$$P_G = [U_1, U_2, ..., U_M, [CLASS]]$$
- $U$：类别无关的上下文向量（所有类别共享）
- $[CLASS]$：类别名称的词向量

**局部提示（细粒度）**：
$$P_L = [V_1, ..., V_{M1}, W_1, ..., W_{M2}, [CLASS]]$$
- $V$：类别无关的上下文向量（共享）
- $W$：类别特定的上下文向量（每个类别独立）
- $[CLASS]$：类别名称的词向量

**提示设计原理**：
- 全局提示提供整个图像的类别一般描述
- 局部提示具体描述对象类别的详细特征
- 通过双粒度提示，模型能同时学习类别间的共性和差异

## 4. 实现要点

### 4.1 数据预处理

1. **图像裁剪**：
   - iSAID：896×896像素，重叠384（步长512）
   - Potsdam和Vaihingen：512×512像素，重叠128（步长384）

2. **数据增强**：
   - 随机旋转（90°, 180°, 270°）
   - 随机缩放（0.5到2.0）
   - 随机水平和垂直翻转

### 4.2 模型配置

| 组件 | 配置 |
|------|------|
| 图像编码器 | ResNet-50, ResNet-101, ViT-B |
| 文本编码器 | CLIP Transformer |
| 图像解码器 | UperNet |
| 可学习提示长度 | 16（全局和局部） |
| 学习率 | 0.0001 |
| 权重衰减 | 0.001 |
| 训练轮数 | iSAID: 40, Potsdam: 200, Vaihingen: 600 |
| 批次大小 | 2 |

### 4.3 代码实现关键点

```python
# 1. 加载预训练CLIP模型
import clip
from clip import tokenize

model, preprocess = clip.load("ViT-B/32", device="cuda")

# 2. 定义双粒度提示
class DualGranularityPrompt:
    def __init__(self, num_classes, context_length=16):
        # 全局提示：类别无关上下文 + 类别名称
        self.global_prompt = nn.Parameter(torch.randn(1, context_length, 512))
        
        # 局部提示：类别无关上下文 + 类别特定上下文 + 类别名称
        self.local_prompt_shared = nn.Parameter(torch.randn(1, context_length//2, 512))
        self.local_prompt_specific = nn.Parameter(torch.randn(num_classes, context_length//2, 512))
        
        self.class_names = ["class1", "class2", ...]  # 类别名称

    def forward(self, class_idx):
        # 全局提示
        global_prompt = torch.cat([
            self.global_prompt.expand(1, -1, -1),
            self.class_names[class_idx].unsqueeze(0)
        ], dim=1)
        
        # 局部提示
        local_prompt_shared = self.local_prompt_shared.expand(1, -1, -1)
        local_prompt_specific = self.local_prompt_specific[class_idx].unsqueeze(0)
        local_prompt = torch.cat([
            local_prompt_shared,
            local_prompt_specific,
            self.class_names[class_idx].unsqueeze(0)
        ], dim=1)
        
        return global_prompt, local_prompt

# 3. 两阶段训练策略
def train_stage_1(model, dataloader, num_classes):
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 仅优化提示参数
    optimizer = optim.Adam(model.prompt.parameters(), lr=0.001)
    
    for epoch in range(50):
        for images, labels in dataloader:
            # 生成提示
            global_prompt, local_prompt = model.prompt(class_idx)
            
            # 前向传播
            image_features = model.image_encoder(images)
            text_features_global = model.text_encoder(global_prompt)
            text_features_local = model.text_encoder(local_prompt)
            
            # 计算损失
            loss = compute_contrastive_loss(image_features, text_features_global, text_features_local)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train_stage_2(model, dataloader, num_classes):
    # 冻结提示和文本编码器
    for param in model.prompt.parameters():
        param.requires_grad = False
    for param in model.text_encoder.parameters():
        param.requires_grad = False
    
    # 优化图像编码器
    optimizer = optim.Adam(model.image_encoder.parameters(), lr=0.0001)
    
    for epoch in range(100):
        for images, labels in dataloader:
            # 获取提示
            global_prompt, local_prompt = model.prompt(class_idx)
            
            # 前向传播
            image_features = model.image_encoder(images)
            text_features_global = model.text_encoder(global_prompt)
            text_features_local = model.text_encoder(local_prompt)
            
            # 计算双粒度损失
            global_loss = compute_global_loss(image_features, text_features_global)
            local_loss = compute_local_loss(image_features, text_features_local)
            aux_loss = compute_aux_loss(image_features, text_features_local)
            
            total_loss = 0.3 * global_loss + 0.3 * aux_loss + 0.4 * local_loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

## 5. 实验结果

CLIP2RS在三个广泛使用的遥感数据集上进行了实验：

| 数据集 | 模型 | mIoU | 优势类别 |
|--------|------|------|----------|
| iSAID | CLIP2RS (ViT-B) | 68.0% | baseball diamond, basketball court, helicopter等 |
| Potsdam | CLIP2RS (ViT-B) | mF1: 93.80%, OA: 93.35%, mIoU: 88.32% | buildings, low vegetation, trees, cars |
| Vaihingen | CLIP2RS (ViT-B) | mF1: 93.52%, OA: 92.76%, mIoU: 85.25% | impervious surface, building, low vegetation, car |

**关键发现**：
1. CLIP2RS在所有数据集上都达到了最先进的性能
2. 对于小样本类别（如存储罐、直升机等），性能提升尤为显著
3. 两阶段训练策略有效缓解了域差距问题
4. 双粒度对齐框架显著改善了类别不平衡问题

## 6. 为什么有效？

1. **两阶段训练**：第一阶段生成适合遥感图像的文本描述，第二阶段利用这些描述优化视觉表示
2. **双粒度对齐**：同时考虑局部细节和全局上下文，平衡类别梯度贡献
3. **双粒度提示**：全局提示提供类别一般描述，局部提示提供类别特定细节

## 7. 代码实现建议

以下是一个简化版的CLIP2RS实现框架：

```python
import torch
import torch.nn as nn
import clip
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class CLIP2RS(nn.Module):
    def __init__(self, num_classes, image_encoder="ViT-B/32", context_length=16):
        super().__init__()
        # 加载预训练CLIP模型
        self.clip_model, self.preprocess = clip.load(image_encoder, device="cuda")
        
        # 冻结CLIP参数
        for param in self.clip_model.parameters():
            param.requires_grad = False
        
        # 双粒度提示
        self.global_prompt = nn.Parameter(torch.randn(1, context_length, 512))
        self.local_prompt_shared = nn.Parameter(torch.randn(1, context_length//2, 512))
        self.local_prompt_specific = nn.Parameter(torch.randn(num_classes, context_length//2, 512))
        
        # 图像解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, images, class_idx):
        # 图像编码
        image_features = self.clip_model.encode_image(images)
        
        # 文本提示
        global_prompt = self._get_global_prompt(class_idx)
        local_prompt = self._get_local_prompt(class_idx)
        
        # 文本编码
        text_features_global = self.clip_model.encode_text(global_prompt)
        text_features_local = self.clip_model.encode_text(local_prompt)
        
        # 双粒度对齐
        global_sim = torch.cosine_similarity(image_features, text_features_global, dim=1)
        local_sim = torch.cosine_similarity(image_features, text_features_local, dim=1)
        
        # 图像解码
        features = image_features.view(-1, 512, 14, 14)
        logits = self.decoder(features)
        
        return logits, global_sim, local_sim
    
    def _get_global_prompt(self, class_idx):
        # 生成全局提示
        class_name = self.class_names[class_idx]
        prompt = self.global_prompt.expand(1, -1, -1)
        return torch.cat([prompt, class_name], dim=1)
    
    def _get_local_prompt(self, class_idx):
        # 生成局部提示
        class_name = self.class_names[class_idx]
        shared_prompt = self.local_prompt_shared.expand(1, -1, -1)
        specific_prompt = self.local_prompt_specific[class_idx].unsqueeze(0)
        return torch.cat([shared_prompt, specific_prompt, class_name], dim=1)
```

## 8. 总结

CLIP2RS通过利用预训练的视觉-语言模型，为遥感图像语义分割提供了新的思路。其核心创新点包括：

1. **两阶段训练策略**：有效克服自然图像和遥感图像之间的域差距
2. **双粒度对齐框架**：同时对齐局部和全局特征，缓解类别不平衡问题
3. **双粒度提示设计**：充分利用CLIP的文本描述能力，同时捕捉类别间的共性和差异

这些创新使得CLIP2RS在多个遥感图像语义分割基准数据集上达到了最先进的性能，特别是对于小样本类别和具有显著类别内变化的场景。
