# SM3Det 算法详解

## 1. 任务定义：M2Det

### 1.1 问题背景
传统遥感目标检测的局限性：
- 单一数据集训练
- 单一成像模态
- 单一标注格式（水平框或旋转框）
- 无法利用跨模态共享知识

### 1.2 M2Det任务目标
构建统一的检测模型，能够：
- 处理任意传感器模态（SAR、光学、红外等）
- 处理多种检测任务（水平框HBB、旋转框OBB）
- 无需空间对齐的图像对

## 2. 核心挑战

### 2.1 表示约束
密集模型使用相同参数处理多模态数据，导致：
- 特征空间拥挤
- 难以拟合不同数据分布

### 2.2 优化不一致
- 不同模态学习难度不同
- 不同任务优化方向冲突
- 收敛速度不同步

## 3. SM3Det解决方案

### 3.1 整体架构
```
输入图像 → 网格级MoE主干 → 多任务检测头
              ↓
        动态子模块优化(DSO)
```

### 3.2 网格级稀疏MoE（核心创新1）

#### 3.2.1 基本原理
```python
# MoE层数学公式
def moe_forward(x_ij):
    # x_ij: 第i行j列的局部特征
    # N: 专家总数, k: 激活的专家数
    
    # 门控计算
    gating_scores = softmax((E^T * W * x_ij) / (τ * ||W*x_ij|| * ||E||))
    topk_indices = topk(gating_scores, k)  # 选择top-k专家
    
    # 加权求和
    output = 0
    for idx in topk_indices:
        expert_output = conv_1x1_n(x_ij)  # 第n个专家的1x1卷积
        weight = gating_scores[idx]
        output += weight * expert_output
    
    return output
```

#### 3.2.2 源码实现
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GridLevelMoE(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=8, top_k=2, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # 专家网络：1x1卷积
        self.experts = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1) 
            for _ in range(num_experts)
        ])
        
        # 专家嵌入矩阵 E
        self.expert_embeddings = nn.Parameter(
            torch.randn(num_experts, out_channels)
        )
        
        # 特征变换矩阵 W
        self.feature_proj = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        """
        x: [B, C, H, W] 输入特征图
        返回: [B, C, H, W] 输出特征图
        """
        B, C, H, W = x.shape
        
        # 变换输入特征
        projected_x = self.feature_proj(x)  # [B, C, H, W]
        
        # 重塑为网格级特征
        grid_features = projected_x.permute(0, 2, 3, 1).reshape(B * H * W, -1)  # [B*H*W, C]
        original_features = x.permute(0, 2, 3, 1).reshape(B * H * W, -1)  # [B*H*W, C]
        
        # 计算门控分数
        expert_emb_norm = F.normalize(self.expert_embeddings, p=2, dim=1)  # [N, C]
        feature_norm = F.normalize(grid_features, p=2, dim=1)  # [B*H*W, C]
        
        # 相似度计算
        similarity = torch.matmul(feature_norm, expert_emb_norm.t())  # [B*H*W, N]
        gating_scores = F.softmax(similarity / self.temperature, dim=1)  # [B*H*W, N]
        
        # 选择top-k专家
        topk_scores, topk_indices = torch.topk(gating_scores, self.top_k, dim=1)  # [B*H*W, k]
        
        # 重新归一化top-k分数
        topk_scores = topk_scores / topk_scores.sum(dim=1, keepdim=True)
        
        # 应用专家网络
        output = torch.zeros_like(grid_features)
        
        # 对每个位置应用选中的专家
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # [B*H*W]
            expert_weight = topk_scores[:, i].unsqueeze(1)  # [B*H*W, 1]
            
            # 应用对应的专家网络
            for batch_idx in range(B):
                batch_mask = torch.arange(B * H * W) // (H * W) == batch_idx
                batch_expert_idx = expert_idx[batch_mask]
                
                # 对每个专家分别处理
                unique_experts = torch.unique(batch_expert_idx)
                for expert_id in unique_experts:
                    expert_mask = batch_expert_idx == expert_id
                    global_mask = batch_mask.clone()
                    global_mask[batch_mask] = expert_mask
                    
                    if expert_mask.sum() > 0:
                        # 提取对应的原始特征
                        expert_input = original_features[global_mask].unsqueeze(-1).unsqueeze(-1)  # [M, C, 1, 1]
                        
                        # 应用专家网络
                        expert_output = self.experts[expert_id](expert_input)  # [M, C, 1, 1]
                        expert_output = expert_output.squeeze(-1).squeeze(-1)  # [M, C]
                        
                        # 加权求和
                        weights = expert_weight[global_mask]
                        output[global_mask] += weights * expert_output
        
        # 重塑回原形状
        output = output.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        return output
```

#### 3.2.3 关键优势
- **网格级处理**：每个空间位置独立选择专家
- **动态路由**：根据输入内容自适应选择专家
- **稀疏激活**：只激活top-k专家，保证效率
- **共享+特有表示**：同时学习跨模态共享知识和模态特有模式

### 3.3 动态子模块优化DSO（核心创新2）

#### 3.3.1 基本原理
```python
# DSO优化机制
def dso_optimization(current_losses, historical_losses):
    # 计算每个任务的损失比率
    r_t = current_loss_t / historical_loss_t
    
    # 任务头学习率调整（平衡收敛速度）
    head_lr_scale = (r_t) ** p  # p为超参数
    
    # 计算一致性分数C
    C = std(r_1, r_2, ..., r_T) / mean(r_1, r_2, ..., r_T)
    
    # 主干网络学习率调整（保证优化方向一致）
    backbone_lr_scale = 2 * sigmoid((C - b) * τ)
    
    return head_lr_scale, backbone_lr_scale
```

#### 3.3.2 源码实现
```python
class DynamicSubmoduleOptimizer:
    def __init__(self, optimizer, num_tasks, alpha=0.9, p=0.5, bias=0.4, temperature=3.0):
        self.optimizer = optimizer
        self.num_tasks = num_tasks
        self.alpha = alpha  # EMA平滑系数
        self.p = p  # 任务头调整指数
        self.bias = bias  # 重加权阈值
        self.temperature = temperature  # 温度参数
        
        # 历史损失记录
        self.historical_losses = [0.0] * num_tasks
        self.initialized = [False] * num_tasks
        
    def update_learning_rates(self, current_losses, iteration):
        """
        current_losses: 当前迭代各任务的损失列表
        iteration: 当前迭代次数
        """
        head_scales = []
        task_ratios = []
        
        # 更新历史损失并计算比率
        for i, current_loss in enumerate(current_losses):
            if not self.initialized[i]:
                self.historical_losses[i] = current_loss
                self.initialized[i] = True
                ratio = 1.0
            else:
                # EMA更新历史损失
                self.historical_losses[i] = (
                    self.alpha * current_loss + 
                    (1 - self.alpha) * self.historical_losses[i]
                )
                # 计算损失比率
                ratio = current_loss / (self.historical_losses[i] + 1e-8)
            
            task_ratios.append(ratio)
            
            # 任务头学习率调整
            head_scale = ratio ** self.p
            head_scales.append(head_scale)
        
        # 计算一致性分数C
        ratios_tensor = torch.tensor(task_ratios)
        C = torch.std(ratios_tensor) / (torch.mean(ratios_tensor) + 1e-8)
        C = C.item()
        
        # 主干网络学习率调整
        backbone_scale = 2 * torch.sigmoid(torch.tensor((C - self.bias) * self.temperature))
        backbone_scale = backbone_scale.item()
        
        # 应用学习率调整
        self.apply_scales(head_scales, backbone_scale)
        
        return head_scales, backbone_scale
    
    def apply_scales(self, head_scales, backbone_scale):
        """应用计算得到的学习率缩放因子"""
        param_groups = self.optimizer.param_groups
        
        # 假设前N组是任务头参数，最后一组是主干网络参数
        for i, scale in enumerate(head_scales):
            if i < len(param_groups) - 1:
                for param in param_groups[i]['params']:
                    param_group_lr = param_groups[i]['lr']
                    param_groups[i]['lr'] = param_group_lr * scale
        
        # 调整主干网络学习率
        backbone_group = param_groups[-1]
        original_lr = backbone_group['lr']
        backbone_group['lr'] = original_lr * backbone_scale

# 使用示例
def train_step(model, dataloader, dso_optimizer):
    for batch_idx, (images, targets) in enumerate(dataloader):
        # 前向传播
        outputs, losses = model(images, targets)
        
        # 获取各任务损失
        task_losses = [losses['task1'], losses['task2'], losses['task3']]
        
        # 更新学习率
        head_scales, backbone_scale = dso_optimizer.update_learning_rates(
            task_losses, batch_idx
        )
        
        # 反向传播
        total_loss = sum(task_losses)
        total_loss.backward()
        
        # 优化器步进
        dso_optimizer.optimizer.step()
        dso_optimizer.optimizer.zero_grad()
```

#### 3.3.3 关键优势
- **细粒度控制**：分别调整任务头和主干网络的学习率
- **自适应平衡**：根据任务收敛情况动态调整
- **优化一致性**：保证多任务优化方向一致
- **训练稳定性**：防止某些任务主导训练过程

## 4. 完整模型架构

```python
class SM3Det(nn.Module):
    def __init__(self, backbone, num_modalities, num_tasks, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_modalities = num_modalities
        self.num_tasks = num_tasks
        
        # 网格级MoE层（集成到主干网络中）
        self.moe_layers = self._insert_moe_layers(backbone)
        
        # 多任务检测头
        self.task_heads = nn.ModuleList([
            TaskSpecificHead(backbone.output_channels, num_classes)
            for _ in range(num_tasks)
        ])
        
    def _insert_moe_layers(self, backbone):
        """将MoE层插入到主干网络的特定阶段"""
        moe_layers = nn.ModuleList()
        
        # 在主干网络的最后两个阶段的偶数层插入MoE
        for stage in backbone.stages[-2:]:
            for i, layer in enumerate(stage):
                if i % 2 == 0:  # 偶数层
                    moe_layer = GridLevelMoE(
                        layer.conv1.in_channels, 
                        layer.conv1.out_channels
                    )
                    moe_layers.append(moe_layer)
                    layer.conv1 = moe_layer
        
        return moe_layers
    
    def forward(self, x, modality_idx, task_idx):
        # 提取特征
        features = self.backbone(x)
        
        # 任务特定检测头
        outputs = self.task_heads[task_idx](features)
        
        return outputs
```

## 5. 训练流程

```python
def train_sm3det(model, train_loader, val_loader, config):
    # 初始化优化器
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    dso_optimizer = DynamicSubmoduleOptimizer(
        base_optimizer, 
        num_tasks=model.num_tasks,
        bias=config.dso_bias,
        temperature=config.dso_temperature
    )
    
    for epoch in range(config.epochs):
        model.train()
        
        for batch_idx, batch in enumerate(train_loader):
            # 多模态多任务批次数据
            images, modalities, tasks, targets = batch
            
            total_loss = 0
            task_losses = []
            
            # 处理批次中的每个样本
            for i in range(len(images)):
                image = images[i].unsqueeze(0)
                modality = modalities[i]
                task = tasks[i]
                target = targets[i]
                
                # 前向传播
                output = model(image, modality, task)
                loss = compute_loss(output, target, task)
                
                total_loss += loss
                task_losses.append(loss.item())
            
            # 平均损失
            total_loss /= len(images)
            avg_task_losses = [sum(task_losses) / len(task_losses)]  # 简化
            
            # DSO学习率调整
            dso_optimizer.update_learning_rates(avg_task_losses, batch_idx)
            
            # 反向传播
            base_optimizer.zero_grad()
            total_loss.backward()
            base_optimizer.step()
        
        # 验证
        if epoch % config.val_interval == 0:
            validate(model, val_loader, epoch)
```

## 6. 关键创新总结

1. **网格级MoE**：
   - 局部特征级别的专家选择
   - 同时学习共享和特有表示
   - 动态路由机制

2. **动态子模块优化**：
   - 任务头：平衡收敛速度
   - 主干网络：保证优化方向一致
   - 自适应学习率调整

3. **统一多模态多任务框架**：
   - 支持任意传感器模态
   - 处理多种检测任务格式
   - 无需空间对齐数据

## 7. 性能优势

- **精度提升**：相比单个模型提升1.97 mAP
- **参数效率**：相比基线模型参数更少但性能更好
- **泛化能力强**：适配多种主干网络和检测器
- **扩展性好**：支持更大模型规模持续提升性能

这种设计使得SM3Det成为遥感领域多模态目标检测的强大基础技术，特别适合无人机、卫星等低空经济应用场景。
