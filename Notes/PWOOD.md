

### 一、 核心模块深度解析

论文的基础架构采用了师生模型（Teacher-Student Paradigm），并基于 FCOS（无锚框检测器）、ResNet50 和 FPN 构建 。为了让模型在极度匮乏的标注下学习，作者设计了两个核心组件：

#### 1. OS-Student (方向与尺度感知学生模型)

由于弱标注（如单点）缺乏角度和尺度信息，OS-Student 通过自监督机制来“无中生有”地学习这些特征 ：


**方向学习 (Orientation Learning)**：引入对称性学习（Symmetry-aware）。如果对输入图像进行翻转或旋转，网络的预测结果也必须满足相同的几何映射关系。模型通过最小化原始视图和变换视图预测结果之间的角度损失（$\mathcal{L}_{Ang}^s$）来学习旋转不变性 。



**尺度学习 (Scale Learning)**：引入空间布局学习（Spatial Layout Learning）来约束目标尺度的上下界 。



*上界*：将预测框建模为高斯分布，利用巴氏系数（Bhattacharyya coefficient）最小化不同预测框之间的高斯重叠度 。



*下界*：利用 Voronoi 图和分水岭算法（Watershed），以单点标注为前景，Voronoi 脊线为背景进行图像分割，从而获得目标宽高的回归目标 。





#### 2. CPF (类别无关的伪标签过滤) —— 详细的“选择 (Selection)”机制解析

在非 Transformer 架构（如本篇使用的 FCOS ）中，广义的“Query 选择”体现在**高质量伪标签的动态过滤与分配**上。传统的半监督检测通常依赖固定的静态阈值（如 0.5）来筛选 Teacher 生成的伪标签 。然而，在训练初期，Teacher 能力较弱，固定阈值会导致大量漏检；而在后期又会引入噪声 。

PWOOD 提出了 **CPF (Class-Agnostic Pseudo-Label Filtering)**，这是一种基于数据驱动的动态选择机制 ：


**高斯混合模型 (GMM) 建模**：将 Teacher 输出的所有预测得分 $s$ 视为两个一维高斯分布的混合：正样本分布 $\mathcal{N}_p$ 和负样本分布 $\mathcal{N}_n$ 。



**动态参数初始化**：正样本均值 $\mu_p^{(0)}$ 初始化为当前得分的最大值，负样本均值 $\mu_n^{(0)}$ 初始化为最低得分，两者权重均设为 0.5 。



**EM 算法迭代查询**：通过期望最大化（Expectation-Maximization, EM）算法，推导出后验概率 $\mathcal{P}_p$，即某个预测框被选定为“真实伪目标（Pseudo-object）”的似然概率 。



**自适应阈值截断**：最终的动态过滤阈值 $T_d$ 定义为使正样本后验概率最大化的点：$T_d = \arg\max \mathcal{P}_p(s, \mu_p, \sigma_p^2)$ 。



**这种 Selection 机制的优势**：极大地降低了模型对静态超参数的敏感度。通过 EM 算法动态查询当前特征空间中的得分分布，CPF 能够自适应地分离出高质量的前景 Query（伪标签），从而在添加噪声的实验中展现出极强的鲁棒性 。

---

### 二、 PWOOD 核心伪代码示意 (GitHub Markdown 兼容)

以下伪代码抽象了 PWOOD 的核心训练逻辑，可直接置于 GitHub 的 README 或技术文档中渲染。

```python
import torch
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture # 示意 EM 算法

class PWOOD:
    def __init__(self, student_model, teacher_model):
        self.student = student_model  # OS-Student (FCOS-based)
        self.teacher = teacher_model  # Teacher (FCOS-based)
        self.ema_momentum = 0.999
        
    def orientation_learning_loss(self, preds_original, preds_transformed, transform_type):
        """对称性方向学习 (翻转/旋转一致性)"""
        # preds: 包含 (class, centerness, box, angle)
        angle_ori = preds_original['angle']
        angle_trans = preds_transformed['angle']
        
        if transform_type == 'flip':
            return F.smooth_l1_loss(angle_trans + angle_ori, 0)
        elif transform_type == 'rotate':
            return F.smooth_l1_loss(angle_trans - angle_ori, self.rotation_theta)
            
    def scale_learning_loss(self, preds, points):
        """自监督尺度学习 (Voronoi 分水岭下界 + Bhattacharyya 重叠上界)"""
        # 1. 计算 Voronoi 分水岭生成的伪边界框宽/高
        pseudo_wh = compute_voronoi_watershed(preds, points) 
        loss_lower_bound = gaussian_wasserstein_distance(preds['wh'], pseudo_wh)
        
        # 2. 最小化高斯分布间的重叠 (Bhattacharyya coefficient)
        loss_upper_bound = minimize_bhattacharyya_overlap(preds['gaussian_dist'])
        
        return loss_lower_bound + loss_upper_bound

    def class_agnostic_pseudo_label_filtering(self, teacher_scores):
        """CPF 动态伪标签过滤机制 (核心 Query/Sample 选择)"""
        # 将得分转换为 numpy 数组进行 GMM 拟合
        scores = teacher_scores.detach().cpu().numpy().reshape(-1, 1)
        
        # 初始化双高斯分布: \mu_p 为最大值, \mu_n 为最小值
        init_means = np.array([[scores.min()], [scores.max()]])
        gmm = GaussianMixture(n_components=2, means_init=init_means, max_iter=10)
        
        # EM 算法迭代拟合
        gmm.fit(scores)
        
        # 计算后验概率 P_p
        probas = gmm.predict_proba(scores)[:, 1] # 获取归属正样本分布的概率
        
        # 获取自适应阈值 T_d (最大化 P_p 的决策边界)
        dynamic_threshold = compute_decision_boundary(scores, probas)
        return dynamic_threshold

    def train_step(self, weakly_labeled_data, unlabeled_data):
        # ==================== 1. 弱监督学习分支 (Burn-in 阶段) ====================
        # 输入原始视图和变换视图
        img_ori, img_aug, weak_labels = weakly_labeled_data 
        
        pred_ori = self.student(img_ori)
        pred_aug = self.student(img_aug)
        
        # 计算基础监督损失 (Focal Loss, Smooth L1 等)
        loss_sup = compute_supervised_loss(pred_ori, weak_labels)
        
        # 计算 OS-Student 特有的方向与尺度损失
        loss_ori = self.orientation_learning_loss(pred_ori, pred_aug, transform_type='rotate')
        loss_scale = self.scale_learning_loss(pred_ori, weak_labels['points'])
        
        L_s = loss_sup + 0.2 * loss_ori + 10 * loss_scale['upper'] + 5 * loss_scale['lower']

        # ==================== 2. 无监督半监督分支 ====================
        img_unlabeled_weak_aug, img_unlabeled_strong_aug = unlabeled_data
        
        with torch.no_grad():
            # Teacher 模型提取伪标签 (使用弱数据增强)
            teacher_preds = self.teacher(img_unlabeled_weak_aug)
            teacher_scores = teacher_preds['scores']
            
            # 使用 CPF 动态计算阈值 T_d
            T_d = self.class_agnostic_pseudo_label_filtering(teacher_scores)
            
            # 筛选高质量伪标签
            valid_mask = teacher_scores >= T_d
            pseudo_labels = filter_predictions(teacher_preds, valid_mask)
            
        # Student 模型在强增强数据上进行预测
        student_preds_unlabeled = self.student(img_unlabeled_strong_aug)
        
        # 计算无监督损失 (以高质量伪标签的得分为权重)
        L_u = compute_unsupervised_loss(student_preds_unlabeled, pseudo_labels)

        # ==================== 3. 总体优化与 EMA 更新 ====================
        Total_Loss = L_s + L_u
        Total_Loss.backward()
        optimizer.step()
        
        # 指数移动平均 (EMA) 更新 Teacher 模型参数
        update_ema(self.teacher, self.student, self.ema_momentum)
        
        return Total_Loss

```

### 三、 总结

PWOOD 通过 OS-Student 解耦了繁重的几何标注需求，同时利用基于 EM 算法的 CPF 机制替代了传统的静态阈值过滤。这种设计不仅巧妙地规避了半监督学习中“阈值敏感度高”的通病，而且极大提升了在极低标注比例（如10%）、高噪声情况下的网络泛化性能 。在实现训练免干预（Training-free thresholding）和多模态提示（Point/HBox Prompt）联合训练方面，该思路对后续的开放词汇（Open-Vocabulary）或大模型微调均有较强的启发意义。
