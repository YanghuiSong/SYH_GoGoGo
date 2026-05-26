
### 第一板块：遥感分割 (Remote Sensing Segmentation)
---

<table>
  <thead>
    <tr>
      <th width="40">序号</th>
      <th>论文标题</th>
      <th>作者</th>
      <th width="450">大致解析</th>
      <th width="60">PDF</th>
      <th width="100">代码</th>
    </tr>
  </thead>
  <tbody>
    <tr valign="top">
      <td align="center">1</td>
      <td><b>ReAttnCLIP: Training-Free Open-Vocabulary Remote Sensing Image Segmentation via Re-defined Attention in CLIP</b></td>
      <td>Xin Niu, Manqi Zhao, Dongsheng Jiang, Yingying Wu, Bing Su</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：实现无需训练的开放词汇分割(OVS)，解决CLIP遥感定位弱问题</li>
          <li>⚙️ <b>方法</b>：重设计CLIP注意力机制，增强空间感知与区域-文本对齐</li>
          <li>✨ <b>特点</b>：Training-free、零样本快速部署</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Niu_ReAttnCLIP_Training-Free_Open-Vocabulary_Remote_Sensing_Image_Segmentation_via_Re-defined_Attention_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><sub>暂未公开</sub></td>
    </tr>
    <tr valign="top">
      <td align="center">2</td>
      <td><b>UniGeoSeg: Towards Unified Open-World Segmentation for Geospatial Scenes</b></td>
      <td>Shuo Ni, Di Wang, He Chen, Haonan Guo, Ning Zhang, Jing Zhang</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：构建统一开放世界框架，解决遥感分割任务碎片化问题</li>
          <li>⚙️ <b>方法</b>：构建GeoSeg-1M数据集；提出任务感知文本增强与潜在知识记忆策略</li>
          <li>✨ <b>特点</b>：统一多任务(指代/交互/推理)、指令驱动</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Ni_UniGeoSeg_Towards_Unified_Open-World_Segmentation_for_Geospatial_Scenes_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><a href="https://github.com/MiliLab/UniGeoSeg">GitHub</a></td>
    </tr>
    <tr valign="top">
      <td align="center">3</td>
      <td><b>F2Net: A Frequency-Fused Network for Ultra-High Resolution Remote Sensing Segmentation</b></td>
      <td>Hengzhi Chen, Liqian Feng, Wenhua Wu, Xiaogang Zhu, Qiuxia Wu, Lianlei Shan, Kun Hu</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：解决超高分辨率遥感下采样丢细节与分块割裂上下文问题</li>
          <li>⚙️ <b>方法</b>：高低频双分支(全分辨率细节+长短依赖)；混合频域融合与跨频损失</li>
          <li>✨ <b>特点</b>：频域分解、细节-全局权衡</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_F2Net_A_Frequency-Fused_Network_for_Ultra-High_Resolution_Remote_Sensing_Segmentation_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><sub>待公开</sub></td>
    </tr>
    <tr valign="top">
      <td align="center">4</td>
      <td><b>MM-OVSeg: Multimodal Optical-SAR Fusion for Open-Vocabulary Segmentation in Remote Sensing</b></td>
      <td>Yimin Wei, Aoran Xiao, Hongruixuan Chen, Junshi Xia, Naoto Yokoya</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：解决恶劣天气(云雾)下光学OVS性能退化问题</li>
          <li>⚙️ <b>方法</b>：CMU将SAR对齐到RGB空间；DEF融合CLIP(全局)与DINO(局部)特征</li>
          <li>✨ <b>特点</b>：光学+SAR跨模态、云雾鲁棒</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Wei_MM-OVSeg_Multimodal_Optical-SAR_Fusion_for_Open-Vocabulary_Segmentation_in_Remote_Sensing_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><a href="https://github.com/Jimmyxichen/MM-OVSeg">GitHub</a></td>
    </tr>
    <tr valign="top">
      <td align="center">5</td>
      <td><b>SegEarth-R2: Towards Comprehensive Language-guided Segmentation for Remote Sensing Images</b></td>
      <td>Zepeng Xin, Kaiyu Li, Luodi Chen, Wanchen Li, Xiao Yuchen, Hui Qiao, Weizhan Zhang, Deyu Meng, Xiangyong Cao</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：构建全面语言引导的遥感分割模型，支持多种交互任务</li>
          <li>⚙️ <b>方法</b>：扩展多任务指令数据，强化视觉-语言对齐与空间推理能力</li>
          <li>✨ <b>特点</b>：语言驱动、交互式/推理分割</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Xin_SegEarth-R2_Towards_Comprehensive_Language-guided_Segmentation_for_Remote_Sensing_Images_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><a href="https://github.com/earth-insights/SegEarth-R2">GitHub</a></td>
    </tr>
    <tr valign="top">
      <td align="center">6</td>
      <td><b>BEV-CAR: Enhancing Monocular Bird's Eye View Segmentation with Context-Aware Rasterization</b></td>
      <td>Yixin Xiong, Ke Wang, Tongtong Cheng, Chunhui Liu, Kai Liu</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：提升单目->BEV分割性能，缓解视角变换信息丢失</li>
          <li>⚙️ <b>方法</b>：上下文感知光栅化精准投射特征，结合结构约束与上下文聚合</li>
          <li>✨ <b>特点</b>：单目转BEV、上下文聚合</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Xiong_BEV-CAR_Enhancing_Monocular_Birds_Eye_View_Segmentation_with_Context-Aware_Rasterization_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><sub>暂未公开</sub></td>
    </tr>
    <tr valign="top">
      <td align="center">7</td>
      <td><b>PRUE: A Practical Recipe for Field Boundary Segmentation at Scale</b></td>
      <td>Gedeon Muhawenayo, Caleb Robinson, Subash Khanal, Zhanpei Fang, Isaac Corley, Alexander Wollam, Tianyi Gao, Leonard Strnad, Ryan Avery, Lyndon Estes, Ana Tárano, Nathan Jacobs, Hannah Kerner</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：大规模农田边界分割的落地，解决噪声标签与尺度变化</li>
          <li>⚙️ <b>方法</b>：提出系统性优化配方，涵盖数据构建、模型设计与推理流程</li>
          <li>✨ <b>特点</b>：实用化、大规模农业场景</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Muhawenayo_PRUE_A_Practical_Recipe_for_Field_Boundary_Segmentation_at_Scale_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><sub>暂未公开</sub></td>
    </tr>
    <tr valign="top">
      <td align="center">8</td>
      <td><b>VGGT-Segmentor: Geometry-Enhanced Cross-View Segmentation</b></td>
      <td>Yulu Gao, Bohao Zhang, Zongheng Tang, Jitong Liao, Wenjun Wu, Si Liu</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：解决跨视角(如Ego-Exo)大视角/尺度变化导致的匹配漂移</li>
          <li>⚙️ <b>方法</b>：设计Union Seg Head(融合->点引导->细化)；单图自监督训练</li>
          <li>✨ <b>特点</b>：几何增强、实例级分割</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_VGGT-Segmentor_Geometry-Enhanced_Cross-View_Segmentation_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><a href="https://github.com/buaa-colalab/VGGT-S">GitHub</a></td>
    </tr>
    <tr valign="top">
      <td align="center">9</td>
      <td><b>HySeg: Learning Generative Priors for Structure-Aware Remote Sensing Segmentation</b></td>
      <td>Jie Qiu, Xin Li, Fan Yang, Yan Wang, Dong Yu, Changying Wang, Linwei Dai, Yongxiang Chen, Youqin Chen, Jianzhang Chen</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：引入结构感知与生成先验，提升复杂地物建模能力</li>
          <li>⚙️ <b>方法</b>：通过生成模型学习结构先验并融入分割网络，改善边缘一致性</li>
          <li>✨ <b>特点</b>：生成先验、适合复杂建筑/地块</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Qiu_HySeg_Learning_Generative_Priors_for_Structure-Aware_Remote_Sensing_Segmentation_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><sub>待公开</sub></td>
    </tr>
    <tr valign="top">
      <td align="center">10</td>
      <td><b>CrossEarth-Gate: Fisher-Guided Adaptive Tuning Engine for Efficient Adaptation of Cross-Domain Remote Sensing Semantic Segmentation</b></td>
      <td>Shilei Cao, Ziyang Gong, Hehai Lin, Yang Liu, Jiashun Cheng, Xiaoxing Hu, Haoyuan Liang, Guowen Li, Chengwei Qin, Hong Cheng, Xue Yang, Juepeng Zheng, Haohuan Fu</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：解决跨域(不同区域/传感器/时相)遥感分割高效自适应</li>
          <li>⚙️ <b>方法</b>：利用Fisher信息引导自适应调参引擎，在预训练基础上轻量调优</li>
          <li>✨ <b>特点</b>：Fisher引导、兼容域适应与泛化</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_CrossEarth-Gate_Fisher-Guided_Adaptive_Tuning_Engine_for_Efficient_Adaptation_of_Cross-Domain_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><a href="https://github.com/ShileiCao/CrossEarth-Gate">GitHub</a></td>
    </tr>
    <tr valign="top">
      <td align="center">11</td>
      <td><b>Test-Time Multi-Prompt Adaptation for Open-Vocabulary Remote Sensing Image Segmentation</b></td>
      <td>Ting Yang, Qilong Wang, Qibin Hou, Qinghua Hu</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：提升OVS模型对新类别/新域的即时适应能力</li>
          <li>⚙️ <b>方法</b>：多提示(文本/视觉)测试时适应策略，在线优化无需更新权重</li>
          <li>✨ <b>特点</b>：测试时适应、零回传更新</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Test-Time_Multi-Prompt_Adaptation_for_Open-Vocabulary_Remote_Sensing_Image_Segmentation_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><sub>暂未公开</sub></td>
    </tr>
    <tr valign="top">
      <td align="center">12</td>
      <td><b>Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation</b></td>
      <td>Yunkai Yang, Yudong Zhang, Kunquan Zhang, Jinxiao Zhang, Xinying Chen, Haohuan Fu, Runmin Dong</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：提升长尾/难样本分割性能</li>
          <li>⚙️ <b>方法</b>：基于MM-DiT条件生成模型合成数据；提出CRFM控制轨迹纠正误差</li>
          <li>✨ <b>特点</b>：生成式增广、高质量合成</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Task-Oriented_Data_Synthesis_and_Control-Rectify_Sampling_for_Remote_Sensing_Semantic_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><a href="https://github.com/Yunkai-Yang/crfm">GitHub</a></td>
    </tr>
    <tr valign="top">
      <td align="center">13</td>
      <td><b>SkySense-VITA: Towards Universal In-context Segmentation of Multi-modal Remote Sensing Imagery</b></td>
      <td>Kang Wu, Lei Yu, Junwei Luo, Bo Dang, Junjian Zhang, Xiangyuan Cai, Hongwei Hu, Jingdong Chen, Yansheng Li</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：构建通用上下文分割框架，统一多模态遥感多任务</li>
          <li>⚙️ <b>方法</b>：引入VITA上下文建模机制，统一光学/SAR/多光谱等指令</li>
          <li>✨ <b>特点</b>：多模态统一、指代/交互/推理通用</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_SkySense-VITA_Towards_Universal_In-context_Segmentation_of_Multi-modal_Remote_Sensing_Imagery_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><sub>暂未公开</sub></td>
    </tr>
    <tr valign="top">
      <td align="center">14</td>
      <td><b>CycleBEV: Regularizing View Transformation Networks via View Cycle Consistency for Bird's-Eye-View Semantic Segmentation</b></td>
      <td>Jeongbin Hong, Dooseop Choi, Taeg-Hyun An, Kyounghwan An, Kyoung-Wook Min</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：提升VTN在BEV分割中的稳定性，缓解投影误差</li>
          <li>⚙️ <b>方法</b>：利用视角循环一致性(前向+反向)正则化VTN网络</li>
          <li>✨ <b>特点</b>：BEV分割、视角结构一致</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_CycleBEV_Regularizing_View_Transformation_Networks_via_View_Cycle_Consistency_for_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><a href="https://github.com/JeongbinHong/CycleBEV">GitHub</a></td>
    </tr>
    <tr valign="top">
      <td align="center">15</td>
      <td><b>ReSAM: Refine, Requery, and Reinforce: Self-Prompting Point-Supervised Segmentation for Remote Sensing Images</b></td>
      <td>Muhammad Naseer Subhani</td>
      <td>
        <ul>
          <li>🎯 <b>目标</b>：点级弱监督下实现高质量分割，减少全量掩膜依赖</li>
          <li>⚙️ <b>方法</b>：基于SAM设计Refine-Requery-Reinforce自提示循环迭代细化</li>
          <li>✨ <b>特点</b>：点监督、SAM-based交互</li>
        </ul>
      </td>
      <td align="center"><a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Subhani_ReSAM_Refine_Requery_and_Reinforce_Self-Prompting_Point-Supervised_Segmentation_for_Remote_CVPR_2026_paper.pdf">PDF</a></td>
      <td align="center"><sub>暂未公开</sub></td>
    </tr>
  </tbody>
</table>

---


### 第二板块：OVSS (Open-Vocabulary Semantic Segmentation)
---
| 序号 | 论文标题 | 作者 | 大致解析 | PDF链接 | 代码链接（如有） |
| :---: | :--- | :--- | :--- | :--- | :--- |
| 1 | **Retrieve and Segment: Are a Few Examples Enough to Bridge the Supervision Gap in Open-Vocabulary Segmentation?** | Tilemachos Aravanis, Vladan Stojnić, Bill Psomas, Nikos Komodakis, Giorgos Tolias | **目标**：缩小零样本开放词汇分割与全监督方法之间的监督差距，仅用少量像素级示例图像。<br>**方法**：提出检索增强的测试时适配器，利用少样本带标注图像检索并适配 VLM，缓解语义模糊与监督不足问题。<br>**特点**：检索增强、少样本适配、缩小零样本-全监督差距。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Aravanis_Retrieve_and_Segment_Are_a_Few_Examples_Enough_to_Bridge_CVPR_2026_paper.pdf) | [GitHub](https://github.com/TilemahosAravanis/Retrieve-and-Segment) |
| 2 | **Direct Segmentation without Logits Optimization for Training-Free Open-Vocabulary Semantic Segmentation** | Jiahao Li, Yang Lu, Yachao Zhang, Fangyong Wang, Yuan Xie, Yanyun Qu | **目标**：在无需训练和 logits 优化的前提下，实现更直接、高效的开放词汇语义分割。<br>**方法**：提出 DSLO 框架，绕过传统 logits 优化，直接从 VLM 特征+文本嵌入构建分割掩码，显著简化训练-free 流程。<br>**特点**：Training-free、无需 logits 优化、流程更直接。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Direct_Segmentation_without_Logits_Optimization_for_Training-Free_Open-Vocabulary_Semantic_Segmentation_CVPR_2026_paper.pdf) | [GitHub](https://github.com/liblacklucy/DSLO) |
| 3 | **Mitigating Objectness Bias and Region-to-Text Misalignment for Open-Vocabulary Panoptic Segmentation** | Nikolay Kormushev, Josip Šarić, Matej Kristan | **目标**：解决开放词汇全景分割中 CLIP 的“物体性偏差”和掩码-文本错位问题。<br>**方法**：提出 OVRCOAT，通过物体性调整和掩码到文本对齐精炼，校正区域-文本关联，提升全景分割一致性。<br>**特点**：物体性去偏、掩码-文本对齐、开放词汇全景分割。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kormushev_Mitigating_Objectness_Bias_and_Region-to-Text_Misalignment_for_Open-Vocabulary_Panoptic_Segmentation_CVPR_2026_paper.pdf) | [GitHub](https://github.com/nickormushev/OVRCOAT) |
| 4 | **HOPS: Hierarchical Open-vocabulary Part Segmentation with Attention-Aware Filtering and Affinity-Guided Enhancement** | Xinlong Li, Di Lin, Shaoyiyi Gao, Yaxuan Liu, Jixian He, Jiaxin Li, Ruonan Liu, Qing Guo, Kairui Yang, Wei Feng | **目标**：实现层次化的开放词汇部件分割，兼顾粗-细语义与空间结构。<br>**方法**：提出 HOPS，通过注意力感知过滤抑制噪声，亲和度引导增强增强部件内聚性，构建层次化分割与推理。<br>**特点**：层次化部件分割、注意力过滤、亲和度增强。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_HOPS_Hierarchical_Open-vocabulary_Part_Segmentation_with_Attention-Aware_Filtering_and_Affinity-Guided_CVPR_2026_paper.pdf) | 暂未公开（截至检索未见官方代码） |
| 5 | **PCA-Seg: Revisiting Cost Aggregation for Open-Vocabulary Semantic and Part Segmentation** | Jianjian Yin, Tao Chen, Yi Chen, Gensheng Pei, Xiangbo Shu, Yazhou Yao, Fumin Shen | **目标**：重新审视代价聚合在开放词汇语义/部件分割中的作用，提升细粒度对齐质量。<br>**方法**：提出 PCA-Seg，设计细粒度图像-文本对应代价聚合模块，增强对部件/语义边界的区分能力，支持语义+部件统一分割。<br>**特点**：代价聚合重设计、细粒度对齐、语义+部件分割。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yin_PCA-Seg_Revisiting_Cost_Aggregation_for_Open-Vocabulary_Semantic_and_Part_Segmentation_CVPR_2026_paper.pdf) | [GitHub](https://github.com/NUST-Machine-Intelligence-Laboratory/PCA-Seg) |
| 6 | **Training-Free Open-Vocabulary Camouflaged Object Segmentation via Fine-Grained Object Binding and Adaptive Hybrid Prompt** | Peng Ren, Cheng Jiang, Chuande Yang, Fuming Sun, Tian Bai | **目标**：在无需训练的开放词汇设定下实现伪装目标分割（COD），解决细粒度目标绑定与提示设计问题。<br>**方法**：提出细粒度对象绑定与自适应混合提示策略，将 VLM 特征与 COD 先验结合，无需额外训练即可分割伪装目标。<br>**特点**：Training-free、开放词汇 COD、细粒度绑定+混合提示。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ren_Training-Free_Open-Vocabulary_Camouflaged_Object_Segmentation_via_Fine-Grained_Object_Binding_and_CVPR_2026_paper.pdf) | 暂未公开（CVPR 2026 虚拟页面暂无代码链接） |
| 7 | **SPAR: Single-Pass Any-Resolution ViT for Open-vocabulary Segmentation** | Naomi Kombol, Ivan Martinović, Siniša Šegvić, Giorgos Tolias | **目标**：实现任意分辨率、单次前向的开放词汇分割，提升推理效率与分辨率适应能力。<br>**方法**：提出 SPAR，基于 ViT 设计单次前向任意分辨率分割框架，避免多裁剪/多尺度重复计算，兼顾效率与精度。<br>**特点**：Any-resolution、单次前向、高效开放词汇分割。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kombol_SPAR_Single-Pass_Any-Resolution_ViT_for_Open-vocabulary_Segmentation_CVPR_2026_paper.pdf) | [GitHub](https://github.com/naomikombol/SPAR) |
| 8 | **Seeing Both Sides: Towards Bidirectional Semantic Alignment for Open-Vocabulary Camouflaged Object Segmentation** | Guohui Zhang, Fuming Sun, Yu Zhao, Yuqiu Kong, Jing Sun, Fasheng Wang | **目标**：解决开放词汇伪装目标分割中的语义对齐不充分问题，实现双向对齐（视觉→文本 & 文本→视觉）。<br>**方法**：提出双向语义对齐框架，同时约束视觉特征到文本、文本到视觉的映射，提升伪装目标的开放词汇识别与分割。<br>**特点**：双向语义对齐、开放词汇 COD。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Seeing_Both_Sides_Towards_Bidirectional_Semantic_Alignment_for_Open-Vocabulary_Camouflaged_CVPR_2026_paper.pdf) | 暂未公开（截至检索未见官方代码） |
| 9 | **Open-Vocabulary Domain Generalization in Urban-Scene Segmentation** | Dong Zhao, Qi Zang, Nan Pu, Wenjing Li, Nicu Sebe, Zhun Zhong | **目标**：在开放词汇设定下实现城市场景分割的域泛化，应对未见域/未见类别。<br>**方法**：提出 OVDG-SS 与 S2-Corr 机制，校正域漂移导致的文本-图像关联扭曲，提升跨域+跨类别泛化能力。<br>**特点**：开放词汇+域泛化、S2-Corr 校正关联。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_Open-Vocabulary_Domain_Generalization_in_Urban-Scene_Segmentation_CVPR_2026_paper.pdf) | [GitHub](https://github.com/DZhaoXd/s2_corr) |
| 10 | **The Power of Prior: Training-Free Open-Vocabulary Semantic Segmentation with LLaVA** | Bingfeng Zhang, Siyue Yu, Hui Li, Jiahua Lin, Wenwu Wang, Jimin Xiao | **目标**：利用 LLaVA 等大模型先验，实现无需训练的开放词汇语义分割。<br>**方法**：提出 FSeg-LLaVA，将 LLaVA 的先验知识转化为分割监督，通过先验引导的分割头实现 training-free OVS。<br>**特点**：LLaVA 先验、Training-free、零样本分割。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_The_Power_of_Prior_Training-Free_Open-Vocabulary_Semantic_Segmentation_with_LLaVA_CVPR_2026_paper.pdf) | [GitHub](https://github.com/zbf1991/FSeg-LLaVA) |
| 11 | **Looking Beyond the Window: Global-Local Aligned CLIP for Training-free Open-Vocabulary Semantic Segmentation** | ByeongCheol Lee, Hyun Seok Seong, Sangeek Hyun, Gilhan Park, WonJun Moon, Jae-Pil Heo | **目标**：解决 CLIP 局部窗口导致的上下文不足问题，实现全局-局部对齐的 training-free OVS。<br>**方法**：提出 GLA-CLIP，通过全局-局部对齐机制增强窗口内外特征一致性，提升开放词汇分割的全局语义与边界精度。<br>**特点**：全局-局部对齐、Training-free、CLIP 增强。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_Looking_Beyond_the_Window_Global-Local_Aligned_CLIP_for_Training-free_Open-Vocabulary_CVPR_2026_paper.pdf) | [GitHub](https://github.com/2btlFe/GLA-CLIP) |
| 12 | **PEARL: Geometry Aligns Semantics for Training-Free Open-Vocabulary Semantic Segmentation** | Gensheng Pei, Xiruo Jiang, Xinhao Cai, Tao Chen, Yazhou Yao, Byeungwoo Jeon | **目标**：利用几何约束对齐语义，实现无需训练的开放词汇语义分割。<br>**方法**：提出 PEARL，通过 Procrustes 对齐 + 文本感知拉普拉斯传播，先对齐再传播，将几何结构引入语义分割。<br>**特点**：几何-语义对齐、Align-then-Propagate、Training-free。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Pei_PEARL_Geometry_Aligns_Semantics_for_Training-Free_Open-Vocabulary_Semantic_Segmentation_CVPR_2026_paper.pdf) | [GitHub](https://github.com/PGSmall/PEARL) |
| 13 | **S2C2Seg: Semantic-Spatial Consistency and Category Optimization for Open-Vocabulary Segmentation** | Yuhao Qing, Yueying Wang, Chaoyang Chen, Weidong Zhang, Jie Wen, Xin Xu | **目标**：缓解开放词汇分割中的语义-空间不一致与类别混淆问题。<br>**方法**：提出 S2C2Seg，通过语义-空间一致性约束和类别优化模块，减少空间混淆与类别误判，提升分割可靠性。<br>**特点**：语义-空间一致性、类别优化、减少空间混淆。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Qing_S2C2Seg_Semantic-Spatial_Consistency_and_Category_Optimization_for_Open-Vocabulary_Segmentation_CVPR_2026_paper.pdf) | 暂未公开（截至检索未见官方代码） |
| 14 | **GeoGuide: Hierarchical Geometric Guidance for Open-Vocabulary 3D Semantic Segmentation** | Xujing Tao, Chuxin Wang, Yubo Ai, Zhixin Cheng, Zhuoyuan Li, Liangsheng Liu, Yujia Chen, Xinjun Li, Qiao Li, Wenfei Yang, Tianzhu Zhang | **目标**：将层次化几何先验引入开放词汇 3D 语义分割，提升 3D 场理解精度。<br>**方法**：提出 GeoGuide，利用层次化几何引导（如法线、曲率、空间关系）约束 3D 开放词汇分割，改善几何结构与语义一致性。<br>**特点**：层次化几何引导、开放词汇 3D 分割。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Tao_GeoGuide_Hierarchical_Geometric_Guidance_for_Open-Vocabulary_3D_Semantic_Segmentation_CVPR_2026_paper.pdf) | 暂未公开（作者主页标注“请求代码”，暂未发布公开仓库） |
| 15 | **CoSMo3D: Open-World Promptable 3D Semantic Segmentation through LLM-Guided Canonical Spatial Modeling** | Li Jin, Weikai Chen, Yujie Wang, Yingda Yin, Zeyu Hu, Runze Zhang, Keyang Luo, Shengju Qian, Xin Wang, Xueying Qin | **目标**：实现开放世界、可提示的 3D 语义分割，通过 LLM 引导的典型空间建模统一 3D 开放词汇分割。<br>**方法**：提出 CoSMo3D，利用 LLM 引导的典型空间建模将 3D 场景规范到统一空间，支持开放类别与提示式分割。<br>**特点**：LLM 引导、开放世界、可提示 3D 分割。 | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Jin_CoSMo3D_Open-World_Promptable_3D_Semantic_Segmentation_through_LLM-Guided_Canonical_Spatial_CVPR_2026_paper.pdf) | [GitHub](https://github.com/JinLi998/CoSMo3D) |
---



### 第三板块：其他论文 (医学、视频、3D、弱监督、通用分割等)

以下是完整的第三板块表格，包含全部 162 篇论文及其 PDF 链接。

---

### 第三板块：其他论文 (医学、视频、3D、弱监督、通用分割等)

| 序号 | 论文标题 | 作者 | PDF链接 |
| :--- | :--- | :--- | :--- |
| 1 | **AD-GBC: Anisotropic Granular-Ball Skip-Connection Refiner for UNet-Based Medical Image Segmentation** | Xiya Shen, Qinglin Zhao, Li Feng | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_AD-GBC_Anisotropic_Granular-Ball_Skip-Connection_Refiner_for_UNet-Based_Medical_Image_Segmentation_CVPR_2026_paper.pdf) |
| 2 | **Boundary-Responsive Differentiable Gating for Superpixel-Based Segmentation** | Fatmaelzahraa Ahmed, Zhihe Lu, Gianni Caro, Diram Tabaa, Mohamed Hamdy, Muraam Abdel-Ghani, Abdulaziz Al-Ali, Muhammad Arsalan, Shidin Balakrishnan | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ahmed_Boundary-Responsive_Differentiable_Gating_for_Superpixel-Based_Segmentation_CVPR_2026_paper.pdf) |
| 3 | **Towards Robust Multi-Modal Semantic Segmentation with Teacher-Student Framework and Hybrid Prototype Distillation** | Jiaqi Tan, Xu Zheng, Yang Liu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Tan_Towards_Robust_Multi-Modal_Semantic_Segmentation_with_Teacher-Student_Framework_and_Hybrid_CVPR_2026_paper.pdf) |
| 4 | **SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation** | Xiaogang Du, Jiawei Zhang, Tongfei Liu, Tao Lei, Yingbo Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Du_SPEGC_Continual_Test-Time_Adaptation_via_Semantic-Prompt-Enhanced_Graph_Clustering_for_Medical_CVPR_2026_paper.pdf) |
| 5 | **OSA: Echocardiography Video Segmentation via Orthogonalized State Update and Anatomical Prior-aware Feature Enhancement** | Rui Wang, Huisi Wu, Jing Qin | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_OSA_Echocardiography_Video_Segmentation_via_Orthogonalized_State_Update_and_Anatomical_CVPR_2026_paper.pdf) |
| 6 | **The Missing Point in Vision Transformers for Universal Image Segmentation** | Sajjad Shahabodini, Mobina Mansoori, Farnoush Bayatmakou, Jamshid Abouei, Konstantinos Plataniotis, Arash Mohammadi | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shahabodini_The_Missing_Point_in_Vision_Transformers_for_Universal_Image_Segmentation_CVPR_2026_paper.pdf) |
| 7 | **CaptionFormer: Unified Segmentation, Tracking, and Captioning for Spatio-Temporal Objects** | Gabriel Fiastre, Antoine Yang, Cordelia Schmid | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Fiastre_CaptionFormer_Unified_Segmentation_Tracking_and_Captioning_for_Spatio-Temporal_Objects_CVPR_2026_paper.pdf) |
| 8 | **From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings** | Jiajie Zhang, Sören Schwertfeger, Alexander Kleiner | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_From_Observation_to_Action_Latent_Action-based_Primitive_Segmentation_for_VLA_CVPR_2026_paper.pdf) |
| 9 | **Delving Aleatoric Uncertainty in Medical Image Segmentation via Vision Foundation Models** | Ruiyang Li, Fang Liu, Licheng Jiao, Xinglin Xie, Jiayao Hao, Shuo Li, Xu Liu, Jingyi Yang, Lingling Li, Puhua Chen, Wenping Ma | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Delving_Aleatoric_Uncertainty_in_Medical_Image_Segmentation_via_Vision_Foundation_CVPR_2026_paper.pdf) |
| 10 | **PanDA: Unsupervised Domain Adaptation for Multimodal 3D Panoptic Segmentation in Autonomous Driving** | Yining Pan, Shijie Li, Yuchen Wu, Xulei Yang, Na Zhao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Pan_PanDA_Unsupervised_Domain_Adaptation_for_Multimodal_3D_Panoptic_Segmentation_in_CVPR_2026_paper.pdf) |
| 11 | **Moving Border Ownership for Event-based Motion Segmentation** | Zhiyuan Hua, Cornelia Fermüller, Yiannis Aloimonos | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Hua_Moving_Border_Ownership_for_Event-based_Motion_Segmentation_CVPR_2026_paper.pdf) |
| 12 | **CLP: A Real-World Dataset of Contaminated Lens Protectors for Robust Semantic Segmentation** | Sungyong Park, Sooyoung Choi, Hyunsuh Koh, Youngjae Choi, Heewon Kim | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Park_CLP_A_Real-World_Dataset_of_Contaminated_Lens_Protectors_for_Robust_CVPR_2026_paper.pdf) |
| 13 | **ST4R-Splat: Spatio-Temporal Referring Segmentation in 4D Gaussian Splatting** | Yuming Meng, Dong Wu, Hongbin Zha | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Meng_ST4R-Splat_Spatio-Temporal_Referring_Segmentation_in_4D_Gaussian_Splatting_CVPR_2026_paper.pdf) |
| 14 | **Harmonized Feature Conditioning and Frequency-Prompt Personalization for Multi-Rater Medical Segmentation** | Sanaz Karimijafarbigloo, Armin Khosravi, Alireza Kheyrkhah, Reza Azad, Mauricio Reyes, Dorit Merhof | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Karimijafarbigloo_Harmonized_Feature_Conditioning_and_Frequency-Prompt_Personalization_for_Multi-Rater_Medical_Segmentation_CVPR_2026_paper.pdf) |
| 15 | **Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models** | Nimrod Berman, Adam Botach, Emanuel Ben-Baruch, Shunit Haviv Hakimi, Asaf Gendler, Ilan Naiman, Erez Yosef, Igor Kviatkovsky | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Berman_Scene-VLM_Multimodal_Video_Scene_Segmentation_via_Vision-Language_Models_CVPR_2026_paper.pdf) |
| 16 | **CrackSSM: Reviving SSMs for Crack Segmentation via Dynamic Scanning** | Yubin Gu, Boyang Hou, Yuan Meng, Wenting Luo, Jiayi Ji, Xiaoshuai Sun | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gu_CrackSSM_Reviving_SSMs_for_Crack_Segmentation_via_Dynamic_Scanning_CVPR_2026_paper.pdf) |
| 17 | **SAMIX: Reinforcing SAM2 with Semantic Adapter and Reference Selecting Policy for Mix-Supervised Segmentation** | Qiang Hu, Jiajie Wei, Zhenyu Yi, Zhifen Yan, Yingjie Guo, Hongkuan Shi, Ge-Peng Ji, Qiang Li, Zhiwei Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Hu_SAMIX_Reinforcing_SAM2_with_Semantic_Adapter_and_Reference_Selecting_Policy_CVPR_2026_paper.pdf) |
| 18 | **Uni-Encoder Meets Multi-Encoders: Representation Before Fusion for Brain Tumor Segmentation with Missing Modalities** | Peibo Song, Xiaotian Xue, Jinshuo Zhang, Zihao Wang, Jinhua Liu, Shujun Fu, Fangxun Bao, Si Yong Yeo | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Song_Uni-Encoder_Meets_Multi-Encoders_Representation_Before_Fusion_for_Brain_Tumor_Segmentation_CVPR_2026_paper.pdf) |
| 19 | **MV3DIS: Multi-View Mask Matching via 3D Guides for Zero-Shot 3D Instance Segmentation** | Yibo Zhao, Yigong Zhang, Jin Xie | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_MV3DIS_Multi-View_Mask_Matching_via_3D_Guides_for_Zero-Shot_3D_CVPR_2026_paper.pdf) |
| 20 | **Rewis3d: Reconstruction Improves Weakly-Supervised Semantic Segmentation** | Jonas Ernst, Wolfgang Boettcher, Lukas Hoyer, Jan Eric Lenssen, Bernt Schiele | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ernst_Rewis3d_Reconstruction_Improves_Weakly-Supervised_Semantic_Segmentation_CVPR_2026_paper.pdf) |
| 21 | **IBISAgent: Reinforcing Pixel-Level Visual Reasoning in MLLMs for Universal Biomedical Object Referring and Segmentation** | Yankai Jiang, Qiaoru Li, Binlu Xu, Haoran Sun, Chao Ding, Junting Dong, Yuxiang Cai, Xuhong Zhang, Jianwei Yin | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Jiang_IBISAgent_Reinforcing_Pixel-Level_Visual_Reasoning_in_MLLMs_for_Universal_Biomedical_CVPR_2026_paper.pdf) |
| 22 | **Virtual Nodes Guided Dynamic Graph Neural Network for Brain Tumor Segmentation with Missing Modalities** | Sha Tao, Jiao Pan, Yu Guo, Chao Yao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Tao_Virtual_Nodes_Guided_Dynamic_Graph_Neural_Network_for_Brain_Tumor_CVPR_2026_paper.pdf) |
| 23 | **Spatial Matters: Position-Guided 3D Referring Expression Segmentation** | Yabing Wang, Zhuotao Tian, Le Wang, Zheng Qin, Sanping Zhou | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Spatial_Matters_Position-Guided_3D_Referring_Expression_Segmentation_CVPR_2026_paper.pdf) |
| 24 | **GeoFree-CoSeg: Unsupervised Point Cloud-Image Cross-Modal Co-Segmentation Without Geometric Alignment** | Xin Duan, Xiabi Liu, Liyuan Pan | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Duan_GeoFree-CoSeg_Unsupervised_Point_Cloud-Image_Cross-Modal_Co-Segmentation_Without_Geometric_Alignment_CVPR_2026_paper.pdf) |
| 25 | **VoxTell: Free-Text Promptable Universal 3D Medical Image Segmentation** | Maximilian Rokuss, Moritz Langenberg, Yannick Kirchhoff, Fabian Isensee, Benjamin Hamm, Constantin Ulrich, Sebastian Regnery, Lukas Bauer, Efthimios Katsigiannopulos, Tobias Norajitra, Klaus Maier-Hein | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Rokuss_VoxTell_Free-Text_Promptable_Universal_3D_Medical_Image_Segmentation_CVPR_2026_paper.pdf) |
| 26 | **TerraSeg: Self-Supervised Ground Segmentation for Any LiDAR** | Ted Lentsch, Santiago Montiel-Marín, Holger Caesar, Dariu M. Gavrila | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lentsch_TerraSeg_Self-Supervised_Ground_Segmentation_for_Any_LiDAR_CVPR_2026_paper.pdf) |
| 27 | **Rethinking Box Supervision: Bias-Free Weakly Supervised Medical Segmentation** | Jun Wei, Hui Huang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wei_Rethinking_Box_Supervision_Bias-Free_Weakly_Supervised_Medical_Segmentation_CVPR_2026_paper.pdf) |
| 28 | **Exploring the Underwater World Segmentation without Extra Training** | Bingyu Li, Tao Huo, Da Zhang, Zhiyuan Zhao, Junyu Gao, Xuelong Li | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Exploring_the_Underwater_World_Segmentation_without_Extra_Training_CVPR_2026_paper.pdf) |
| 29 | **Learning and Aligning Click-Aware Shape Prior for Interactive Amodal Instance Segmentation** | Junjie Chen, Junwei Lin, Ren Hong, Shengjie Liu, Yuming Fang, Feng Qian, Yifan Zuo | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Learning_and_Aligning_Click-Aware_Shape_Prior_for_Interactive_Amodal_Instance_CVPR_2026_paper.pdf) |
| 30 | **InterRVOS: Interaction-Aware Referring Video Object Segmentation** | Woojeong Jin, Seongchan Kim, Jaeho Lee, Seungryong Kim | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Jin_InterRVOS_Interaction-Aware_Referring_Video_Object_Segmentation_CVPR_2026_paper.pdf) |
| 31 | **Bootstrapping Video Semantic Segmentation Model via Distillation-assisted Test-Time Adaptation** | Jihun Kim, Hoyong Kwon, Hyeokjun Kweon, Kuk-Jin Yoon | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Bootstrapping_Video_Semantic_Segmentation_Model_via_Distillation-assisted_Test-Time_Adaptation_CVPR_2026_paper.pdf) |
| 32 | **Cross-Domain Few-Shot Segmentation via Multi-view Progressive Adaptation** | Jiahao Nie, Guanqiao Fu, Wenbin An, Yap-Peng Tan, Alex C. Kot, Shijian Lu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Nie_Cross-Domain_Few-Shot_Segmentation_via_Multi-view_Progressive_Adaptation_CVPR_2026_paper.pdf) |
| 33 | **PGR-Net: Prior-Guided ROI Reasoning Network for Brain Tumor MRI Segmentation** | Jiacheng Lu, Hui Ding, Shiyu Zhang, Guoping Huo | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_PGR-Net_Prior-Guided_ROI_Reasoning_Network_for_Brain_Tumor_MRI_Segmentation_CVPR_2026_paper.pdf) |
| 34 | **GenMask: Adapting DiT for Segmentation via Direct Mask Generation** | Yuhuan Yang, Xianwei Zhuang, Yuxuan Cai, Chaofan Ma, Shuai Bai, Jiangchao Yao, Ya Zhang, Junyang Lin, Yanfeng Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_GenMask_Adapting_DiT_for_Segmentation_via_Direct_Mask_Generation_CVPR_2026_paper.pdf) |
| 35 | **Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation** | Ziqian Yang, Xianglin Qiu, Xinqiao Zhao, Xiaolei Wang, Quan Zhang, Jimin Xiao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Frequency-Aware_Affinity_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2026_paper.pdf) |
| 36 | **Efficient Video Object Segmentation and Tracking with Recurrent Dynamic Submodel** | Weidong Tang, Zhiyuan Liang, Xinyan Wan, Chen Zhu, Zhaopan Xu, Pengfei Zhou, Yan Song, Yang You, Wangbo Zhao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Tang_Efficient_Video_Object_Segmentation_and_Tracking_with_Recurrent_Dynamic_Submodel_CVPR_2026_paper.pdf) |
| 37 | **MixerCSeg: An Efficient Mixer Architecture for Crack Segmentation via Decoupled Mamba Attention** | Zilong Zhao, Zhengming Ding, Pei Niu, Wenhao Sun, Feng Guo | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_MixerCSeg_An_Efficient_Mixer_Architecture_for_Crack_Segmentation_via_Decoupled_CVPR_2026_paper.pdf) |
| 38 | **GeCo: Geometry-Consistent Regularization for Domain Generalized Semantic Segmentation** | Qi Zang, Dong Zhao, Nan Pu, Wenjing Li, Zhun Zhong, Meng Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zang_GeCo_Geometry-Consistent_Regularization_for_Domain_Generalized_Semantic_Segmentation_CVPR_2026_paper.pdf) |
| 39 | **REALM: An MLLM-Agent Framework for Open World 3D Reasoning Segmentation and Editing on Gaussian Splatting** | Changyue Shi, Minghao Chen, Yiping Mao, Chuxiao Yang, Xinyuan Hu, Jiajun Ding, Zhou Yu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shi_REALM_An_MLLM-Agent_Framework_for_Open_World_3D_Reasoning_Segmentation_CVPR_2026_paper.pdf) |
| 40 | **CDICS: Delving Into Fine-Grained Attribute for In-Context Segmentation via Compositional Prompts and Phased Decoupling** | Zhiyu Li, Dianmo Sheng, Qi Chu, Shilong Chen, Tao Gong, Zhou Wei, Nenghai Yu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_CDICS_Delving_Into_Fine-Grained_Attribute_for_In-Context_Segmentation_via_Compositional_CVPR_2026_paper.pdf) |
| 41 | **Learning to Identify Out-of-Distribution Objects for 3D LiDAR Anomaly Segmentation** | Simone Mosco, Daniel Fusaro, Alberto Pretto | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Mosco_Learning_to_Identify_Out-of-Distribution_Objects_for_3D_LiDAR_Anomaly_Segmentation_CVPR_2026_paper.pdf) |
| 42 | **Bridging RGB and Hematoxylin Components: An Interleaved Guidance and Fusion Framework for Point Supervised Nuclei Segmentation** | Zihan Huan, Xipeng Pan, Hualong Zhang, Siyang Feng, Rushi Lan, Huadeng Wang, Haoxiang Lu, Zhenbing Liu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huan_Bridging_RGB_and_Hematoxylin_Components_An_Interleaved_Guidance_and_Fusion_CVPR_2026_paper.pdf) |
| 43 | **SegMoTE: Token-Level Mixture of Experts for Medical Image Segmentation** | Yujie Lu, Jingwen Li, Sibo Ju, Yanzhou Su, He Yao, Yisong Liu, Min Zhu, Junlong Cheng | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_SegMoTE_Token-Level_Mixture_of_Experts_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf) |
| 44 | **ReScene4D: Temporally Consistent Semantic Instance Segmentation of Evolving Indoor 3D Scenes** | Emily Steiner, Jianhao Zheng, Henry Howard-Jenkins, Chris Xie, Iro Armeni | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Steiner_ReScene4D_Temporally_Consistent_Semantic_Instance_Segmentation_of_Evolving_Indoor_3D_CVPR_2026_paper.pdf) |
| 45 | **Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token** | Anqi Zhang, Xiaokang Ji, Guangyu Gao, Jianbo Jiao, Chi Harold Liu, Yunchao Wei | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Rethinking_MLLM_Itself_as_a_Segmenter_with_a_Single_Segmentation_CVPR_2026_paper.pdf) |
| 46 | **Reinforcing Video Object Segmentation to Think before it Segments** | Sitong Gong, Yunzhi Zhuge, Lu Zhang, Jiazuo Yu, Pingping Zhang, Xu Jia, Huchuan Lu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gong_Reinforcing_Video_Object_Segmentation_to_Think_before_it_Segments_CVPR_2026_paper.pdf) |
| 47 | **Scene-Centric Unsupervised Video Panoptic Segmentation** | Christoph Reich, Oliver Hahn, Nikita Araslanov, Laura Leal-Taixé, Christian Rupprecht, Daniel Cremers, Stefan Roth | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Reich_Scene-Centric_Unsupervised_Video_Panoptic_Segmentation_CVPR_2026_paper.pdf) |
| 48 | **B$^3$-Seg: Camera-Free, Training-Free 3DGS Segmentation via Analytic EIG and Beta-Bernoulli Bayesian Updates** | Hiromichi Kamata, Samuel Arthur Munro, Fuminori Homma | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kamata_B3-Seg_Camera-Free_Training-Free_3DGS_Segmentation_via_Analytic_EIG_and_Beta-Bernoulli_CVPR_2026_paper.pdf) |
| 49 | **Spatial-SAM: Spatially Consistent 3D Electron Microscopy Segmentation with SDF Memory and Semi-Supervised Learning** | Yikai Huang, Renmin Han, Yuxuan Wang, Youcheng Cai, Ligang Liu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Spatial-SAM_Spatially_Consistent_3D_Electron_Microscopy_Segmentation_with_SDF_Memory_CVPR_2026_paper.pdf) |
| 50 | **BiPA: Bilevel Prompt Adaptation for Underwater Instance Segmentation** | Long Ma, Haoze Zheng, Yuhang Mao, Jinyuan Liu, Chengpei Xu, Xinwei Xue, Yi Wang, Xiangjian He, Weimin Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_BiPA_Bilevel_Prompt_Adaptation_for_Underwater_Instance_Segmentation_CVPR_2026_paper.pdf) |
| 51 | **BackSplit: The Importance of Sub-dividing the Background in Biomedical Lesion Segmentation** | Rachit Saluja, Asli Cihangir, Ruining Deng, Johannes C. Paetzold, Fengbei Liu, Mert R. Sabuncu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Saluja_BackSplit_The_Importance_of_Sub-dividing_the_Background_in_Biomedical_Lesion_CVPR_2026_paper.pdf) |
| 52 | **R2-Seg: Training-Free OOD Medical Tumor Segmentation via Anatomical Reasoning and Statistical Rejection** | Shuaike Shen, Ke Liu, Jiaqing Xie, Shangde Gao, Chunhua Shen, Ge Liu, Mireia Crispin-Ortuzar, Shangqi Gao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_R2-Seg_Training-Free_OOD_Medical_Tumor_Segmentation_via_Anatomical_Reasoning_and_CVPR_2026_paper.pdf) |
| 53 | **CompetitorFormer: Mitigating Query Conflicts for 3D Instance Segmentation via Competitive Strategy** | Duanchu Wang, Junjie Yang, Haoran Gong, Jing Liu, Di Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_CompetitorFormer_Mitigating_Query_Conflicts_for_3D_Instance_Segmentation_via_Competitive_CVPR_2026_paper.pdf) |
| 54 | **Hilbert Curve-Based Attention Enabling Topology-Preserving Image Tensor Representation for Semantic Segmentation Network** | Linkang Xu, Gang Li, Yue Song, Xiangxin Ji | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_Hilbert_Curve-Based_Attention_Enabling_Topology-Preserving_Image_Tensor_Representation_for_Semantic_CVPR_2026_paper.pdf) |
| 55 | **Joint Spectral Image Reconstruction and Semantic Segmentation with Cooperative Unfolding** | Zijun He, Ping Wang, Xiaodong Wang, Chang Chen, Xin Yuan | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/He_Joint_Spectral_Image_Reconstruction_and_Semantic_Segmentation_with_Cooperative_Unfolding_CVPR_2026_paper.pdf) |
| 56 | **Spectral Scalpel: Amplifying Adjacent Action Discrepancy via Frequency-Selective Filtering for Skeleton-Based Action Segmentation** | Haoyu Ji, Bowen Chen, Zhihao Yang, Wenze Huang, Yu Gao, Xueting Liu, Weihong Ren, Zhiyong Wang, Honghai Liu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ji_Spectral_Scalpel_Amplifying_Adjacent_Action_Discrepancy_via_Frequency-Selective_Filtering_for_CVPR_2026_paper.pdf) |
| 57 | **SemiGDA: Generative Dual-distribution Alignment for Semi-Supervised Medical Image Segmentation** | Kaiwen Huang, Yi Zhou, Yizhe Zhang, Jingxiong Li, Tao Zhou | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_SemiGDA_Generative_Dual-distribution_Alignment_for_Semi-Supervised_Medical_Image_Segmentation_CVPR_2026_paper.pdf) |
| 58 | **Synthetic Object Compositions for Scalable and Accurate Learning in Detection, Segmentation, and Grounding** | Weikai Huang, Jieyu Zhang, Taoyang Jia, Chenhao Zheng, Ziqi Gao, Jae Sung Park, Ranjay Krishna | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Synthetic_Object_Compositions_for_Scalable_and_Accurate_Learning_in_Detection_CVPR_2026_paper.pdf) |
| 59 | **GeoSAM2: Unleashing the Power of SAM2 for 3D Part Segmentation** | Ken Deng, Yunhan Yang, Jingxiang Sun, Xihui Liu, Yebin Liu, Ding Liang, Yan-Pei Cao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Deng_GeoSAM2_Unleashing_the_Power_of_SAM2_for_3D_Part_Segmentation_CVPR_2026_paper.pdf) |
| 60 | **Seeing Through the Noise: Improving Infrared Small Target Detection and Segmentation from Noise Suppression Perspective** | Maoxun Yuan, Duanni Meng, Ziteng Xi, Tianyi Zhao, Shiji Zhao, Yimian Dai, Xingxing Wei | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_CVPR_2026_paper.pdf) |
| 61 | **SegCompass: Exploring Interpretable Alignment with Sparse Autoencoders for Enhanced Reasoning Segmentation** | Zhenyu Lu, Liupeng Li, Jinpeng Wang, Haoqian Kang, Yan Feng, Ke Chen, Yaowei Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_SegCompass_Exploring_Interpretable_Alignment_with_Sparse_Autoencoders_for_Enhanced_Reasoning_CVPR_2026_paper.pdf) |
| 62 | **GeoSemba: Reconstructing State Space Model for Cross Paradigm Representation in Medical Image Segmentation** | Xutao Sun, Jiarui Li, Junwen Liu, Yonggong Ren | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Sun_GeoSemba_Reconstructing_State_Space_Model_for_Cross_Paradigm_Representation_in_CVPR_2026_paper.pdf) |
| 63 | **Fast Reasoning Segmentation for Images and Videos** | Yiqing Shen, Mathias Unberath | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_Fast_Reasoning_Segmentation_for_Images_and_Videos_CVPR_2026_paper.pdf) |
| 64 | **PromptMoE: A Segmentation Refinement Framework Leveraging Mixture of Experts for Improved Prompting** | Stephen Price, Danielle L. Cote, Elke A. Rundensteiner | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Price_PromptMoE_A_Segmentation_Refinement_Framework_Leveraging_Mixture_of_Experts_for_CVPR_2026_paper.pdf) |
| 65 | **Hierarchical Action Learning for Weakly-Supervised Action Segmentation** | Junxian Huang, Ruichu Cai, Juntao Fang, Hao Zhu, Boyan Xu, Weilin Chen, Zijian Li, Shenghua Gao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Hierarchical_Action_Learning_for_Weakly-Supervised_Action_Segmentation_CVPR_2026_paper.pdf) |
| 66 | **Instruction-Guided Lesion Segmentation for Chest X-rays with Automatically Generated Large-Scale Dataset** | Geon Choi, Hangyul Yoon, Hyunju Shin, Hyunki Park, Sang Hoon Seo, Eunho Yang, Edward Choi | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Choi_Instruction-Guided_Lesion_Segmentation_for_Chest_X-rays_with_Automatically_Generated_Large-Scale_CVPR_2026_paper.pdf) |
| 67 | **MVGGT: Multimodal Visual Geometry Grounded Transformer for Multiview 3D Referring Expression Segmentation** | Changli Wu, Haodong Wang, Jiayi Ji, Yutian Yao, Chunsai Du, Jihua Kang, Yanwei Fu, Liujuan Cao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wu_MVGGT_Multimodal_Visual_Geometry_Grounded_Transformer_for_Multiview_3D_Referring_CVPR_2026_paper.pdf) |
| 68 | **Image-to-Point Cloud Feature Back-Projection for Multimodal Training of 3D Semantic Segmentation** | Jiawei Han, Matteo Poggi, Li Huan, Changshuo Wang, Kaiqi Liu, Wei Li | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Han_Image-to-Point_Cloud_Feature_Back-Projection_for_Multimodal_Training_of_3D_Semantic_CVPR_2026_paper.pdf) |
| 69 | **Diffusion-Based Native Adversarial Synthesis for Enhanced Medical Segmentation Generalization** | Hongyu Zhang, Haipeng Chen, Zhimin Xu, Chengxin Yang, Yingda Lyu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Diffusion-Based_Native_Adversarial_Synthesis_for_Enhanced_Medical_Segmentation_Generalization_CVPR_2026_paper.pdf) |
| 70 | **RS-SSM: Refining Forgotten Specifics in State Space Model for Video Semantic Segmentation** | Kai Zhu, Zhenyu Cui, Zehua Zang, Jiahuan Zhou | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_RS-SSM_Refining_Forgotten_Specifics_in_State_Space_Model_for_Video_CVPR_2026_paper.pdf) |
| 71 | **LangRef3DGS: Natural Language-Guided 3D Referential Segmentation from Partial Observations via 3D Gaussian Splatting** | Xulun Ye, Qin Zhang, Kun Zhou | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ye_LangRef3DGS_Natural_Language-Guided_3D_Referential_Segmentation_from_Partial_Observations_via_CVPR_2026_paper.pdf) |
| 72 | **Discover, Segment, and Select: A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation** | Yilong Yang, Jianxin Tian, Shengchuan Zhang, Liujuan Cao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Discover_Segment_and_Select_A_Progressive_Mechanism_for_Zero-shot_Camouflaged_CVPR_2026_paper.pdf) |
| 73 | **Hyperbolic Prototype Learning with Uncertainty-Aware Consistency for Continual Test-Time Segmentation** | Siddhant Gole, Akash Pal, Amit More, S Divakar Bhat, Subhasis Chaudhuri, Biplab Banerjee | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gole_Hyperbolic_Prototype_Learning_with_Uncertainty-Aware_Consistency_for_Continual_Test-Time_Segmentation_CVPR_2026_paper.pdf) |
| 74 | **Photo-Guided Tooth Segmentation on 3D Oral Scan Model** | Shaojie Zhuang, Guangshun Wei, Jiangxin He, Yuanfeng Zhou | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhuang_Photo-Guided_Tooth_Segmentation_on_3D_Oral_Scan_Model_CVPR_2026_paper.pdf) |
| 75 | **Structure-Aware Representation Distillation for Tiny-Dense Object Segmentation** | Xuesong Liu, Anke Xu, Wenbo Cao, Emmett Ientilucci | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_Structure-Aware_Representation_Distillation_for_Tiny-Dense_Object_Segmentation_CVPR_2026_paper.pdf) |
| 76 | **MARIS: Marine Open-Vocabulary Instance Segmentation** | Bingyu Li, Feiyu Wang, Da Zhang, Zhiyuan Zhao, Junyu Gao, Xuelong Li | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_MARIS_Marine_Open-Vocabulary_Instance_Segmentation_CVPR_2026_paper.pdf) |
| 77 | **Mixture of Prototypes for Test-time Adaptive Segmentation** | Guangrui Li, Zhengyu Zhu, Yongxin Ge | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_Mixture_of_Prototypes_for_Test-time_Adaptive_Segmentation_CVPR_2026_paper.pdf) |
| 78 | **MambaLiteUNet: Cross-Gated Adaptive Feature Fusion for Robust Skin Lesion Segmentation** | Md Maklachur Rahman, Soon Ki Jung, Tracy Hammond | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Rahman_MambaLiteUNet_Cross-Gated_Adaptive_Feature_Fusion_for_Robust_Skin_Lesion_Segmentation_CVPR_2026_paper.pdf) |
| 79 | **Geometry-Aware Cross-Modal Graph Alignment for Referring Segmentation in 3D Gaussian Splatting** | Yuwen Tao, Kanglei Zhou, Chang Li, Liyuan Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Tao_Geometry-Aware_Cross-Modal_Graph_Alignment_for_Referring_Segmentation_in_3D_Gaussian_CVPR_2026_paper.pdf) |
| 80 | **VIRST: Video-Instructed Reasoning Assistant for SpatioTemporal Segmentation** | Jihwan Hong, Jaeyoung Do | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_VIRST_Video-Instructed_Reasoning_Assistant_for_SpatioTemporal_Segmentation_CVPR_2026_paper.pdf) |
| 81 | **Differentiable Laplacian Matrix Guided Superpixel Segmentation** | Jeremy Juybari, Josh Hamilton, Shuvra Das, Chaofan Chen, Andre Khalil, Yifeng Zhu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Juybari_Differentiable_Laplacian_Matrix_Guided_Superpixel_Segmentation_CVPR_2026_paper.pdf) |
| 82 | **Simple-ViLMedSAM: Simple Text Prompts Meet Vision-Language Models for Medical Image Segmentation** | Chengcan Qian, Dong Nie, Geng Chen, Daoqiang Zhang, Xuyun Wen | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Qian_Simple-ViLMedSAM_Simple_Text_Prompts_Meet_Vision-Language_Models_for_Medical_Image_CVPR_2026_paper.pdf) |
| 83 | **Dual-level Adapter Boosting Prompt-free Curvilinear Structure Segmentation** | Kai Zhu, Li Chen, Jun Cheng | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhu_Dual-level_Adapter_Boosting_Prompt-free_Curvilinear_Structure_Segmentation_CVPR_2026_paper.pdf) |
| 84 | **Beyond Text: Visual Description Assembly by Probabilistic Model for CLIP-based Weakly Supervised Semantic Segmentation** | Xianglin Qiu, Jian Wang, Xiaolei Wang, Zhen Zhang, Jimin Xiao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Qiu_Beyond_Text_Visual_Description_Assembly_by_Probabilistic_Model_for_CLIP-based_CVPR_2026_paper.pdf) |
| 85 | **MatchMask: Mask-Centric Generative Data Augmentation for Label-Scarce Semantic Segmentation** | Yuqi Lin, Hao Zhang, Wenqi Shao, Shiqu Liu, Zhihong Gu, Wenxiao Wang, Xiaofei He, Kaipeng Zhang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_MatchMask_Mask-Centric_Generative_Data_Augmentation_for_Label-Scarce_Semantic_Segmentation_CVPR_2026_paper.pdf) |
| 86 | **Semi-supervised Echocardiography Video Segmentation via Anchor Semantic Awareness and Continuous Pseudo-label Reforging** | Yunpeng Fang, Yimu Sun, Jingxing Guo, Huisi Wu, Jing Qin | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Fang_Semi-supervised_Echocardiography_Video_Segmentation_via_Anchor_Semantic_Awareness_and_Continuous_CVPR_2026_paper.pdf) |
| 87 | **RAVEN: Radar Adaptive Vision Encoders for Efficient Chirp-wise Object Detection and Segmentation** | Anuvab Sen, Mir Sayeed Mohammad, Saibal Mukhopadhyay | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Sen_RAVEN_Radar_Adaptive_Vision_Encoders_for_Efficient_Chirp-wise_Object_Detection_CVPR_2026_paper.pdf) |
| 88 | **MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation** | Taha Koleilat, Hojat Asgariandehkordi, Omid Nejatimanzari, Berardino Barile, Yiming Xiao, Hassan Rivaz | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_Adaptation_for_Data-Efficient_and_Generalizable_Medical_Image_CVPR_2026_paper.pdf) |
| 89 | **ELVIS: Enhance Low-Light for Video Instance Segmentation in the Dark** | Joanne Lin, Ruirui Lin, Yini Li, David Bull, Nantheera Anantrasirichai | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lin_ELVIS_Enhance_Low-Light_for_Video_Instance_Segmentation_in_the_Dark_CVPR_2026_paper.pdf) |
| 90 | **SegGBC: Justifiable Coarse-to-Fine Granular-Ball Computing for Enhancing Clustering Image Segmentation** | Qianpeng Chong, Wenyi Zeng, Xiuxuan Shen, Jiajie Li, Qian Yin, Xin Zheng | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chong_SegGBC_Justifiable_Coarse-to-Fine_Granular-Ball_Computing_for_Enhancing_Clustering_Image_Segmentation_CVPR_2026_paper.pdf) |
| 91 | **Long-RVOS: A Comprehensive Benchmark for Long-term Referring Video Object Segmentation** | Tianming Liang, Haichao Jiang, Yuting Yang, Chaolei Tan, Shuai Li, Wei-Shi Zheng, Jian-Fang Hu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_Long-RVOS_A_Comprehensive_Benchmark_for_Long-term_Referring_Video_Object_Segmentation_CVPR_2026_paper.pdf) |
| 92 | **TSTM: Temporal Segmentation for Task-relevant Mask in Visual Reinforcement Learning Generalization** | Weicheng Du, Wenjia Meng, Zhengzhe Zhang, Yilong Yin, Xiankai Lu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Du_TSTM_Temporal_Segmentation_for_Task-relevant_Mask_in_Visual_Reinforcement_Learning_CVPR_2026_paper.pdf) |
| 93 | **Robust Promptable Video Object Segmentation** | Sohyun Lee, Yeho Gwon, Lukas Hoyer, Konrad Schindler, Christos Sakaridis, Suha Kwak | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_Robust_Promptable_Video_Object_Segmentation_CVPR_2026_paper.pdf) |
| 94 | **SAM2Text: Towards Prompt-Free and Multi-Resolution Video Scene Text Segmentation** | Jing-Yao Zhang, Heng Zhang, Mingsen Zhang, Binbin Yang, Fei Yin | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_SAM2Text_Towards_Prompt-Free_and_Multi-Resolution_Video_Scene_Text_Segmentation_CVPR_2026_paper.pdf) |
| 95 | **VesMamba: 3D Pulmonary Vessel Segmentation from CT images via Mamba with Structural Perception and Scale-aware Filtering** | Zhipeng Liu, Guilian Chen, Zheng Jiang, Huisi Wu, Jing Qin | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_VesMamba_3D_Pulmonary_Vessel_Segmentation_from_CT_images_via_Mamba_CVPR_2026_paper.pdf) |
| 96 | **RDF-MIG: A Robust Diffusion Framework for Masked Image Generation to Augment Semantic Segmentation and Change Detection** | Zian Cao, Wei Wei, Qingshan Gao, Yuanyuan Fu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cao_RDF-MIG_A_Robust_Diffusion_Framework_for_Masked_Image_Generation_to_CVPR_2026_paper.pdf) |
| 97 | **S$^2$AM3D: Scale-controllable Part Segmentation of 3D Point Clouds** | Han Su, Tianyu Huang, Zichen Wan, Xiaohe Wu, Wangmeng Zuo | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Su_S2AM3D_Scale-controllable_Part_Segmentation_of_3D_Point_Clouds_CVPR_2026_paper.pdf) |
| 98 | **Unlocking 3D Affordance Segmentation with 2D Semantic Knowledge** | Yu Huang, Zelin Peng, Changsong Wen, Xiaokang Yang, Wei Shen | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_Unlocking_3D_Affordance_Segmentation_with_2D_Semantic_Knowledge_CVPR_2026_paper.pdf) |
| 99 | **DeRVOS: Decoupling Consistent Trajectory Generation and Multimodal Understanding for Referring Video Object Segmentation** | Wenxuan Cheng, Ming Dai, Huimin Lu, Wankou Yang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cheng_DeRVOS_Decoupling_Consistent_Trajectory_Generation_and_Multimodal_Understanding_for_Referring_CVPR_2026_paper.pdf) |
| 100 | **SOUPLE: Enhancing Audio-Visual Localization and Segmentation with Learnable Prompt Contexts** | Khanh Binh Nguyen, Chae Jung Park | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Nguyen_SOUPLE_Enhancing_Audio-Visual_Localization_and_Segmentation_with_Learnable_Prompt_Contexts_CVPR_2026_paper.pdf) |
| 101 | **XSeg: A Large-scale X-ray Contraband Segmentation Benchmark For Real-World Security Screening** | Hongxia Gao, Yixin Chen, Jiali Wen, Litao Li, Qianyun Liu, Kaijie Zhang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Gao_XSeg_A_Large-scale_X-ray_Contraband_Segmentation_Benchmark_For_Real-World_Security_CVPR_2026_paper.pdf) |
| 102 | **Unsupervised Multi-Scale Segmentation of 3D Subcellular World with Stable Diffusion Foundation Model** | Mostofa Rafid Uddin, HM Shadman Tabib, Thanh-Huy Nguyen, Kashish Gandhi, Min Xu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Uddin_Unsupervised_Multi-Scale_Segmentation_of_3D_Subcellular_World_with_Stable_Diffusion_CVPR_2026_paper.pdf) |
| 103 | **Test-Time Training for LiDAR Semantic Segmentation under Corruption via Geometric Inlier Discrimination** | Hyeonseong Kim, Hyun-Kurl Jang, Kuk-Jin Yoon | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Test-Time_Training_for_LiDAR_Semantic_Segmentation_under_Corruption_via_Geometric_CVPR_2026_paper.pdf) |
| 104 | **Denoise and Align: Towards Source-Free UDA for Robust Panoramic Semantic Segmentation** | Yaowen Chang, Zhen Cao, Xu Zheng, Xiaoxin Mi, Zhen Dong | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chang_Denoise_and_Align_Towards_Source-Free_UDA_for_Robust_Panoramic_Semantic_CVPR_2026_paper.pdf) |
| 105 | **Multimodal Causality-Driven Representation Learning for Generalizable Medical Image Segmentation** | Xusheng Liang, Lihua Zhou, Nianxin Li, Miao Xu, Ziyang Song, Dong Yi, Jinlin Wu, Jiawei Ma, Hongbin Liu, Zhen Lei, Jiebo Luo | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liang_Multimodal_Causality-Driven_Representation_Learning_for_Generalizable_Medical_Image_Segmentation_CVPR_2026_paper.pdf) |
| 106 | **ClimaOoD: Improving Anomaly Segmentation via Physically Realistic Synthetic Data** | Yuxing Liu, Zheng Li, Huanhuan Liang, Ji Zhang, Zeyu Sun, Yong Liu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_ClimaOoD_Improving_Anomaly_Segmentation_via_Physically_Realistic_Synthetic_Data_CVPR_2026_paper.pdf) |
| 107 | **Masked Representation Modeling for Domain-Adaptive Segmentation** | Wenlve Zhou, Zhiheng Zhou, Tiantao Xian, Yikui Zhai, Weibin Wu, Biyun MA | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_Masked_Representation_Modeling_for_Domain-Adaptive_Segmentation_CVPR_2026_paper.pdf) |
| 108 | **SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons** | Haiyang Xu, Ronghuan Wu, Li-Yi Wei, Nanxuan Zhao, Chenxi Liu, Cuong Nguyen, Zhuowen Tu, Zhaowen Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Xu_SemLayer_Semantic-aware_Generative_Segmentation_and_Layer_Construction_for_Abstract_Icons_CVPR_2026_paper.pdf) |
| 109 | **MARSS: Radar Semantic Segmentation via Modular Attention and State Space Models** | Fengyu Chen, Tiao Tan, Teng Li, Yuantian Quan, Qingmin Liao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_MARSS_Radar_Semantic_Segmentation_via_Modular_Attention_and_State_Space_CVPR_2026_paper.pdf) |
| 110 | **High-Precision Dichotomous Image Segmentation via Depth Integrity-Prior and Fine-Grained Patch Strategy** | Xianjie Liu, Keren Fu, Qijun Zhao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_High-Precision_Dichotomous_Image_Segmentation_via_Depth_Integrity-Prior_and_Fine-Grained_Patch_CVPR_2026_paper.pdf) |
| 111 | **LaDy: Lagrangian-Dynamic Informed Network for Skeleton-based Action Segmentation via Spatial-Temporal Modulation** | Haoyu Ji, Xueting Liu, Yu Gao, Wenze Huang, Zhihao Yang, Weihong Ren, Zhiyong Wang, Honghai Liu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ji_LaDy_Lagrangian-Dynamic_Informed_Network_for_Skeleton-based_Action_Segmentation_via_Spatial-Temporal_CVPR_2026_paper.pdf) |
| 112 | **WalkGPT: Grounded Vision-Language Conversation with Depth-Aware Segmentation for Pedestrian Navigation** | Rafi Ibn Sultan, Hui Zhu, Xiangyu Zhou, Chengyin Li, Prashant Khanduri, Marco Brocanelli, Dongxiao Zhu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ibn_Sultan_WalkGPT_Grounded_Vision-Language_Conversation_with_Depth-Aware_Segmentation_for_Pedestrian_Navigation_CVPR_2026_paper.pdf) |
| 113 | **Generalizable Knowledge Distillation from Vision Foundation Models for Semantic Segmentation** | Chonghua Lv, Dong Zhao, Shuang Wang, Dou Quan, Ning Huyan, Nicu Sebe, Zhun Zhong | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lv_Generalizable_Knowledge_Distillation_from_Vision_Foundation_Models_for_Semantic_Segmentation_CVPR_2026_paper.pdf) |
| 114 | **EvObj: Learning Evolving Object-centric Representations for 3D Instance Segmentation without Scene Supervision** | Jiahao Chen, Zihui Zhang, Yafei Yang, Jinxi Li, Shenxing Wei, Zhixuan Sun, Bo Yang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_EvObj_Learning_Evolving_Object-centric_Representations_for_3D_Instance_Segmentation_without_CVPR_2026_paper.pdf) |
| 115 | **PIX-TAB: Efficient PIXel-Precise TABle Structure Recognition Approach with Speculative Decoding and Region-Based Image Segmentation** | Viktor Zaytsev, Olena Vynokurova, Pavlo Tytarchuk, Dmytro Kozii, Vitalii Pohribnyi, Olga Radyvonenko, Artem Shcherbina | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zaytsev_PIX-TAB_Efficient_PIXel-Precise_TABle_Structure_Recognition_Approach_with_Speculative_Decoding_CVPR_2026_paper.pdf) |
| 116 | **Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance** | Vanessa Emanuela Guarino, Claudia Winklmayr, Jannik Franzen, Josef Lorenz Rumberger, Manuel Pfeuffer, Sonja Greven, Klaus Maier-Hein, Dagmar Kainmueller, Christoph Karg, Carsten T. Lüth | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.pdf) |
| 117 | **MORE-STEM: Long-Short MemOry REcall and Spatio-TEmporal Consistency Model for Query-Driven 3D/4D Point Cloud Segmentation** | Chade Li, Haida Feng, Pengju Zhang, Yihong Wu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_MORE-STEM_Long-Short_MemOry_REcall_and_Spatio-TEmporal_Consistency_Model_for_Query-Driven_CVPR_2026_paper.pdf) |
| 118 | **VidEoMT: Your ViT is Secretly Also a Video Segmentation Model** | Narges Norouzi, Idil Esen Zulfikar, Niccolò Cavagnero, Tommie Kerssies, Bastian Leibe, Gijs Dubbelman, Daan de Geus | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Norouzi_VidEoMT_Your_ViT_is_Secretly_Also_a_Video_Segmentation_Model_CVPR_2026_paper.pdf) |
| 119 | **SPOT: Spatiotemporal Prompt Optimization for Motion-Stabilized MLLM-Guided Video Segmentation** | Jiayi Fan, Zheyun Qin, Xiaoming Xi, Xiushan Nie, Yilong Yin | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Fan_SPOT_Spatiotemporal_Prompt_Optimization_for_Motion-Stabilized_MLLM-Guided_Video_Segmentation_CVPR_2026_paper.pdf) |
| 120 | **Live Interactive Training for Video Segmentation** | Xinyu Yang, Haozheng Yu, Yihong Sun, Bharath Hariharan, Jennifer J. Sun | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Live_Interactive_Training_for_Video_Segmentation_CVPR_2026_paper.pdf) |
| 121 | **SHAPE: Structure-aware Hierarchical Unsupervised Domain Adaptation with Plausibility Evaluation for Medical Image Segmentation** | Linkuan Zhou, Yinghao Xia, Yufei Shen, Xiangyu Li, Wenjie Du, Cong Cong, Leyi Wei, Ran Su, Qiangguo Jin | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhou_SHAPE_Structure-aware_Hierarchical_Unsupervised_Domain_Adaptation_with_Plausibility_Evaluation_for_CVPR_2026_paper.pdf) |
| 122 | **D-Convexity: A Unified Differentiable Convex Shape Prior via Quasi-Concavity for Data-driven Image Segmentation** | Shengzhe Chen, Hao Yan | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_D-Convexity_A_Unified_Differentiable_Convex_Shape_Prior_via_Quasi-Concavity_for_CVPR_2026_paper.pdf) |
| 123 | **PMRNet: Physics-informed Multi-scale Refinement Network for Medical Image Segmentation** | Boce Kang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Kang_PMRNet_Physics-informed_Multi-scale_Refinement_Network_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf) |
| 124 | **NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization** | Yik San Cheng, Runkai Zhao, Weidong Cai | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/San_Cheng_NeuroSeg_Meets_DINOv3_Transferring_2D_Self-Supervised_Visual_Priors_to_3D_CVPR_2026_paper.pdf) |
| 125 | **INSID3: Training-Free In-Context Segmentation with DINOv3** | Claudia Cuttano, Gabriele Trivigno, Christoph Reich, Daniel Cremers, Carlo Masone, Stefan Roth | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Cuttano_INSID3_Training-Free_In-Context_Segmentation_with_DINOv3_CVPR_2026_paper.pdf) |
| 126 | **SAGE: Style-Adaptive Generalization for Privacy-Constrained Semantic Segmentation Across Domains** | Qingmei Li, Yang Zhang, Peifeng Zhang, Haohuan Fu, Juepeng Zheng | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_SAGE_Style-Adaptive_Generalization_for_Privacy-Constrained_Semantic_Segmentation_Across_Domains_CVPR_2026_paper.pdf) |
| 127 | **Bootstrap Your Own AV-Proxies: Adaptive Contrastive and Prototype Learning for Audio-Visual Segmentation** | Junbo Zhang, Hang Su, Zhaofan Li, Hang Dong, Chao Sun | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Bootstrap_Your_Own_AV-Proxies_Adaptive_Contrastive_and_Prototype_Learning_for_CVPR_2026_paper.pdf) |
| 128 | **Selective, Regularized, and Calibrated: Harnessing Vision Foundation Models for Cross-Domain Few-Shot Semantic Segmentation** | Junyuan Ma, Xunzhi Xiang, Wenbin Li, Qi Fan, Yang Gao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ma_Selective_Regularized_and_Calibrated_Harnessing_Vision_Foundation_Models_for_Cross-Domain_CVPR_2026_paper.pdf) |
| 129 | **Annotation-Efficient Coreset Selection for Context-dependent Segmentation** | Jin Zhang, Zhe Cao, Biwen Yang, Ruiheng Zhang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Annotation-Efficient_Coreset_Selection_for_Context-dependent_Segmentation_CVPR_2026_paper.pdf) |
| 130 | **REL-SF4PASS: Panoramic Semantic Segmentation with REL Depth Representation and Spherical Fusion** | Xuewei Li, Xinghan Bao, Zhimin Chen, Xi Li | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_REL-SF4PASS_Panoramic_Semantic_Segmentation_with_REL_Depth_Representation_and_Spherical_CVPR_2026_paper.pdf) |
| 131 | **Towards Streaming Referring Video Segmentation via Large Language Model** | Wenkang Zhang, Kaicheng Yang, Xiang An, Qiang Li, Ziyong Feng, Wankou Yang, Jiankang Deng | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Towards_Streaming_Referring_Video_Segmentation_via_Large_Language_Model_CVPR_2026_paper.pdf) |
| 132 | **CROWn: A Unified Framework for Anti-Aliased Downsampling and Phase-Calibrated Fusion in 3D Medical Segmentation** | Xingru Huang, Shuanghua Ye, Zhao Huang, Wenwen Tang, Huiyu Zhou, Zhiwen Zheng, Jin Liu, Xiaoshuai Zhang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_CROWn_A_Unified_Framework_for_Anti-Aliased_Downsampling_and_Phase-Calibrated_Fusion_CVPR_2026_paper.pdf) |
| 133 | **CG-Reasoner: Centroid-Guided Positional Reasoning Segmentation for Medical Imaging with a Robust Visual-Text Consistency Metric** | Lakshmikar Reddy Polamreddy, Ming Ma | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Polamreddy_CG-Reasoner_Centroid-Guided_Positional_Reasoning_Segmentation_for_Medical_Imaging_with_a_CVPR_2026_paper.pdf) |
| 134 | **Seeing Beyond: Extrapolative Domain Adaptive Panoramic Segmentation** | Yuanfan Zheng, Kunyu Peng, Xu Zheng, Kailun Yang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_Seeing_Beyond_Extrapolative_Domain_Adaptive_Panoramic_Segmentation_CVPR_2026_paper.pdf) |
| 135 | **PointGS: Semantic-Consistent Unsupervised 3D Point Cloud Segmentation with 3D Gaussian Splatting** | Yixiao Song, Qingyong Li, Wen Wang, Zhicheng Yan | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Song_PointGS_Semantic-Consistent_Unsupervised_3D_Point_Cloud_Segmentation_with_3D_Gaussian_CVPR_2026_paper.pdf) |
| 136 | **Towards High-Quality Image Segmentation: Improving Topology Accuracy by Penalizing Neighbor Pixels** | Juan Miguel Valverde, Dim P. Papadopoulos, Rasmus Larsen, Anders Bjorholm Dahl | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Valverde_Towards_High-Quality_Image_Segmentation_Improving_Topology_Accuracy_by_Penalizing_Neighbor_CVPR_2026_paper.pdf) |
| 137 | **Best Segmentation Buddies for Image-Shape Correspondence** | Itai Lang, Dongwei Lyu, Dale Decatur, Rana Hanocka | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lang_Best_Segmentation_Buddies_for_Image-Shape_Correspondence_CVPR_2026_paper.pdf) |
| 138 | **Better, Stronger, Faster: Tackling the Trilemma in MLLM-based Segmentation with Simultaneous Textual Mask Prediction** | Jiazhen Liu, Mingkuan Feng, Long Chen | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_Better_Stronger_Faster_Tackling_the_Trilemma_in_MLLM-based_Segmentation_with_CVPR_2026_paper.pdf) |
| 139 | **FlowDIS: Language-Guided Dichotomous Image Segmentation with Flow Matching** | Andranik Sargsyan, Shant Navasardyan | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Sargsyan_FlowDIS_Language-Guided_Dichotomous_Image_Segmentation_with_Flow_Matching_CVPR_2026_paper.pdf) |
| 140 | **From Infusion to Assimilation Distillation for Medical Image Segmentation** | Jiankang Hong, Ye Luo, Yinan Liu, Junsong Yuan | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Hong_From_Infusion_to_Assimilation_Distillation_for_Medical_Image_Segmentation_CVPR_2026_paper.pdf) |
| 141 | **Conversational Image Segmentation: Grounding Abstract Concepts with Scalable Supervision** | Aadarsh Sahoo, Georgia Gkioxari | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Sahoo_Conversational_Image_Segmentation_Grounding_Abstract_Concepts_with_Scalable_Supervision_CVPR_2026_paper.pdf) |
| 142 | **NG-GS: NeRF-guided 3D Gaussian Splatting Segmentation** | Yi He, Tao Wang, Yi Jin, Congyan Lang, Yidong Li, Haibin Ling | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/He_NG-GS_NeRF-guided_3D_Gaussian_Splatting_Segmentation_CVPR_2026_paper.pdf) |
| 143 | **DIMOS: Disentangling Instance-level Moving Object Segmentation** | Hongxiang Huang, Hongwei Ren, Xiaopeng Lin, Yulong Huang, Zeke Xie, Bojun Cheng | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_DIMOS_Disentangling_Instance-level_Moving_Object_Segmentation_CVPR_2026_paper.pdf) |
| 144 | **UniVerse: A Unified Modulation Framework for Segmentation-Free, Disentangled Multi-Concept Personalization** | Quynh Phung, Sandesh Ghimire, Minsi Hu, Chung-Chi Tsai, Jia-Bin Huang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Phung_UniVerse_A_Unified_Modulation_Framework_for_Segmentation-Free_Disentangled_Multi-Concept_Personalization_CVPR_2026_paper.pdf) |
| 145 | **GeoMotion: Rethinking Motion Segmentation via Latent 4D Geometry** | Xiankang He, Peile Lin, Ying Cui, Dongyan Guo, Chunhua Shen, Xiaoqin Zhang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/He_GeoMotion_Rethinking_Motion_Segmentation_via_Latent_4D_Geometry_CVPR_2026_paper.pdf) |
| 146 | **RMAE-ProGRess: Advancing Semantic Segmentation in Unstructured Environments** | Manish Bhurtel, Danda B. Rawat | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Bhurtel_RMAE-ProGRess_Advancing_Semantic_Segmentation_in_Unstructured_Environments_CVPR_2026_paper.pdf) |
| 147 | **PR-MaGIC: Prompt Refinement Via Mask Decoder Gradient Flow For In-Context Segmentation** | Minjae Lee, Sungwoo Hur, Soojin Hwang, Won Hwa Kim | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Lee_PR-MaGIC_Prompt_Refinement_Via_Mask_Decoder_Gradient_Flow_For_In-Context_CVPR_2026_paper.pdf) |
| 148 | **PixDLM: A Dual-Path Multimodal Language Model for UAV Reasoning Segmentation** | Shuyan Ke, Yifan Mei, Changli Wu, Yonghan Zheng, Jiayi Ji, Liujuan Cao, Rongrong Ji | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Ke_PixDLM_A_Dual-Path_Multimodal_Language_Model_for_UAV_Reasoning_Segmentation_CVPR_2026_paper.pdf) |
| 149 | **Divide, Conquer, and Aggregate: Asymmetric Experts for Class-Imbalanced Semi-Supervised Medical Image Segmentation** | Yajun Liu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Liu_Divide_Conquer_and_Aggregate_Asymmetric_Experts_for_Class-Imbalanced_Semi-Supervised_Medical_CVPR_2026_paper.pdf) |
| 150 | **Refer-Agent: A Collaborative Multi-Agent System with Reasoning and Reflection for Referring Video Object Segmentation** | Haichao Jiang, Tianming Liang, Wei-Shi Zheng, Jian-Fang Hu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Jiang_Refer-Agent_A_Collaborative_Multi-Agent_System_with_Reasoning_and_Reflection_for_CVPR_2026_paper.pdf) |
| 151 | **Heuristic Self-Paced Learning for Domain Adaptive Semantic Segmentation under Adverse Conditions** | Shiqin Wang, Haoyang Chen, Huaizhou Huang, Yinkan He, Dongfang Sun, Xiaoqing Chen, Xingyu Liu, Zheng Wang, Kaiyan Zhao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Wang_Heuristic_Self-Paced_Learning_for_Domain_Adaptive_Semantic_Segmentation_under_Adverse_CVPR_2026_paper.pdf) |
| 152 | **Hugging Visual Prompt and Segmentation Tokens: Consistency Learning for Fine-Grained Visual Understanding in MLLMs** | Jing Yang, Sen Yang, Boqiang Duan, Ming Dai, Wei Zhang, Xiao Tan, Kunbin Chen, Wei He, Jingdong Wang, Hanli Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Hugging_Visual_Prompt_and_Segmentation_Tokens_Consistency_Learning_for_Fine-Grained_CVPR_2026_paper.pdf) |
| 153 | **Concept-Aware LoRA for Domain-Aligned Segmentation Dataset Generation** | Minho Park, Sunghyun Park, Jungsoo Lee, Hyojin Park, Kyuwoong Hwang, Fatih Porikli, Jaegul Choo, Sungha Choi | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Park_Concept-Aware_LoRA_for_Domain-Aligned_Segmentation_Dataset_Generation_CVPR_2026_paper.pdf) |
| 154 | **AG-VAS: Anchor-Guided Zero-Shot Visual Anomaly Segmentation with Large Multimodal Models** | Zhen Qu, Xian Tao, Xiaoyi Bao, Dingrong Wang, ShiChen Qu, Zhengtao Zhang, Xingang Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Qu_AG-VAS_Anchor-Guided_Zero-Shot_Visual_Anomaly_Segmentation_with_Large_Multimodal_Models_CVPR_2026_paper.pdf) |
| 155 | **Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation** | Ziqian Yang, Xinqiao Zhao, Xiaolei Wang, Quan Zhang, Jimin Xiao | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Leveraging_Class_Distributions_in_CLIP_for_Weakly_Supervised_Semantic_Segmentation_CVPR_2026_paper.pdf) |
| 156 | **Polyphony: Diffusion-based Dual-Hand Action Segmentation with Alternating Vision Transformer and Semantic Conditioning** | Hao Zheng, Hu Wang, Tiantian Zheng, Prajjwal Bhattarai, Tuka Alhanai | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_Polyphony_Diffusion-based_Dual-Hand_Action_Segmentation_with_Alternating_Vision_Transformer_and_CVPR_2026_paper.pdf) |
| 157 | **SAQN: Semantic-based Adaptive Query Network for 3D Referring Expression Segmentation** | Jiale Huang, Shangfei Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Huang_SAQN_Semantic-based_Adaptive_Query_Network_for_3D_Referring_Expression_Segmentation_CVPR_2026_paper.pdf) |
| 158 | **Focus on Background: Exploring SAM's Potential in Few-shot Medical Image Segmentation with Background-centric Prompting** | Yuntian Bo, Yazhou Zhu, Piotr Koniusz, Haofeng Zhang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Bo_Focus_on_Background_Exploring_SAMs_Potential_in_Few-shot_Medical_Image_CVPR_2026_paper.pdf) |
| 159 | **Geometric-Aware Hypergraph Reasoning for Novel Class Discovery in Point Cloud Segmentation** | Zihao Zhang, Aming Wu, Yang Li, Yahong Han, Jialie Shen | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_Geometric-Aware_Hypergraph_Reasoning_for_Novel_Class_Discovery_in_Point_Cloud_CVPR_2026_paper.pdf) |
| 160 | **Discriminative Perception via Anchored Description for Reasoning Segmentation** | Tao Yang, Qing Zhou, Yanliang Li, Qi Wang | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Discriminative_Perception_via_Anchored_Description_for_Reasoning_Segmentation_CVPR_2026_paper.pdf) |
| 161 | **SD-FSMIS: Adapting Stable Diffusion for Few-Shot Medical Image Segmentation** | Meihua Li, Yang Zhang, Weizhao He, Hu Qu, Yisong Li | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Li_SD-FSMIS_Adapting_Stable_Diffusion_for_Few-Shot_Medical_Image_Segmentation_CVPR_2026_paper.pdf) |
| 162 | **Bayesian Decomposition and Semantic Completion for Few-shot Semantic Segmentation** | Guangchen Shi, Yirui Wu, Wei Zhu, Tao Wang, Hao Zhang, Bo Li, Tong Lu | [PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Shi_Bayesian_Decomposition_and_Semantic_Completion_for_Few-shot_Semantic_Segmentation_CVPR_2026_paper.pdf) |

---
