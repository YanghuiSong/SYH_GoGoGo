# SYH_GoGoGo
The following is the code for reproducing recently read papers and the work currently in progress.
# Content
1. [[Pilot Study](#PilotStudy)]
2. [[Remote Sensing](#Remote_Sensing)]
3. [[Classification](#classification)]
4. [[Segmentation](#segmentation)]    
       
-----------------------------------------------------------------------------------------------
<a name="PilotStudy"></a>  
## Pilot Study
1. [2024 ICLR] **FeatUp: A Model-Agnostic Framework for Features at Any Resolution** [[paper]](https://openreview.net/pdf?id=GkJiNn2QDF) [[code]](https://github.com/mhamilton723/FeatUp)[[Notes](#FeatUpLearning)]   
2. [2025 CVPR] **SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.pdf) [[code]](https://github.com/likyoo/SegEarth-OV)
3. [2025 arXiv] **AnyUp: Universal Feature Upsampling** [[paper]](https://arxiv.org/abs/2510.12764) [[code]](https://github.com/wimmerth/anyup)

<a name="Remote_Sensing"></a>  
## Remote Sensing
1. [2025 arXiv] **DynamicEarth: How Far are We from Open-Vocabulary Change Detection?** [[paper]](https://arXiv.org/abs/2501.12931) [[code]](https://github.com/likyoo/DynamicEarth)
2. [2025 TGRS] **A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation.** [[paper]](https://ieeexplore.ieee.org/document/11063320) [[code]](https://github.com/sstary/SSRS)
3. [2025 ICASSP] **Enhancing Remote Sensing Vision-Language Models for Zero-Shot Scene Classification.** [[paper]](https://arXiv.org/abs/2409.00698) [[code]](https://github.com/elkhouryk/RS-TransCLIP)
4. [2025 ICCV] **https://github.com/mburges-cvl/ICCV_AL4FM.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Burges_Active_Learning_Meets_Foundation_Models_Fast_Remote_Sensing_Data_Annotation_ICCV_2025_paper.pdf) [[code]](https://github.com/mburges-cvl/ICCV_AL4FM)
5. [2025 ICCV] **Dynamic Dictionary Learning for Remote Sensing Image Segmentation.** [[paper]](https://arXiv.org/pdf/2503.06683) [[code]](https://github.com/XavierJiezou/D2LS)
6. [2025 ICCV] **GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks.** [[paper]](https://arxiv.org/pdf/2411.19325) [[code]](https://github.com/The-AI-Alliance/GEO-Bench-VLM)
7. [2025 ICCV] **SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation.** [[paper]](https://arXiv.org/abs/2507.12857) [[code]](https://github.com/HuangShiqi128/SCORE)
8. [2025 ICCV] **When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning.** [[paper]](https://arXiv.org/pdf/2503.07588) [[code]](https://github.com/VisionXLab/LRS-VQA)
9. [2025 ICCV] **SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Sumbul_SMARTIES_Spectrum-Aware_Multi-Sensor_Auto-Encoder_for_Remote_Sensing_Images_ICCV_2025_paper.pdf) [[code]](https://github.com/gsumbul/SMARTIES)
10. [2025 ICCV] **Continuous Remote Sensing Image Super-Resolution via Neural Operator Diffusion** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_NeurOp-Diff_Continuous_Remote_Sensing_Image_Super-Resolution_via_Neural_Operator_Diffusion_ICCV_2025_paper.pdf) [[code]](https://github.com/zerono000/NeurOp-Diff)
11. [2025 ICCV] **HoliTracer: Holistic Vectorization of Geographic Objects from Large-Size Remote Sensing Imagery.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_HoliTracer_Holistic_Vectorization_of_Geographic_Objects_from_Large-Size_Remote_Sensing_ICCV_2025_paper.pdf) [[code]](https://github.com/vvangfaye/HoliTracer) 
12. [2025 AAAI] **ZoRI: Towards discriminative zero-shot remote sensing instance segmentation.** [[paper]](https://arXiv.org/abs/2412.12798) [[code]](https://github.com/HuangShiqi128/ZoRI)
13. [2024 NIPS] **Segment Any Change.** [[paper]](https://proceedings.NIPS.cc/paper_files/paper/2024/file/9415416201aa201902d1743c7e65787b-Paper-Conference.pdf) [[code]](https://github.com/Z-Zheng/pytorch-change-models)
14. [2025 CVPR] **SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images.** [[paper]](https://arXiv.org/abs/2410.01768) [[code]](https://github.com/likyoo/SegEarth-OV)
15. [2025 CVPR] **XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?** [[paper]](https://arXiv.org/abs/2503.23771) [[code]](https://github.com/EvolvingLMMs-Lab/XLRS-Bench)
16. [2025 CVPR] **Exact: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Exact_Exploring_Space-Time_Perceptive_Clues_for_Weakly_Supervised_Satellite_Image_CVPR_2025_paper.pdf) [[code]](https://github.com/MiSsU-HH/Exact)
17. [2025 Arxiv] **SegEarth-OV-2: Annotation-Free Open-Vocabulary Segmentation for Remote-Sensing Images** [[paper]](https://arxiv.org/abs/2508.18067)  [[code]](https://github.com/earth-insights/SegEarth-OV-2)
18. [2025 AAAI] **Towards Open-Vocabulary Remote Sensing Image Semantic Segmentation** [[paper]](https://arxiv.org/abs/2412.19492) [[code]](https://github.com/yecy749/GSNet)
19. [2025 Arxiv] **InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition** [[paper]](https://arxiv.org/pdf/2505.15818) [[code]](https://github.com/VoyagerXvoyagerx/InstructSAM)
20. [2025 Arxiv] **DescribeEarth: Describe Anything for Remote Sensing Images** [[paper]](https://arxiv.org/pdf/2509.25654v1) [[code]](https://github.com/earth-insights/DescribeEarth)
21. [2025 NIPS] **GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset** [[paper]](https://arxiv.org/abs/2507.14697) [[code]](https://github.com/Z-ZW-WXQ/GTPBD)
22. [2025 Arxiv] **RS3DBench: A Comprehensive Benchmark for 3D Spatial Perception in Remote Sensing** [[paper]](https://arxiv.org/abs/2509.18897) [[code]](https://rs3dbench.github.io)
23. [2025 Arxiv] **DGL-RSIS: Decoupling Global Spatial Context and Local Class Semantics for Training-Free Remote Sensing Image Segmentation** [[paper]](https://arxiv.org/pdf/2509.00598) [[code]](https://github.com/designer1024/DGL-RSIS)
24. [2025 TGRS] **A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection** [[paper]](https://ieeexplore.ieee.org/document/11172325) [[code]](https://github.com/fuyimin96/SAM-SPL)
25. [2025 TGRS] **Semantic Prototyping With CLIP for Few-Shot Object Detection in Remote Sensing Images** [[paper]](https://ieeexplore.ieee.org/document/10930588)
26. [2025 Arxiv] **ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild** [[paper]](https://arxiv.org/abs/2501.13354) [[code]](https://github.com/waterdisappear/ATRNet-STAR)
27. [2025 Arxiv] **RSKT-Seg: Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing** [[paper]](https://arxiv.org/pdf/2509.12040) [[code]](https://github.com/LiBingyu01/RSKT-Seg)
28. [2025 ISPRS]  **AdaptVFMs-RSCD: Advancing Remote Sensing Change Detection from binary to semantic with SAM and CLIP** [[paper]](https://doi.org/10.1016/j.isprsjprs.2025.09.010) [[data]](https://github.com/Jiang-CHD-YunNan/RS-VFMs-Fine-tuning-Dataset)
29. **PeftCD: Leveraging Vision Foundation Models with Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection** [[paper]](https://arxiv.org/pdf/2509.09572) [[code]](https://github.com/dyzy41/PeftCD)
30. [2025 Arxiv] **AlignCLIP: Self-Guided Alignment for Remote Sensing Open-Vocabulary Semantic Segmentation** [[paper]](https://openreview.net/forum?id=hpD3tn7Xbp) [[code]](https://openreview.net/attachment?id=hpD3tn7Xbp&name=supplementary_material)
31. [2025 Arxiv] **Few-Shot Adaptation Benchmark for Remote Sensing Vision-Language Models** [[paper]](https://arxiv.org/pdf/2510.07135) [[code]](https://github.com/elkhouryk/fewshot_RSVLMs)
32. [2025 RSE] **Strategic sampling for training a semantic segmentation model in operational mapping: Case studies on cropland parcel extraction** [[paper]](https://doi.org/10.1016/j.rse.2025.115034) [[data]](https://doi.org/10.5281/zenodo.16595511) [[code]](https://github.com/Remote-Sensing-of-Land-Resource-Lab/Training-Sample-Selection)
33. [2025 TIP] **Universal Fine-Grained Visual Categorization by Concept Guided Learning** [[paper]](https://ieeexplore.ieee.org/document/10829548) [[data]](https://drive.google.com/file/d/11hYbdO32hyspucDKp5wwjwvCaD38AEKe/view?usp=sharing) [[code]](https://github.com/BiQiWHU/CGL)
34. [2025 TIP] **SARATR-X: Towards Building A Foundation Model for SAR Target Recognition** [[paper]](https://ieeexplore.ieee.org/document/10856784) [[code]](https://github.com/waterdisappear/SARATR-X)
35. [2025 TIP] **HSLabeling: Towards Efficient Labeling for Large-scale Remote Sensing Image Segmentation with Hybrid Sparse Labeling** [[paper]](https://ieeexplore.ieee.org/document/10829548) [[data]](https://drive.google.com/drive/folders/1CiYzJyBn1rV-xsrsYQ6o2HDQjdfnadHl) [[code]](https://github.com/linjiaxing99/HSLabeling)

<a name="classification"></a>  
## Classification
...
<a name="segmentation"></a>  
## Segmentation
...

<a name="FeatUpLearning"></a>  
## FeatUp
**FeatUp的核心思想与方法**

FeatUp的核心灵感来自NeRF的多视图一致性原理：通过观察同一图像经微小变换（如裁剪、翻转、缩放）后的多个低分辨率特征视图，学习高分辨率特征的空间一致性。具体包括以下关键设计：

1. 多视图一致性损失（核心监督信号）
对输入图像施加随机微小变换（如填充、缩放、水平翻转），得到多个“抖动”版本的低分辨率特征。FeatUp学习一个高分辨率特征图，使其经下采样后能匹配所有抖动视图的低分辨率特征，通过高斯似然损失（含自适应不确定性）监督这一过程，确保高分辨率特征的空间一致性。

2. 两种下采样器（模拟模型池化行为）
   
为匹配不同模型的特征降采样机制，设计两种下采样器：

简单下采样器：学习非负归一化模糊核，通过卷积实现特征平滑下采样，适用于固定感受野模型（如CNN）。

注意力下采样器：通过1×1卷积预测显著性图，动态调整下采样核权重，适应动态感受野或对象显著性（如ViT的 patch 注意力机制）。
3. 两种上采样器（核心创新）
FeatUp提供两种即插即用的上采样变体，可直接替换现有特征：

JBU FeatUp（通用前向传播上采样）
基于联合双边滤波（JBU） 的改进，通过堆叠参数化JBU层，利用输入图像的高分辨率信号引导特征上采样。关键优化：

设计高效CUDA内核，比标准PyTorch实现快10倍、内存占用低2个数量级；

用MLP替代传统JBU的固定高斯核，学习特征与图像高频细节的关联，保留语义的同时恢复边缘信息。

Implicit FeatUp（单图像隐式上采样）

过拟合一个小型隐式网络到单图像特征，通过傅里叶特征编码（含颜色信息）实现任意分辨率的特征重建。优势：

参数仅为显式特征存储的1/100，支持超高分辨率输出；

结合总变差正则化避免噪声，适合需要精细细节的场景。
关键贡献

模型无关框架：适用于任意视觉 backbone（CNN、ViT、自监督模型如DINO等），无需修改原模型结构。
高效JBU实现：提出首个高效CUDA版联合双边滤波，解决传统JBU计算瓶颈，支持大规模模型部署。
即插即用提升：上采样特征可直接替换现有特征，在不重新训练下游模型的情况下提升性能（如分割mIoU、深度估计精度）。
