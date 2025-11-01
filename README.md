# SYH_GoGoGo
The following is the code for reproducing recently read papers and the work currently in progress.
# Content
1. [[Pilot Study](#PilotStudy)]
2. [[上采样在遥感图像中的应用](#上采样与遥感)]
3. [[OVSS](#OVSS)]
4. [[Features are vital](#About_Features)]
5. [[Remote Sensing](#Remote_Sensing)]
6. [[Classification](#classification)]
7. [[Segmentation](#segmentation)]
       
-----------------------------------------------------------------------------------------------
<a name="PilotStudy"></a>  
## Pilot Study
1. [2024 ICLR] **FeatUp: A Model-Agnostic Framework for Features at Any Resolution** [[paper]](https://openreview.net/pdf?id=GkJiNn2QDF) [[code]](https://github.com/mhamilton723/FeatUp)[[Notes](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/FeatUp.md)]
2. [2024 CVPR] **EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation**[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf)[[code]](https://github.com/SLDGroup/EMCAD)   
3. [2025 CVPR] **SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.pdf) [[code]](https://github.com/likyoo/SegEarth-OV)[[Notes](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SegEarth.md)] 
4. [2025 NIPS] **InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition** [[paper]](https://arxiv.org/pdf/2505.15818) [[code]](https://github.com/VoyagerXvoyagerx/InstructSAM?tab=readme-ov-file)
5. [2025 arXiv] **AnyUp: Universal Feature Upsampling** [[paper]](https://arxiv.org/abs/2510.12764) [[code]](https://github.com/wimmerth/anyup)[[Notes](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/AnyUp.md)] 
6. [2025 ICME] **LG-CD: Enhancing Language-Guided Change Detection through SAM2 Adaptation** [[paper]](https://arxiv.org/pdf/2509.21894)
7. [2025 ICCV] **LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_LoftUp_Learning_a_Coordinate-Based_Feature_Upsampler_for_Vision_Foundation_Models_ICCV_2025_paper.pdf)
8. [2025 arXiv] **Benchmarking Feature Upsampling Methods for Vision Foundation Models using Interactive Segmentation(复现了LoftUp)** [[paper]](https://arxiv.org/pdf/2505.02075)[[code]](https://github.com/havrylovv/iSegProbe)

<a name="上采样与遥感"></a>  
## 上采样在遥感图像中的应用
**AnyUp和FeatUp这两种图像上采样方法都可以应用于遥感图像处理领域，包括遥感图像分割、检测、分类以及变化检测等任务。**

**对于遥感图像分割任务，这两种方法可以帮助提高分割精度和效率，特别是当原始图像分辨率较低或者存在噪声和遮挡等情况时。例如，可以通过将低分辨率的遥感图像上采样为高分辨率图像，然后利用深度学习模型进行像素级别的分类或语义分割，以获得更准确的结果。**

**对于遥感图像检测和分类任务，这两种方法也可以帮助提高识别率和鲁棒性。例如，可以通过将低分辨率的遥感图像上采样为高分辨率图像，然后利用深度学习模型进行目标检测或分类，以获得更高的准确性和可靠性。**

**对于遥感图像变化检测任务，这两种方法可以帮助捕捉地表的变化情况。例如，可以通过将历史遥感图像与当前遥感图像进行比较，利用图像上采样技术增强图像细节，然后利用深度学习模型进行变化检测，以发现地表的变化和演变趋势。**

**综上所述，AnyUp和FeatUp这两种图像上采样方法在遥感图像处理领域有着广泛的应用前景，可以帮助提高遥感图像处理的效率和准确性。**

**AnyUp和FeatUp是两种不同的图像上采样方法，它们的原理有所不同。**

AnyUp使用了一种基于卷积神经网络的方法，通过将低分辨率图像作为输入，在卷积层中逐步增加特征图的空间维度，最终得到高分辨率图像作为输出。具体来说，AnyUp使用了类似于图像放大（super-resolution）的技术，通过学习低分辨率图像到高分辨率图像之间的映射关系，从而实现图像上采样的目的。AnyUp还可以应用于多种不同的特征类型，并且可以用于各种下游任务，具有较高的灵活性和可扩展性。

FeatUp则是一种基于深度神经网络的图像上采样方法，它通过在低分辨率图像上进行训练来生成高分辨率图像。FeatUp使用了一个类似于编码器-解码器（encoder-decoder）的结构，其中编码器部分用于提取低分辨率图像中的特征，而解码器部分则用于将这些特征映射回高分辨率图像。FeatUp的优点是在某些情况下可以获得更好的图像质量，特别是在对细节要求较高的任务中。然而，FeatUp需要大量的计算资源和时间来进行训练，并且需要针对每个任务进行微调，因此其适用范围相对较小。

总的来说，AnyUp和FeatUp都是有效的图像上采样方法，但它们的原理和应用场景略有不同。


<a name="OVSS"></a>  
## OVSS
1. [2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.17150)[[code]](https://github.com/MICV-yonsei/CASS)
2. [2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation**[[paper]](https://arxiv.org/pdf/2411.17150)[[code]](https://github.com/MICV-yonsei/CASS?tab=readme-ov-file)
3. [2025 ICCV]**LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_LawDIS_Language-Window-based_Controllable_Dichotomous_Image_Segmentation_ICCV_2025_paper.pdf)[[code]](https://github.com/XinyuYanTJU/LawDIS)
4. [2025 ICCV]**CoralSRT: Revisiting Coral Reef Semantic Segmentation by Feature Rectification via Self-supervised Guidance**[[paper]](https://coralsrt.hkustvgd.com/papers/CoralSRT.pdf)[[code]](https://github.com/zhengziqiang/CoralSRT)
5. [2025 ICCV]**CLIP-Adapted Region-to-Text Learning for Generative Open-Vocabulary Semantic Segmentation**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Ge_CLIP-Adapted_Region-to-Text_Learning_for_Generative_Open-Vocabulary_Semantic_Segmentation_ICCV_2025_paper.pdf)

<a name="About_Features"></a>  
## Features are vital
1. [2025 CVPR] **DFM:Differentiable Feature Matching for Anomaly Detection**[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_DFM_Differentiable_Feature_Matching_for_Anomaly_Detection_CVPR_2025_paper.pdf)

<a name="Remote_Sensing"></a>  
## Remote Sensing
1. [2025 arXiv] **DynamicEarth: How Far are We from Open-Vocabulary Change Detection?** [[paper]](https://arXiv.org/abs/2501.12931) [[code]](https://github.com/likyoo/DynamicEarth)
2. [2025 Nature MI] **A semantic-enhanced multi-modal remote sensing foundation model for Earth observation**[[paper]](https://www.nature.com/articles/s42256-025-01078-8)[[code]](https://github.com/kang-wu/SkySensePlusPlus)
3. [2025 TGRS] **A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation.** [[paper]](https://ieeexplore.ieee.org/document/11063320) [[code]](https://github.com/sstary/SSRS)
4. [2025 ICASSP] **Enhancing Remote Sensing Vision-Language Models for Zero-Shot Scene Classification.** [[paper]](https://arXiv.org/abs/2409.00698) [[code]](https://github.com/elkhouryk/RS-TransCLIP)
5. [2025 ICCV] **Active Learning Meets Foundation Models: Fast Remote Sensing Data Annotation for Object Detection.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Burges_Active_Learning_Meets_Foundation_Models_Fast_Remote_Sensing_Data_Annotation_ICCV_2025_paper.pdf) [[code]](https://github.com/mburges-cvl/ICCV_AL4FM)
6. [2025 ICCV] **Dynamic Dictionary Learning for Remote Sensing Image Segmentation.** [[paper]](https://arXiv.org/pdf/2503.06683) [[code]](https://github.com/XavierJiezou/D2LS)
7. [2025 ICCV] **GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks.** [[paper]](https://arxiv.org/pdf/2411.19325) [[code]](https://github.com/The-AI-Alliance/GEO-Bench-VLM)
8. [2025 ICCV] **SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation.** [[paper]](https://arXiv.org/abs/2507.12857) [[code]](https://github.com/HuangShiqi128/SCORE)
9. [2025 ICCV] **When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning.** [[paper]](https://arXiv.org/pdf/2503.07588) [[code]](https://github.com/VisionXLab/LRS-VQA)
10. [2025 ICCV] **SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Sumbul_SMARTIES_Spectrum-Aware_Multi-Sensor_Auto-Encoder_for_Remote_Sensing_Images_ICCV_2025_paper.pdf) [[code]](https://github.com/gsumbul/SMARTIES)
11. [2025 ICCV] **Continuous Remote Sensing Image Super-Resolution via Neural Operator Diffusion** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_NeurOp-Diff_Continuous_Remote_Sensing_Image_Super-Resolution_via_Neural_Operator_Diffusion_ICCV_2025_paper.pdf) [[code]](https://github.com/zerono000/NeurOp-Diff)
12. [2025 ICCV] **HoliTracer: Holistic Vectorization of Geographic Objects from Large-Size Remote Sensing Imagery.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_HoliTracer_Holistic_Vectorization_of_Geographic_Objects_from_Large-Size_Remote_Sensing_ICCV_2025_paper.pdf) [[code]](https://github.com/vvangfaye/HoliTracer)[[Notes](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/HoliTracer.md)
13. [2025 ICCV] **RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Hu_RS-vHeat_Heat_Conduction_Guided_Efficient_Remote_Sensing_Foundation_Model_ICCV_2025_paper.pdf)该文章无开源代码，但是相同技术的代码链接为[[code]](https://github.com/MzeroMiko/vHeat)[[Notes](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/RS-vHeat.md)
14. [2025 ICCV] **OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf)[[code(暂时没有开源代码)]](https://github.com/floatingstarZ/OpenRSD)
15. [2025 arXiv] **SAR-KnowLIP: Towards Multimodal Foundation Models for Remote Sensing.** [[paper]](https://arxiv.org/pdf/2509.23927) [[code]](https://github.com/yangyifremad/SARKnowLIP)
16. [2025 AAAI] **ZoRI: Towards discriminative zero-shot remote sensing instance segmentation.** [[paper]](https://arXiv.org/abs/2412.12798) [[code]](https://github.com/HuangShiqi128/ZoRI)
17. [2024 NIPS] **Segment Any Change.** [[paper]](https://proceedings.NIPS.cc/paper_files/paper/2024/file/9415416201aa201902d1743c7e65787b-Paper-Conference.pdf) [[code]](https://github.com/Z-Zheng/pytorch-change-models)
18. [2025 CVPR] **SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images.** [[paper]](https://arXiv.org/abs/2410.01768) [[code]](https://github.com/likyoo/SegEarth-OV)
19. [2025 CVPR] **XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?** [[paper]](https://arXiv.org/abs/2503.23771) [[code]](https://github.com/EvolvingLMMs-Lab/XLRS-Bench)
20. [2025 CVPR] **Exact: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Exact_Exploring_Space-Time_Perceptive_Clues_for_Weakly_Supervised_Satellite_Image_CVPR_2025_paper.pdf) [[code]](https://github.com/MiSsU-HH/Exact)
21. [2025 Arxiv] **SegEarth-OV-2: Annotation-Free Open-Vocabulary Segmentation for Remote-Sensing Images** [[paper]](https://arxiv.org/abs/2508.18067)  [[code]](https://github.com/earth-insights/SegEarth-OV-2)
22. [2025 AAAI] **Towards Open-Vocabulary Remote Sensing Image Semantic Segmentation** [[paper]](https://arxiv.org/abs/2412.19492) [[code]](https://github.com/yecy749/GSNet)
23. [2025 Arxiv] **InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition** [[paper]](https://arxiv.org/pdf/2505.15818) [[code]](https://github.com/VoyagerXvoyagerx/InstructSAM)
24. [2025 Arxiv] **DescribeEarth: Describe Anything for Remote Sensing Images** [[paper]](https://arxiv.org/pdf/2509.25654v1) [[code]](https://github.com/earth-insights/DescribeEarth)
25. [2025 NIPS] **GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset** [[paper]](https://arxiv.org/abs/2507.14697) [[code]](https://github.com/Z-ZW-WXQ/GTPBD)
26. [2025 Arxiv] **RS3DBench: A Comprehensive Benchmark for 3D Spatial Perception in Remote Sensing** [[paper]](https://arxiv.org/abs/2509.18897) [[code]](https://rs3dbench.github.io)
27. [2025 Arxiv] **DGL-RSIS: Decoupling Global Spatial Context and Local Class Semantics for Training-Free Remote Sensing Image Segmentation** [[paper]](https://arxiv.org/pdf/2509.00598) [[code]](https://github.com/designer1024/DGL-RSIS)
28. [2025 TGRS] **A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection** [[paper]](https://ieeexplore.ieee.org/document/11172325) [[code]](https://github.com/fuyimin96/SAM-SPL)
29. [2025 TGRS] **Semantic Prototyping With CLIP for Few-Shot Object Detection in Remote Sensing Images** [[paper]](https://ieeexplore.ieee.org/document/10930588)
30. [2025 Arxiv] **ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild** [[paper]](https://arxiv.org/abs/2501.13354) [[code]](https://github.com/waterdisappear/ATRNet-STAR)
31. [2025 Arxiv] **RSKT-Seg: Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing** [[paper]](https://arxiv.org/pdf/2509.12040) [[code]](https://github.com/LiBingyu01/RSKT-Seg)
32. [2025 ISPRS]  **AdaptVFMs-RSCD: Advancing Remote Sensing Change Detection from binary to semantic with SAM and CLIP** [[paper]](https://doi.org/10.1016/j.isprsjprs.2025.09.010) [[data]](https://github.com/Jiang-CHD-YunNan/RS-VFMs-Fine-tuning-Dataset)
33. [2025 Arxiv]**PeftCD: Leveraging Vision Foundation Models with Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection** [[paper]](https://arxiv.org/pdf/2509.09572) [[code]](https://github.com/dyzy41/PeftCD)
34. [2025 Arxiv] **AlignCLIP: Self-Guided Alignment for Remote Sensing Open-Vocabulary Semantic Segmentation** [[paper]](https://openreview.net/forum?id=hpD3tn7Xbp) [[code]](https://openreview.net/attachment?id=hpD3tn7Xbp&name=supplementary_material)
35. [2025 Arxiv] **Few-Shot Adaptation Benchmark for Remote Sensing Vision-Language Models** [[paper]](https://arxiv.org/pdf/2510.07135) [[code]](https://github.com/elkhouryk/fewshot_RSVLMs)
36. [2025 RSE] **Strategic sampling for training a semantic segmentation model in operational mapping: Case studies on cropland parcel extraction** [[paper]](https://doi.org/10.1016/j.rse.2025.115034) [[data]](https://doi.org/10.5281/zenodo.16595511) [[code]](https://github.com/Remote-Sensing-of-Land-Resource-Lab/Training-Sample-Selection)
37. [2025 TIP] **Universal Fine-Grained Visual Categorization by Concept Guided Learning** [[paper]](https://ieeexplore.ieee.org/document/10829548) [[data]](https://drive.google.com/file/d/11hYbdO32hyspucDKp5wwjwvCaD38AEKe/view?usp=sharing) [[code]](https://github.com/BiQiWHU/CGL)
38. [2025 TIP] **SARATR-X: Towards Building A Foundation Model for SAR Target Recognition** [[paper]](https://ieeexplore.ieee.org/document/10856784) [[code]](https://github.com/waterdisappear/SARATR-X)
39. [2025 TIP] **HSLabeling: Towards Efficient Labeling for Large-scale Remote Sensing Image Segmentation with Hybrid Sparse Labeling** [[paper]](https://ieeexplore.ieee.org/document/10829548) [[data]](https://drive.google.com/drive/folders/1CiYzJyBn1rV-xsrsYQ6o2HDQjdfnadHl) [[code]](https://github.com/linjiaxing99/HSLabeling)
40. [2025 CVM] **Remote sensing tuning: A survey** [[paper]](https://ieeexplore.ieee.org/document/11119145) [[code]](https://github.com/DongshuoYin/Remote-Sensing-Tuning-A-Survey/tree/main)
41. [2025 ISPRS]**Domain generalization for semantic segmentation of remote sensing images via vision foundation model fine-tuning**[[paper]](https://www.sciencedirect.com/science/article/pii/S0924271625003569)[[code]](https://github.com/mmmll23/GeoSA-BaSA)
42. [2025 ISPRS]**Meta Feature Disentanglement under continuous-valued domain modeling for generalizable remote sensing image segmentation on unseen domains**[[paper]](https://www.sciencedirect.com/science/article/pii/S0924271625003879)[[code]](https://github.com/LCB1970/MetaFD)


<a name="classification"></a>  
## Classification
...
<a name="segmentation"></a>  
## Segmentation
1. [2025 ICCV] **Adapt Foundational Segmentation Models with Heterogeneous Searching Space**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yi_Adapt_Foundational_Segmentation_Models_with_Heterogeneous_Searching_Space_ICCV_2025_paper.pdf)[[code]](https://github.com/llipika/A2A-HSS)
2. 






