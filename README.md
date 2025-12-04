# SYH_GoGoGo
The following is the code for reproducing recently read papers and the work currently in progress.
# Content
1. [[Feature Upsampling](#Upsampling)]
2. [[上采样在遥感图像中的应用](#上采样与遥感)]
3. [[VLM](#VLM)]
4. [[VFM](#VFM)]
5. [[SAM](#SAM)]
6. [[OVSS](#OVSS)]
7. [[Features are vital](#About_Features)]
8. [[Remote Sensing](#Remote_Sensing)]
9. [[Classification](#Detection)]
10. [[Multi-Modal(VLMs)](#Multi_Modal)]
11. [[Segmentation](#segmentation)]
       
-----------------------------------------------------------------------------------------------
<a name="Upsampling"></a>  
## Feature Upsampling
1. [✨2024 ICLR] **FeatUp: A Model-Agnostic Framework for Features at Any Resolution** [[paper]](https://openreview.net/pdf?id=GkJiNn2QDF) [[code]](https://github.com/mhamilton723/FeatUp)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/FeatUp.md)
2. [2024 TPAMI] **Frequency-aware Feature Fusion for Dense Image Prediction** [[paper]](https://www.arxiv.org/pdf/2408.12879)[[code]](https://github.com/Linwei-Chen/FreqFusion)
3. [2025 arXiv] **AnyUp: Universal Feature Upsampling** [[paper]](https://arxiv.org/abs/2510.12764) [[code]](https://github.com/wimmerth/anyup)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/AnyUp.md)
4. [✨2025 ICCV] **LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_LoftUp_Learning_a_Coordinate-Based_Feature_Upsampler_for_Vision_Foundation_Models_ICCV_2025_paper.pdf)[[code]](https://github.com/andrehuang/loftup)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/LoftUp.md)
5. [2025 arXiv] **Benchmarking Feature Upsampling Methods for Vision Foundation Models using Interactive Segmentation(一种评估方法)** [[paper]](https://arxiv.org/pdf/2505.02075)[[code]](https://github.com/havrylovv/iSegProbe)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/Benchmarking.md)
6. [✨2025 ICML] **FeatSharp: Your Vision Model Features, Sharper** [[paper]](https://arxiv.org/pdf/2502.16025)[[code]](https://github.com/NVlabs/FeatSharp)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/FeatSharp.pdf)
7. [2025 arXiv] **BasicAVSR: Arbitrary-Scale Video Super-Resolution via Image Priors and Enhanced Motion Compensation** [[paper]](https://arxiv.org/pdf/2510.26149)[[code]](https://github.com/shangwei5/BasicAVSR)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/BasicAVSR.md)
8. [2025 NeurIPS] **JAFAR: Jack up Any Feature at Any Resolution** [[paper]](https://arxiv.org/pdf/2506.11136) [[code]](https://github.com/PaulCouairon/JAFAR)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/JAFAR.md)
9. [2025 NeurIPS] **DGSolver: Diffusion Generalist Solver with Universal Posterior Sampling for Image Restoration**[[paper]](https://arxiv.org/pdf/2504.21487)[[code]](https://github.com/MiliLab/DGSolver?tab=readme-ov-file)
10. [2025 arXiv] **Upsample Anything: A Simple and Hard to Beat Baseline for Feature Upsampling** [[paper]](https://arxiv.org/pdf/2511.16301)[[code]](https://github.com/seominseok0429/Upsample-Anything-A-Simple-and-Hard-to-Beat-Baseline-for-Feature-Upsampling)[[简单又强大的特征上采样基线]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/UPA.md)[[UPA的初体验]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/ExUPA.md)
 
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

## 几种上采样器的综合对比分析（FeatUp & AnyUp & LoftUp & FeatSharp）
[***以下是对四种特征上采样方法——AnyUp、FeatUp、LoftUp 和 FeatSharp 的详尽综合分析报告***](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/UpsamplerComparaiong.md)

![Upsampler](https://raw.githubusercontent.com/YanghuiSong/SYH_GoGoGo/main/UploadImage/Upsampler.png)

<a name="VLM"></a>  
## VLM
1. [2025 arXiv] **SigLIP 2: A better multilingual vision language encoder**[[paper]](https://arxiv.org/pdf/2502.14786)[[code]](https://github.com/huggingface/blog/blob/main/siglip2.md)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SigLIP2.md)[[与VFM融合的思路]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/RewithTalk2Dino.md)
2. [⭐ 2024 NIPS]**Boosting Vision-Language Models with Transduction**[[paper]](https://arxiv.org/pdf/2406.01837)[[code]](https://github.com/MaxZanella/transduction-for-vlms)[[TransCLIP的公式算法分析]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/TransCLIP.md)
3. [2025 NIPS] **Vision Transformers with Self-Distilled Registers**[[paper]](https://arxiv.org/pdf/2505.21501v3)[[code]](https://github.com/0raiser0/PH-Reg)[[解决伪影令牌问腿]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/PH_Reg.md)


<a name="VFM"></a>  
## VFM
1. [2025 Meta] **DINOv3**[[paper]](https://arxiv.org/pdf/2508.10104)[[code]](https://github.com/facebookresearch/dinov3)[[DINOv3 通过引入 Gram锚定训练机制，成功解决了大规模自监督学习中密集特征退化的问题，构建了一个无需微调即可在分类、分割、检测、深度估计等多样化任务上达到最先进性能的通用视觉编码器]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/DINOv3.md)

<a name="SAM"></a>  
## SAM
1. [⭐ 2023 Meta] **Segment Anything**[[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)[[code]](https://github.com/facebookresearch/segment-anything)[[SAM系列的对比]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCompare.md)[[SAM掩码生成详解]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMmask.md)[[SAM图解码器]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/SAMimage_encoder.md)[[SAM提示编码器详解]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/prompt_encoder.md)[[SAM掩码Decoder详解]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/SAMmask_decoder.md)[[SAM三个模块的联系]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/sam.md)[[自动掩码生成器详解]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/AMG.md)[[SAM的双向Transformer]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/TwoWayTransformer.md)
2. [2024 Meta] **SAM 2: Segment Anything in Images and Videos**[[paper]](https://arxiv.org/pdf/2408.00714)[[code]](https://github.com/facebookresearch/sam2)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAM2.md)
3. [⭐ 2025 Meta] **SAM 3: Segment Anything with Concepts** [[paper]](https://openreview.net/pdf?id=r35clVtGzw)[[code]](https://github.com/facebookresearch/sam3)[[SAM 3模型的改进思路]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAM3.md)[[SAM3的复现效果]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/ExSAM3.md)[[patch间固有问题对于存在性token可能的影响]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/IssueSAM1.md)[[实例编码器与图像编码器之间可能存在的gap]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/IssueSAM2.md)[[SAM3各层解析]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAM3Layers.md)
4. [2025 arXiv] **SAM3-Adapter: Efficient Adaptation of Segment Anything 3 for Camouflage Object Segmentation** [[paper]](https://arxiv.org/pdf/2511.19425)[[code]](https://github.com/tianrun-chen/SAM-Adapter-PyTorch)[[使用适配器解锁SAM3极精细边界的能力]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAM3_Adapter.md)



<a name="OVSS"></a>  
## OVSS
1. [✨2025 CVPR] **Distilling Spectral Graph for Object-Context Aware Open-Vocabulary Semantic Segmentation** [[paper]](https://arxiv.org/pdf/2411.17150)[[code]](https://github.com/MICV-yonsei/CASS)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/CASS.md)[[与遥感图像、特征上采样的联系]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/CASSinRemoteSensing.md)
2. [2025 CVPR] **Open-Canopy: Towards Very High Resolution Forest Monitoring**[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Fogel_Open-Canopy_Towards_Very_High_Resolution_Forest_Monitoring_CVPR_2025_paper.pdf)[[code]](https://github.com/fajwel/Open-Canopy)
3. [2025 CVPR] **Hybrid Global-Local Representation with Augmented Spatial Guidance for Zero-Shot Referring Image Segmentation**[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Hybrid_Global-Local_Representation_with_Augmented_Spatial_Guidance_for_Zero-Shot_Referring_CVPR_2025_paper.pdf)[[code]](https://github.com/fhgyuanshen/HybridGL)
4. [2025 ICCV] **LawDIS: Language-Window-based Controllable Dichotomous Image Segmentation**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_LawDIS_Language-Window-based_Controllable_Dichotomous_Image_Segmentation_ICCV_2025_paper.pdf)[[code]](https://github.com/XinyuYanTJU/LawDIS)
5. [2025 ICCV] **CoralSRT: Revisiting Coral Reef Semantic Segmentation by Feature Rectification via Self-supervised Guidance**[[paper]](https://coralsrt.hkustvgd.com/papers/CoralSRT.pdf)[[code]](https://github.com/zhengziqiang/CoralSRT)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/CoralSRT.md)
6. [2025 ICCV] **CLIP-Adapted Region-to-Text Learning for Generative Open-Vocabulary Semantic Segmentation**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Ge_CLIP-Adapted_Region-to-Text_Learning_for_Generative_Open-Vocabulary_Semantic_Segmentation_ICCV_2025_paper.pdf)
7. [2025 ICCV] **Talking to DINO: Bridging Self-Supervised Vision Backbones with Language for Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.19331)[[code]](https://github.com/lorebianchi98/Talk2DINO)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/Talk2Dino.md)[[与CASS进行对比]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/CompWithCASS.md)
8. [2025 ICCV] **CorrCLIP: Reconstructing Patch Correlations in CLIP for OVSS** [[paper]](https://arxiv.org/pdf/2411.10086)[[code]](https://github.com/zdk258/CorrCLIP)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/new/main/Notes/CorrCLIP.md)
9. [2024 arXiv] **Self-Calibrated CLIP for Training-Free Open-Vocabulary Segmentation** [[paper]](https://arxiv.org/pdf/2411.15869)[[code]](https://github.com/SuleBai/SC-CLIP)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SelfCLIP.md)
10. [2026 AAAI] **Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing** [[paper]](https://arxiv.org/pdf/2509.12040)[[code]](https://github.com/LiBingyu01/RSKT-Seg?tab=readme-ov-file)
11. [2025 CVPR] **SkySense-O: Towards Open-World Remote Sensing Interpretation with Vision-Centric Visual-Language Modeling**[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_SkySense-O_Towards_Open-World_Remote_Sensing_Interpretation_with_Vision-Centric_Visual-Language_Modeling_CVPR_2025_paper.pdf)[[code]](https://github.com/zqcrafts/SkySense-O)

<a name="About_Features"></a>  
## Features are vital
1. [2025 CVPR] **DFM:Differentiable Feature Matching for Anomaly Detection**[[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Wu_DFM_Differentiable_Feature_Matching_for_Anomaly_Detection_CVPR_2025_paper.pdf)

<a name="Remote_Sensing"></a>  
## Remote Sensing
1. [2025 arXiv] **DynamicEarth: How Far are We from Open-Vocabulary Change Detection?** [[paper]](https://arXiv.org/abs/2501.12931) [[code]](https://github.com/likyoo/DynamicEarth)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/DynamicEarth.md)
2. [✨2025 CVPR] **SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_SegEarth-OV_Towards_Training-Free_Open-Vocabulary_Segmentation_for_Remote_Sensing_Images_CVPR_2025_paper.pdf) [[code]](https://github.com/likyoo/SegEarth-OV)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SegEarth.md)
3. [2025 arXiv] **SegEarth-R1: Geospatial Pixel Reasoning via Large Language Model** [[paper]](https://arxiv.org/pdf/2504.09644)[[code]](https://github.com/earth-insights/SegEarth-R1)
4. [2025 arXiv] **Annotation-Free Open-Vocabulary Segmentation for Remote-Sensing Images**[[paper]](https://arxiv.org/pdf/2508.18067)[[code]](https://github.com/earth-insights/SegEarth-OV-2)[[这篇文章把SegEarth拓展到了SAR图像，两种类型的图像区别是]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SARvsRemoteSensing.md)
5. [2025 arXiv] **FoBa: A Foreground-Background co-Guided Method and New Benchmark for Remote Sensing Semantic Change Detection** [[paper]](https://arxiv.org/pdf/2509.15788)[[code]](https://github.com/zmoka-zht/FoBa)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/FoBa.pdf)
6. [2025 Nature MI] **A semantic-enhanced multi-modal remote sensing foundation model for Earth observation**[[paper]](https://www.nature.com/articles/s42256-025-01078-8)[[code]](https://github.com/kang-wu/SkySensePlusPlus)
7. [2025 TGRS] **A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation.** [[paper]](https://ieeexplore.ieee.org/document/11063320) [[code]](https://github.com/sstary/SSRS)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/UnifiedFram.pdf)
8. [2025 ICASSP] **Enhancing Remote Sensing Vision-Language Models for Zero-Shot Scene Classification.** [[paper]](https://arXiv.org/abs/2409.00698) [[code]](https://github.com/elkhouryk/RS-TransCLIP)
9. [2025 ICCV] **Active Learning Meets Foundation Models: Fast Remote Sensing Data Annotation for Object Detection.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Burges_Active_Learning_Meets_Foundation_Models_Fast_Remote_Sensing_Data_Annotation_ICCV_2025_paper.pdf) [[code]](https://github.com/mburges-cvl/ICCV_AL4FM)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/ActiveLearn.md)
10. [2025 ICCV] **Dynamic Dictionary Learning for Remote Sensing Image Segmentation.** [[paper]](https://arXiv.org/pdf/2503.06683) [[code]](https://github.com/XavierJiezou/D2LS)
11. [2025 ICCV] **GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks.** [[paper]](https://arxiv.org/pdf/2411.19325) [[code]](https://github.com/The-AI-Alliance/GEO-Bench-VLM)
12. [2025 ICCV] **SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation.** [[paper]](https://arXiv.org/abs/2507.12857) [[code]](https://github.com/HuangShiqi128/SCORE)
13. [2025 ICCV] **When Large Vision-Language Model Meets Large Remote Sensing Imagery: Coarse-to-Fine Text-Guided Token Pruning.** [[paper]](https://arXiv.org/pdf/2503.07588) [[code]](https://github.com/VisionXLab/LRS-VQA)
14. [2025 ICCV] **SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Sumbul_SMARTIES_Spectrum-Aware_Multi-Sensor_Auto-Encoder_for_Remote_Sensing_Images_ICCV_2025_paper.pdf) [[code]](https://github.com/gsumbul/SMARTIES)
15. [2025 ICCV] **Continuous Remote Sensing Image Super-Resolution via Neural Operator Diffusion** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_NeurOp-Diff_Continuous_Remote_Sensing_Image_Super-Resolution_via_Neural_Operator_Diffusion_ICCV_2025_paper.pdf) [[code]](https://github.com/zerono000/NeurOp-Diff)
16. [2025 ICCV] **HoliTracer: Holistic Vectorization of Geographic Objects from Large-Size Remote Sensing Imagery.** [[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_HoliTracer_Holistic_Vectorization_of_Geographic_Objects_from_Large-Size_Remote_Sensing_ICCV_2025_paper.pdf) [[code]](https://github.com/vvangfaye/HoliTracer)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/HoliTracer.md)
17. [2025 ICCV] **RS-vHeat: Heat Conduction Guided Efficient Remote Sensing Foundation Model**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Hu_RS-vHeat_Heat_Conduction_Guided_Efficient_Remote_Sensing_Foundation_Model_ICCV_2025_paper.pdf)该文章无开源代码，但是相同技术的代码链接为[[code]](https://github.com/MzeroMiko/vHeat)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/RS-vHeat.md)
18. [2025 ICCV] **OpenRSD: Towards Open-prompts for Object Detection in Remote Sensing Images**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Huang_OpenRSD_Towards_Open-prompts_for_Object_Detection_in_Remote_Sensing_Images_ICCV_2025_paper.pdf)[[code(暂时没有开源代码)]](https://github.com/floatingstarZ/OpenRSD)
19. [2024 CVPR] **SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.pdf)[[code]](https://github.com/Jack-bo1220/SkySense?tab=readme-ov-file)[[因子化时空编码器]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SkySense.md)
20. [2025 ICCV] **SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing** [[paper]](https://arxiv.org/pdf/2507.13812)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SkySense2.md)
21. [2025 ICME] **LG-CD: Enhancing Language-Guided Change Detection through SAM2 Adaptation** [[paper]](https://arxiv.org/pdf/2509.21894)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/LG.pdf)
22. [2025 arXiv] **SAR-KnowLIP: Towards Multimodal Foundation Models for Remote Sensing.** [[paper]](https://arxiv.org/pdf/2509.23927) [[code]](https://github.com/yangyifremad/SARKnowLIP)
23. [2025 AAAI] **ZoRI: Towards discriminative zero-shot remote sensing instance segmentation.** [[paper]](https://arXiv.org/abs/2412.12798) [[code]](https://github.com/HuangShiqi128/ZoRI)
24. [2024 NIPS] **Segment Any Change.** [[paper]](https://proceedings.NIPS.cc/paper_files/paper/2024/file/9415416201aa201902d1743c7e65787b-Paper-Conference.pdf) [[code]](https://github.com/Z-Zheng/pytorch-change-models)
25. [✨2025 NIPS] **InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition** [[paper]](https://arxiv.org/pdf/2505.15818) [[code]](https://github.com/VoyagerXvoyagerx/InstructSAM?tab=readme-ov-file)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/InstructSAM.md)[[和TTAOD的联系]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/InstructSAMwithTT.md)
26. [2025 NIPS] **Bringing SAM to new heights: Leveraging elevation data for tree crown segmentation from drone imagery** [[paper]](https://arxiv.org/pdf/2506.04970)
27. [2025 NIPS] **RSCC: A Large-Scale Remote Sensing Change Caption Dataset for Disaster Events** [[paper]](https://arxiv.org/pdf/2509.01907)[[code]](https://github.com/Bili-Sakura/RSCC)
28. [2025 CVPR] **SegEarth-OV: Towards Training-Free Open-Vocabulary Segmentation for Remote Sensing Images.** [[paper]](https://arXiv.org/abs/2410.01768) [[code]](https://github.com/likyoo/SegEarth-OV)
29. [2025 CVPR] **XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?** [[paper]](https://arXiv.org/abs/2503.23771) [[code]](https://github.com/EvolvingLMMs-Lab/XLRS-Bench)
30. [2025 CVPR] **Exact: Exploring Space-Time Perceptive Clues for Weakly Supervised Satellite Image Time Series Semantic Segmentation.** [[paper]](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhu_Exact_Exploring_Space-Time_Perceptive_Clues_for_Weakly_Supervised_Satellite_Image_CVPR_2025_paper.pdf) [[code]](https://github.com/MiSsU-HH/Exact)
31. [2025 Arxiv] **SegEarth-OV-2: Annotation-Free Open-Vocabulary Segmentation for Remote-Sensing Images** [[paper]](https://arxiv.org/abs/2508.18067)  [[code]](https://github.com/earth-insights/SegEarth-OV-2)
32. [2025 AAAI] **Towards Open-Vocabulary Remote Sensing Image Semantic Segmentation** [[paper]](https://arxiv.org/abs/2412.19492) [[code]](https://github.com/yecy749/GSNet)
33. [2025 Arxiv] **InstructSAM: A Training-Free Framework for Instruction-Oriented Remote Sensing Object Recognition** [[paper]](https://arxiv.org/pdf/2505.15818) [[code]](https://github.com/VoyagerXvoyagerx/InstructSAM)
34. [2025 Arxiv] **DescribeEarth: Describe Anything for Remote Sensing Images** [[paper]](https://arxiv.org/pdf/2509.25654v1) [[code]](https://github.com/earth-insights/DescribeEarth)
35. [2025 NIPS] **GTPBD: A Fine-Grained Global Terraced Parcel and Boundary Dataset** [[paper]](https://arxiv.org/abs/2507.14697) [[code]](https://github.com/Z-ZW-WXQ/GTPBD)
36. [2025 Arxiv] **RS3DBench: A Comprehensive Benchmark for 3D Spatial Perception in Remote Sensing** [[paper]](https://arxiv.org/abs/2509.18897) [[code]](https://rs3dbench.github.io)
37. [2025 Arxiv] **DGL-RSIS: Decoupling Global Spatial Context and Local Class Semantics for Training-Free Remote Sensing Image Segmentation** [[paper]](https://arxiv.org/pdf/2509.00598) [[code]](https://github.com/designer1024/DGL-RSIS)
38. [2025 TGRS] **A Unified SAM-Guided Self-Prompt Learning Framework for Infrared Small Target Detection** [[paper]](https://ieeexplore.ieee.org/document/11172325) [[code]](https://github.com/fuyimin96/SAM-SPL)
39. [2025 TGRS] **Semantic Prototyping With CLIP for Few-Shot Object Detection in Remote Sensing Images** [[paper]](https://ieeexplore.ieee.org/document/10930588)
40. [2025 TGRS] **Multimodal Visual-Language Prompt Network for Remote Sensing Few-Shot Segmentation** [[paper]](https://ieeexplore.ieee.org/abstract/document/11071646)[[code]](https://github.com/Gritiii/MVLPNet)
41. [2025 Arxiv] **ATRNet-STAR: A Large Dataset and Benchmark Towards Remote Sensing Object Recognition in the Wild** [[paper]](https://arxiv.org/abs/2501.13354) [[code]](https://github.com/waterdisappear/ATRNet-STAR)
42. [2025 Arxiv] **RSKT-Seg: Exploring Efficient Open-Vocabulary Segmentation in the Remote Sensing** [[paper]](https://arxiv.org/pdf/2509.12040) [[code]](https://github.com/LiBingyu01/RSKT-Seg)
43. [2025 ISPRS]  **AdaptVFMs-RSCD: Advancing Remote Sensing Change Detection from binary to semantic with SAM and CLIP** [[paper]](https://doi.org/10.1016/j.isprsjprs.2025.09.010) [[data]](https://github.com/Jiang-CHD-YunNan/RS-VFMs-Fine-tuning-Dataset)
44. [2025 Arxiv]**PeftCD: Leveraging Vision Foundation Models with Parameter-Efficient Fine-Tuning for Remote Sensing Change Detection** [[paper]](https://arxiv.org/pdf/2509.09572) [[code]](https://github.com/dyzy41/PeftCD)
45. [2025 Arxiv] **AlignCLIP: Self-Guided Alignment for Remote Sensing Open-Vocabulary Semantic Segmentation** [[paper]](https://openreview.net/forum?id=hpD3tn7Xbp) [[code]](https://openreview.net/attachment?id=hpD3tn7Xbp&name=supplementary_material)
46. [2025 Arxiv] **Few-Shot Adaptation Benchmark for Remote Sensing Vision-Language Models** [[paper]](https://arxiv.org/pdf/2510.07135) [[code]](https://github.com/elkhouryk/fewshot_RSVLMs)
47. [2025 RSE] **Strategic sampling for training a semantic segmentation model in operational mapping: Case studies on cropland parcel extraction** [[paper]](https://doi.org/10.1016/j.rse.2025.115034) [[data]](https://doi.org/10.5281/zenodo.16595511) [[code]](https://github.com/Remote-Sensing-of-Land-Resource-Lab/Training-Sample-Selection)
48. [2025 TIP] **Universal Fine-Grained Visual Categorization by Concept Guided Learning** [[paper]](https://ieeexplore.ieee.org/document/10829548) [[data]](https://drive.google.com/file/d/11hYbdO32hyspucDKp5wwjwvCaD38AEKe/view?usp=sharing) [[code]](https://github.com/BiQiWHU/CGL)
49. [2025 TIP] **SARATR-X: Towards Building A Foundation Model for SAR Target Recognition** [[paper]](https://ieeexplore.ieee.org/document/10856784) [[code]](https://github.com/waterdisappear/SARATR-X)
50. [2025 TIP] **HSLabeling: Towards Efficient Labeling for Large-scale Remote Sensing Image Segmentation with Hybrid Sparse Labeling** [[paper]](https://ieeexplore.ieee.org/document/10829548) [[data]](https://drive.google.com/drive/folders/1CiYzJyBn1rV-xsrsYQ6o2HDQjdfnadHl) [[code]](https://github.com/linjiaxing99/HSLabeling)
51. [2025 CVM] **Remote sensing tuning: A survey** [[paper]](https://ieeexplore.ieee.org/document/11119145) [[code]](https://github.com/DongshuoYin/Remote-Sensing-Tuning-A-Survey/tree/main)
52. [2025 ISPRS]**Domain generalization for semantic segmentation of remote sensing images via vision foundation model fine-tuning**[[paper]](https://www.sciencedirect.com/science/article/pii/S0924271625003569)[[code]](https://github.com/mmmll23/GeoSA-BaSA)
53. [2025 ISPRS]**Meta Feature Disentanglement under continuous-valued domain modeling for generalizable remote sensing image segmentation on unseen domains**[[paper]](https://www.sciencedirect.com/science/article/pii/S0924271625003879)[[code]](https://github.com/LCB1970/MetaFD)
54. [2026 AAAI] **LWGANet: Addressing Spatial and Channel Redundancy in Remote Sensing Visual Tasks with Light-Weight Grouped Attention** [[paper]](https://arxiv.org/pdf/2501.10040)[[code]](https://github.com/AeroVILab-AHU/LWGANet)[[轻量化去除空间冗余和通道冗余]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/LWGA.md)


<a name="Detection"></a>  
## Change Detection
1. [2025 arXiv] **FoBa: A Foreground-Background co-Guided Method and New Benchmark for Remote Sensing Semantic Change Detection** [[paper]](https://arxiv.org/pdf/2509.15788)[[code]](https://github.com/zmoka-zht/FoBa)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/FoBa.pdf)
2. [2025 AAAI] **SM3Det: A Unified Model for Multi-Modal Remote Sensing Object Detection** [[paper]](https://arxiv.org/pdf/2412.20665)[[code]](https://github.com/zcablii/SM3Det)[[Datasets]](https://www.kaggle.com/datasets/greatbird/soi-det)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/tree/main/Notes)

<a name="Multi_Modal"></a>  
## Multi-Modal(VLMs)
1. [2024 CVPR] **SkySense: A Multi-Modal Remote Sensing Foundation Model Towards Universal Interpretation for Earth Observation Imagery** [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Guo_SkySense_A_Multi-Modal_Remote_Sensing_Foundation_Model_Towards_Universal_Interpretation_CVPR_2024_paper.pdf)[[code]](https://github.com/Jack-bo1220/SkySense?tab=readme-ov-file)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SkySense.md)
2. [2025 NMI] **SkySense++:A Semantic-Enhanced Multi-Modal Remote Sensing Foundation Model for Earth Observation** [[paper]](https://www.nature.com/articles/s42256-025-01078-8)[[code]](https://github.com/kang-wu/SkySensePlusPlus)
3. [2025 ICCV] **SkySense V2: A Unified Foundation Model for Multi-modal Remote Sensing** [[paper]](https://arxiv.org/pdf/2507.13812)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SkySense2.md)
4. [2025 ICCV] **Hybrid-grained Feature Aggregation with Coarse-to-fine Language Guidance for Self-supervised Monocular Depth Estimation**[[paper]](https://arxiv.org/pdf/2510.09320)[[code]](https://github.com/Zhangwenyao1/Hybrid-depth)
5. [2024 arXiv] **SkySenseGPT: A Fine-Grained Instruction Tuning Dataset and Model for Remote Sensing Vision-Language Understanding** [[paper]]()[[code]](https://github.com/Luo-Z13/SkySenseGPT/tree/main)
6. [2025 NIPS] **Test-Time Adaptive Object Detection with Foundation Model**[[paper]](https://arxiv.org/pdf/2510.25175)[[code]](https://github.com/gaoyingjay/ttaod_foundation)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/TestTime.md)
7. [2025 NIPS] **Conditional Representation Learning for Customized Tasks** [[paper]](https://arxiv.org/pdf/2510.04564)[[code]](https://github.com/XLearning-SCU/2025-NeurIPS-CRL)
8. [2025 ACM Multimedia] **KAID: Knowledge-Aware Interactive Distillation for Vision-Language Models**[[paper]](https://dl.acm.org/doi/pdf/10.1145/3746027.3755008)
9. [2024 ITGRS]**RS5M and GeoRSCLIP: A Large-Scale Vision- Language Dataset and a Large Vision-Language Model for Remote Sensing**[[paper]](https://ieeexplore.ieee.org/document/10679571)[[code]](https://github.com/om-ai-lab/RS5M)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/RS5M.pdf)
10. [2024 ITGRS]**Remoteclip: A vision language foundation model for remote sensing**[[paper]](https://ieeexplore.ieee.org/document/10504785)[[code]](https://github.com/ChenDelong1999/RemoteCLIP)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/RemoteCLIP.pdf)
11. [2025 arXiv] **ZoomEarth: Active Perception for Ultra-High-Resolution Geospatial Vision-Language Tasks** [[paper]](https://arxiv.org/pdf/2511.12267)[[code]](https://github.com/earth-insights/ZoomEarth)[[面向超高分辨率遥感图像的主动感知]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/ZoomEarth.md)
12. [2024 arXiv]**Enhancing Remote Sensing Vision-Language Models for Zero-Shot Scene Classification**[[paper]](https://arxiv.org/pdf/2409.00698)[[code]](https://github.com/elkhouryk/RS-TransCLIP)
13. [2024 TGRS] **Exploring Fine-Grained Image-Text Alignment for Referring Remote Sensing Image Segmentation** [[paper]](https://arxiv.org/pdf/2409.13637)[[code]](https://github.com/Shaosifan/FIANet)[[通过图像-文本对齐捕捉到区分性的多模态特征]]()
14. [2025 TGRS] **A Unified Framework With Multimodal Fine-Tuning for Remote Sensing Semantic Segmentation** [[paper]](https://ieeexplore.ieee.org/document/11063320)[[code]](https://github.com/sstary/SSRS)[[融合 MMAdapterMMLoRA，让 SAM 在 DSM 数据上高效落地遥感语义分割]]()

<a name="segmentation"></a>  
## Segmentation
1. [2025 ICCV] **Adapt Foundational Segmentation Models with Heterogeneous Searching Space**[[paper]](https://openaccess.thecvf.com/content/ICCV2025/papers/Yi_Adapt_Foundational_Segmentation_Models_with_Heterogeneous_Searching_Space_ICCV_2025_paper.pdf)[[code]](https://github.com/llipika/A2A-HSS)
2. [2024 CVPR] **EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation**[[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.pdf)[[code]](https://github.com/SLDGroup/EMCAD)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/EMCAD.md)
3. [⭐ 2023 Meta] **Segment Anything**[[paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Kirillov_Segment_Anything_ICCV_2023_paper.pdf)[[code]](https://github.com/facebookresearch/segment-anything)[[SAM系列的对比]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCompare.md)[[SAM掩码生成详解]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMmask.md)[[SAM图解码器]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/SAMimage_encoder.md)[[SAM提示编码器详解]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/prompt_encoder.md)[[SAM掩码Decoder详解]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/SAMmask_decoder.md)[[SAM三个模块的联系]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/sam.md)[[自动掩码生成器详解]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/AMG.md)[[SAM的双向Transformer]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAMCodeLearn/TwoWayTransformer.md)
4. [2024 Meta] **SAM 2: Segment Anything in Images and Videos**[[paper]](https://arxiv.org/pdf/2408.00714)[[code]](https://github.com/facebookresearch/sam2)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAM2.md)
5. [⭐ 2025 Meta] **SAM 3: Segment Anything with Concepts** [[paper]](https://openreview.net/pdf?id=r35clVtGzw)[[code]](https://github.com/facebookresearch/sam3)[[SAM 3模型的改进思路]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/SAM3.md)[[SAM3的复现效果]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/ExSAM3.md)[[patch间固有问题对于存在性token可能的影响]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/IssueSAM1.md)[[实例编码器与图像编码器之间可能存在的gap]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/IssueSAM2.md)
6. [2025 AAAI] **ZoRI: Towards Discriminative Zero-Shot Remote Sensing Instance Segmentation** [[paper]](https://arxiv.org/pdf/2412.12798)[[code]](https://github.com/HuangShiqi128/ZoRI)
7. [2025 CVPR] **ROS-SAM: High-Quality Interactive Segmentation for Remote Sensing Moving Object** [[paper]](https://arxiv.org/pdf/2503.12006)[[code]](https://github.com/ShanZard/ROS-SAM)[[Notes]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/ROSSAM.md)
8. [2025 NIPS] **UniPixel: Unified Object Referring and Segmentation for Pixel-Level Visual Reasoning** [[paper]](https://arxiv.org/pdf/2509.18094)[[code]](https://github.com/PolyU-ChenLab/UniPixel)[[内统一对象指代（Referring）和分割（Segmentation）两大能力]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/UniPixel.md)
9. [2025 NIPS] **Seg2Any: Open-set Segmentation-Mask-to-Image Generation with Precise Shape and Semantic Control**[[paper]](https://arxiv.org/pdf/2506.00596)[[code]](https://github.com/0xLDF/Seg2Any)[[这是一个掩码到图像的逆过程基于多模态扩散变换器的分割掩码到图像生成框架]](https://github.com/YanghuiSong/SYH_GoGoGo/blob/main/Notes/Seg2Any.md)






