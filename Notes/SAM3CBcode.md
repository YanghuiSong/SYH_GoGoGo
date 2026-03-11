# 分析Concept Bank的代码
## 运行Concept Bank得到的新的概念库文件
四个遥感数据均是按照[SegEarth](https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md)所给的数据处理文件和教程完成的处理

```python
python sam3_concept_bank.py   --users "loveda=configs/cfg_loveda.py,potsdam=configs/cfg_potsdam.py,vaihingen=configs/cfg_vaihingen.py,isaid=configs/cfg_isaid.py"   --split train   --checkpoint_path /data/public/sam3/sam3.pt   --bpe_path /data/public/sam3/assets/bpe_simple_vocab_16e6.txt.gz   --pad_ratio 0.05   --tau_w 0.15   --bg_thr_mode dice   --output_pt "./configs/concept_bank/new_cb_sam3_rs.pt"
/data/users/ConceptBank/mmengine/optim/optimizer/zero_optimizer.py:11: DeprecationWarning: `TorchScript` support for functional optimizers is deprecated and will be removed in a future PyTorch release. Consider using the `torch.compile` optimizer instead.
  from torch.distributed.optim import \
[INFO] World Size: 1, Seed: 0
[INFO] Processing Users: [('loveda', 'configs/cfg_loveda.py'), ('potsdam', 'configs/cfg_potsdam.py'), ('vaihingen', 'configs/cfg_vaihingen.py'), ('isaid', 'configs/cfg_isaid.py')]

========== [USER] loveda ==========
[INFO] cfg=configs/cfg_loveda.py, name_path=./configs/ext_cls/cls_loveda.txt, C=7, N=2522
[AUTO] proto=10, cap1=3, cap2=3, cache=16384
loveda:Pass1(r0):  10%|█████████████████████████▊                                                                                                                                                                                                                                   | 257/2522 [00:52<07:44,  4.88it/s]
[loveda] Pass1 Done. Valid protos: 7/7
loveda:Pass2(e0,r0):   1%|███▌                                                                                                                                                                                                                                                       | 36/2522 [00:07<08:11,  5.06it/s]
loveda:Select(r0): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:23<00:00,  3.36s/it]
[PROFILE][loveda] S1 0.88m (700 crops) | S2 0.12m (70 crops) | S3 0.39m (70 crops) | Total 1.40m

========== [USER] potsdam ==========
[INFO] cfg=configs/cfg_potsdam.py, name_path=./configs/ext_cls/cls_potsdam.txt, C=6, N=3456
[AUTO] proto=50, cap1=3, cap2=3, cache=16384
potsdam:Pass1(r0):   8%|███████████████████▍                                                                                                                                                                                                                                        | 267/3456 [00:35<07:04,  7.51it/s]
[potsdam] Pass1 Done. Valid protos: 6/6
potsdam:Pass2(e0,r0):   4%|█████████▎                                                                                                                                                                                                                                               | 129/3456 [00:19<08:22,  6.62it/s]
potsdam:Select(r0): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:44<00:00,  7.34s/it]
[PROFILE][potsdam] S1 0.59m (600 crops) | S2 0.33m (300 crops) | S3 0.73m (300 crops) | Total 1.66m

========== [USER] vaihingen ==========
[INFO] cfg=configs/cfg_vaihingen.py, name_path=./configs/ext_cls/cls_vaihingen.txt, C=6, N=344
[AUTO] proto=50, cap1=3, cap2=3, cache=16384
vaihingen:Pass1(r0): 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 344/344 [00:18<00:00, 18.92it/s]
[vaihingen] Pass1 Done. Valid protos: 6/6
vaihingen:Pass2(e0,r0): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 344/344 [00:20<00:00, 17.04it/s]
vaihingen:Pass2(e1,r0):  71%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▏                                                                        | 243/344 [00:02<00:01, 97.63it/s]
vaihingen:Select(r0): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6/6 [00:46<00:00,  7.73s/it]
[PROFILE][vaihingen] S1 0.30m (275 crops) | S2 0.38m (300 crops) | S3 0.77m (300 crops) | Total 1.46m

========== [USER] isaid ==========
[INFO] cfg=configs/cfg_isaid.py, name_path=./configs/ext_cls/cls_isaid.txt, C=16, N=33978
[AUTO] proto=10, cap1=3, cap2=3, cache=16384
isaid:Pass1(r0): 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33978/33978 [02:54<00:00, 194.86it/s]
[isaid] Pass1 Done. Valid protos: 2/16
isaid:Pass2(e0,r0):   0%|                                                                                                                                                                                                                                                         | 12/33978 [00:01<1:03:32,  8.91it/s]
isaid:Select(r0): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:04<00:00,  3.89it/s]
[PROFILE][isaid] S1 2.91m (200 crops) | S2 0.02m (20 crops) | S3 0.07m (20 crops) | Total 3.00m
[OK] Saved to ./configs/concept_bank/new_cb_sam3_rs.pt
[PROFILE][ALL USERS] Total build time: 7.54m
```
## 使用新的概念库完成评估
以下是我重新完成概念库的训练得到的结果
```python
=============== Evaluation Summary ===============
Dataset           aAcc         mIoU         mAcc       
---------------- -------      -------      -------     
loveda           62.7000      47.9100      64.3900     
potsdam          74.1300      57.5600      73.7200     
vaihingen        82.7000      63.6400      80.4700     
isaid            96.2100      32.3200      56.3400     
--------------------------------------------------
MEAN             78.9350      50.3575      68.7300          
```
不难看出自行训练得到的概念库与原始的概念库在除了vaihingen数据集之外性能之间差异明显

以下是原作者的结果
```python
=============== Evaluation Summary ===============
Dataset           aAcc         mIoU         mAcc       
---------------- -------      -------      -------     
loveda           65.0200      49.3800      65.8500     
potsdam          77.4800      60.5100      74.8100     
vaihingen        82.5800      62.9700      80.8500     
isaid            97.0400      35.4400      53.3800     
--------------------------------------------------
MEAN             80.5300      52.0750      68.7225
```     


## 各类单独分析
### LoveDA原始的概念库
```python
+--------------+-------+-------+
|    Class     |  IoU  |  Acc  |
+--------------+-------+-------+
|  background  | 45.73 | 67.09 |
|   building   | 63.92 |  80.0 |
|     road     | 52.62 | 71.55 |
|    water     | 63.13 | 71.39 |
|    barren    | 36.97 | 61.79 |
|    forest    | 36.04 | 49.77 |
| agricultural | 47.24 | 59.39 |
+--------------+-------+-------+
03/08 14:56:03 - mmengine - INFO - Iter(test) [418/418]    aAcc: 65.0200  mIoU: 49.3800  mAcc: 65.8500  data_time: 0.0099  time: 0.2214
```

### LoveDA自训练的概念库
```python
+--------------+-------+-------+
|    Class     |  IoU  |  Acc  |
+--------------+-------+-------+
|  background  | 40.66 | 57.24 |
|   building   | 64.19 | 82.87 |
|     road     | 50.02 | 66.57 |
|    water     | 62.29 | 69.68 |
|    barren    | 37.25 | 63.74 |
|    forest    | 35.67 | 46.34 |
| agricultural |  45.6 | 65.11 |
+--------------+-------+-------+
03/08 15:42:23 - mmengine - INFO - Iter(test) [418/418]    aAcc: 62.7100  mIoU: 47.9600  mAcc: 64.5100  data_time: 0.0096  time: 0.2248
```

### Potsdam原始的概念库结果
```python
+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
| impervious_surface |  73.1 | 84.92 |
|      building      | 84.87 |  92.7 |
|   low_vegetation   | 57.66 | 78.96 |
|        tree        | 42.96 | 45.14 |
|        car         | 84.53 | 97.41 |
|      clutter       | 19.92 | 49.73 |
+--------------------+-------+-------+
03/08 14:58:09 - mmengine - INFO - Iter(test) [504/504]    aAcc: 77.4800  mIoU: 60.5100  mAcc: 74.8100  data_time: 0.0032  time: 0.1837
```
### Potsdam自训练的概念库结果
```python
+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
| impervious_surface | 72.37 | 86.76 |
|      building      | 79.26 | 81.26 |
|   low_vegetation   | 57.57 | 78.83 |
|        tree        | 42.91 | 44.98 |
|        car         |  84.3 |  97.1 |
|      clutter       | 17.52 | 54.67 |
+--------------------+-------+-------+
03/08 15:44:28 - mmengine - INFO - Iter(test) [504/504]    aAcc: 75.3600  mIoU: 58.9900  mAcc: 73.9300  data_time: 0.0038  time: 0.1848
```

### Vaihingen原始的概念库结果
```python
03/08 15:23:10 - mmengine - INFO - 
+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
| impervious_surface | 79.81 | 85.61 |
|      building      | 87.98 | 90.84 |
|   low_vegetation   | 60.82 | 72.19 |
|        tree        | 72.59 | 79.56 |
|        car         | 70.03 |  91.7 |
|      clutter       |  6.57 |  65.2 |
+--------------------+-------+-------+
03/08 15:23:10 - mmengine - INFO - Iter(test) [100/100]    aAcc: 82.5800  mIoU: 62.9700  mAcc: 80.8500  data_time: 0.0055  time: 0.4703
```
### Vaihingen自训练的概念库结果
```python
03/08 15:45:39 - mmengine - INFO - 
+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
| impervious_surface | 79.13 | 84.27 |
|      building      | 86.63 | 88.68 |
|   low_vegetation   | 60.83 | 72.19 |
|        tree        | 73.67 | 81.82 |
|        car         |  67.4 | 90.81 |
|      clutter       |  6.38 | 66.88 |
+--------------------+-------+-------+
03/08 15:45:39 - mmengine - INFO - Iter(test) [100/100]    aAcc: 82.1600  mIoU: 62.3400  mAcc: 80.7800  data_time: 0.0049  time: 0.4505
```
### iSAID原始概念库的结果
```python
+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
|     background     | 97.49 | 98.15 |
|        ship        | 59.76 | 66.85 |
|     store_tank     | 12.94 |  13.1 |
|  baseball_diamond  | 11.96 | 51.86 |
|    tennis_court    | 25.69 | 65.91 |
|  basketball_court  |  7.73 | 44.95 |
| Ground_Track_Field | 27.78 | 31.91 |
|       Bridge       | 19.33 | 49.97 |
|   Large_Vehicle    | 60.81 | 79.26 |
|   Small_Vehicle    | 46.77 | 65.58 |
|     Helicopter     | 15.14 | 18.09 |
|   Swimming_pool    | 47.52 | 51.85 |
|     Roundabout     | 15.16 | 19.72 |
| Soccer_ball_field  | 28.43 | 32.82 |
|       plane        | 69.64 | 95.21 |
|       Harbor       |  20.9 | 68.89 |
+--------------------+-------+-------+
03/08 15:09:57 - mmengine - INFO - Iter(test) [2911/2911]    aAcc: 97.0400  mIoU: 35.4400  mAcc: 53.3800  data_time: 0.0073  time: 0.2258
```
### iSAID自训练的概念库结果
```python
+--------------------+-------+-------+
|       Class        |  IoU  |  Acc  |
+--------------------+-------+-------+
|     background     | 96.67 | 97.29 |
|        ship        | 52.32 | 57.02 |
|     store_tank     | 18.36 | 18.69 |
|  baseball_diamond  | 12.37 | 51.73 |
|    tennis_court    |  24.6 |  68.3 |
|  basketball_court  |  7.87 | 44.48 |
| Ground_Track_Field | 27.93 | 32.15 |
|       Bridge       |  4.66 | 67.76 |
|   Large_Vehicle    | 57.27 | 76.28 |
|   Small_Vehicle    | 46.38 |  65.8 |
|     Helicopter     | 15.07 | 17.99 |
|   Swimming_pool    | 10.91 | 58.67 |
|     Roundabout     | 22.43 | 43.51 |
| Soccer_ball_field  | 30.36 | 36.96 |
|       plane        | 71.71 | 94.43 |
|       Harbor       | 18.21 | 70.28 |
+--------------------+-------+-------+
03/08 15:57:05 - mmengine - INFO - Iter(test) [2911/2911]    aAcc: 96.2100  mIoU: 32.3200  mAcc: 56.3300  data_time: 0.0054  time: 0.2162
```

## 概念库之间的区别
```python
PS D:\CodeReading\ConceptBank> python compare_pt_files.py configs/concept_bank/cb_sam3_rs_enhanced_sg.pt configs/concept_bank/cb_sam3_rs.pt
================================================================================
COMPARING PT FILES
================================================================================
File 1: configs/concept_bank/cb_sam3_rs_enhanced_sg.pt (2.87 MB)
File 2: configs/concept_bank/cb_sam3_rs.pt (2.75 MB)

FILE TYPE COMPARISON:
  Type: dict vs dict

TOP-LEVEL KEYS:
  Common keys: ['build_args', 'build_profile', 'global_text_dims', 'global_text_layout', 'sam3_config', 'type', 'users', 'version']

USER COUNT COMPARISON:
  File 1 user count: 4
  File 2 user count: 4

CLASS COUNT PER USER:
  isaid: 16 vs 16 classes
  loveda: 7 vs 7 classes
  potsdam: 6 vs 6 classes
  vaihingen: 6 vs 6 classes

DETAILED STRUCTURE COMPARISON:
  SAME in 'build_profile': {'total_min': "<class 'float'>", 'per_user': {'loveda': '<truncated at depth 3>', 'potsdam': '<truncated at depth 3>', 'vaihingen': '<truncated at depth 3>', 'isaid': '<truncated at depth 3>'}}     
  SAME in 'global_text_dims': {'T': "<class 'int'>", 'Df': "<class 'int'>", 'De': "<class 'int'>"}
  SAME in 'global_text_layout': {'language_features': {'raw_shape': ['<truncated at depth 3>', '<truncated at depth 3>', '<truncated at depth 3>'], 'batch_dim': "<class 'int'>", 'token_dim': "<class 'int'>", 'd_dim': "<class 'int'>"}, 'language_mask': {'raw_shape': ['<truncated at depth 3>', '<truncated at depth 3>'], 'batch_dim': "<class 'int'>", 'token_dim': "<class 'int'>"}, 'language_embeds': {'raw_shape': ['<truncated at depth 3>', '<truncated at depth 3>', '<truncated at depth 3>'], 'batch_dim': "<class 'int'>", 'token_dim': "<class 'int'>", 'd_dim': "<class 'int'>"}}
  SAME in 'type': <class 'str'>
  SAME in 'version': <class 'str'>

USER-SPECIFIC COMPARISON:

  USER: isaid
    Query words count: 16 (same)
    language_features shape: torch.Size([32, 16, 256]) (same)
    language_mask shape: torch.Size([16, 32]) (same)
    language_embeds shape: torch.Size([32, 16, 1024]) (same)

  USER: loveda
    Query words count: 7 (same)
    language_features shape: torch.Size([32, 7, 256]) (same)
    language_mask shape: torch.Size([7, 32]) (same)
    language_embeds shape: torch.Size([32, 7, 1024]) (same)

  USER: potsdam
    Query words count: 6 (same)
    language_features shape: torch.Size([32, 6, 256]) (same)
    language_mask shape: torch.Size([6, 32]) (same)
    language_embeds shape: torch.Size([32, 6, 1024]) (same)

  USER: vaihingen
    Query words count: 6 (same)
    language_features shape: torch.Size([32, 6, 256]) (same)
    language_mask shape: torch.Size([6, 32]) (same)
    language_embeds shape: torch.Size([32, 6, 1024]) (same)

================================================================================
SUMMARY OF DIFFERENCES:
================================================================================
Size difference: 0.12 MB
User count difference: 0

Detailed differences would require examining the specific user data shown above.
PS D:\CodeReading\ConceptBank> 
```

# 一种可能有用的思路
它的目标是：**在不进行训练的情况下，仅在推理阶段实现所有模块**，并将你之前设计的：

* **GT/点提示先验**
* **多义文本扩展**
* **visual-text prototype bank**
* **prototype attention**
* **pixel-prototype matching**

全部整合进 **SAM3 推理 pipeline**。


---

# SAM3 推理阶段增强模块代码实现提示词

## 目标

对现有 **SAM3 语义分割推理代码**进行修改，实现一个 **完全 training-free 的推理增强框架**：

**Point-Prior Prototype Prompting for SAM3**

要求：

1. **所有模块仅在推理阶段运行**
2. **不修改 SAM3 encoder 权重**
3. **仅新增推理模块**
4. **兼容输入尺寸 1008×1008**
5. **利用扩展同义词文件构建文本语义空间**

---

# 一、需要读取的配置文件

基础类别文件：

```
D:\CodeReading\SAM3-TEXTOP\configs\cls_vaihingen.txt
```

内容：

```
road
building
grass
tree
car
clutter
```

扩展同义词文件：

```
D:\CodeReading\SAM3-TEXTOP\configs\cls_vaihingen_ex.txt
```

示例：

```
road,highway, street, avenue, lane, path, trail, driveway, expressway, boulevard
building,house, office, factory, warehouse, skyscraper, apartment, residential building, commercial structure, industrial building
grass,lawn, meadow, pasture, grassland, turf, herbaceous cover, green vegetation, open field
tree,trees, forest, woodland, grove, canopy, shrub, timber, arboreal cover
car,vehicle, automobile, sedan, passenger car, compact car, truck, bus, commercial vehicle, motorcycle
clutter,unclassified, background, void, noise, unknown region, non-feature, disorganized pixels, unclassified area
```

解析规则：

* 第一项为 **主类别**
* 后续为 **同义词集合**

构建结构：

```
{
 road: [road, highway, street ...],
 building: [...]
}
```

---

# 二、整体系统结构

在现有 SAM3 推理代码中新增模块：

```
sam3_textop/
│
├── sam3_model
│
├── prototype
│   ├── prototype_builder.py
│   ├── prototype_bank.py
│
├── text
│   ├── synonym_parser.py
│   ├── text_encoder.py
│
├── prompt
│   ├── point_sampler.py
│
├── inference
│   ├── prototype_attention.py
│   ├── pixel_proto_match.py
│
└── sam3_enhanced_segmentor.py
```

最终入口：

```
EnhancedSAM3Segmentor
```

---

# 三、模块1：同义词解析模块

文件：

```
synonym_parser.py
```

功能：

读取 `cls_vaihingen_ex.txt` 并构建同义词字典。

实现逻辑：

```
读取文件
按行解析
按逗号分割
strip空格
构建 dict
```

返回结构：

```
class_synonyms = {
 "road": ["road","highway","street"...],
 "building":[...]
}
```

---

# 四、模块2：文本 embedding 构建

文件：

```
text_encoder.py
```

使用：

```
CLIP text encoder
```

流程：

```
for class in class_synonyms:

    for synonym in synonyms:

        embedding = text_encoder(synonym)

    class_text_embedding = mean(all_synonym_embeddings)
```

得到：

```
text_prototypes
```

结构：

```
text_prototypes =

C × D
```

其中：

```
C = 类别数
D = embedding dim
```

---

# 五、模块3：点提示生成（推理阶段）

文件：

```
point_sampler.py
```

目标：

利用 **GT mask** 生成 **点提示**。

规则：

1. 对每个类别 mask
2. 进行 **mask erosion**
3. 在内部采样 **K 个点**

推荐：

```
K = 5
```

算法：

```
mask -> erode
mask_pixels -> random sampling
```

输出：

```
points =

{
 class_id :
    [(x1,y1),(x2,y2)...]
}
```

---

# 六、模块4：SAM3 区域视觉 prototype 提取

文件：

```
prototype_builder.py
```

流程：

1. 输入：

```
image
points
```

2. 调用 SAM3：

```
SAM3(image, point_prompt)
```

3. 获取：

```
pixel_embedding
mask
```

4. 计算区域特征：

```
visual_proto = mean(pixel_embedding[mask])
```

输出：

```
visual_prototypes
```

结构：

```
C × D
```

---

# 七、模块5：visual-text prototype 融合

文件：

```
prototype_bank.py
```

融合方式：

```
p_c = α * visual_proto + (1-α) * text_proto
```

推荐：

```
α = 0.7
```

输出：

```
prototype_bank

C × D
```

---

# 八、模块6：Prototype Attention

文件：

```
prototype_attention.py
```

目的：

利用 prototype 强化 pixel embedding。

输入：

```
pixel_embedding

B × C × H × W
```

处理：

```
reshape -> N × D
```

计算 attention：

```
A = softmax(pixel @ prototype^T)
```

更新 pixel：

```
pixel_refined = pixel + A @ prototype
```

恢复：

```
B × C × H × W
```

---

# 九、模块7：Pixel-Prototype Matching

文件：

```
pixel_proto_match.py
```

输入：

```
pixel_embedding
prototype_bank
```

计算：

```
score_c = cosine(pixel , prototype_c)
```

输出：

```
logits

C × H × W
```

最终：

```
seg_map = argmax(logits)
```

---

# 十、增强 SAM3 Segmentor

文件：

```
sam3_enhanced_segmentor.py
```

新增 pipeline：

```
def inference(image, gt_mask=None):

    1 parse synonyms
    2 build text prototypes

    if gt_mask exists:

        3 sample points
        4 build visual prototypes

    5 fuse prototype bank

    6 SAM3 image encoder

    7 prototype attention

    8 pixel-prototype matching

    9 output segmentation
```

---

# 十一、推理阶段完整流程

```
image
 │
 │
SAM3 encoder
 │
pixel embedding
 │
 │
prototype attention
 │
 │
pixel ↔ prototype similarity
 │
 │
argmax
 │
 │
semantic segmentation
```

---

# 十二、重要约束

实现时必须遵守：

1️⃣ **所有计算仅在推理阶段进行**

禁止：

```
backpropagation
optimizer
training loop
```

2️⃣ **SAM3 backbone 权重保持冻结**

3️⃣ **输入图像固定**

```
1008 × 1008
```

4️⃣ **prototype bank 在推理时动态构建**

5️⃣ **同义词仅来自**

```
cls_vaihingen_ex.txt
```

---

# 十三、最终目标

构建一个增强的 SAM3 推理框架：

```
Enhanced SAM3

= SAM3 encoder
+ point prompt prior
+ visual-text prototype bank
+ prototype attention
+ pixel-prototype matching
```

用于：

```
training-free
open-vocabulary
semantic segmentation
```

