# 分析Concept Bank的代码
## 运行Concept Bank得到的新的概念库文件
四个遥感数据均是按照https://github.com/likyoo/SegEarth-OV/blob/main/dataset_prepare.md所给的数据处理文件和教程完成的处理

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
```python
=============== Evaluation Summary ===============
Dataset           aAcc         mIoU         mAcc       
---------------- -------      -------      -------     
loveda           62.7000      47.9100      64.3900     
potsdam          74.1300      57.5600      73.7200     
vaihingen        82.7000      63.6400      80.4700     
isaid            95.5300      6.6100       53.5800     
--------------------------------------------------
MEAN             78.7650      43.9300      68.0400     
```
不难看出自行训练得到的概念库与原始的概念库在除了vaihingen数据集之外性能之间差异明显

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
