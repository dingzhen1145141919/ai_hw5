# 10235501451 唐屹 人工智能作业5

## 1.项目结构

```
├── models/
│   ├── blip-image-captioning-base/          # BLIP图像描述预训练模型
│   ├── blip-itm-base-coco/                  # BLIP图文匹配预训练模型
│   ├── blip_fusion/
│   │   ├── output_blip_1/                   # blip模型1输出目录
│   │   ├── output_blip_2/                   # blip模型2输出目录
│   │   ├── output_blip_3/                   # blip模型3输出目录
│   │   ├── blip_1.py                        # BLIP实验1脚本
│   │   ├── blip_2.py                        # BLIP实验2脚本
│   │   └── blip_3.py                        # BLIP实验3脚本
│   ├── clip-vit-base-patch32/               # CLIP预训练模型
│   ├── clip_fusion/
│   │   ├── output_clip_1/                   # CLIP实验输出目录
│   │   └── clip_1.py                        # CLIP验脚本
│   ├── early_fusion/
│   │   ├── outputs_early_fusion_optimized/  # early_fusion输出目录
│   │   └── early_fusion.py                  # 早期融合模型脚本
│   ├── late_fusion/
│   │   ├── outputs/                         # 原始实验输出目录
│   │   ├── outputs_english_only/            # 仅英语样本实验输出目录
│   │   ├── outputs_optimized/               # 优化版实验输出目录
│   │   ├── late_fusion_eng.py               # 仅英语样本的后期融合脚本
│   │   ├── late_fusion_optimized.py         # 优化版后期融合脚本
│   │   └── late_fusion_original.py          # 原始后期融合脚本
│   ├── roberta-base/                        # RoBERTa预训练模型
│   └── vit-base/                            # ViT预训练模型
├── data/                                    # 原始图文数据目录
│   ├── {guid}.jpg                           # 图片文件（按guid命名）
│   ├── {guid}.txt                           # 原始文本文件（按guid命名）
│   └── {guid}_pro.txt                       # 生成的增强文本文件
├── utils/                                   # 工具脚本目录
│   ├── generate_caption_files.py            # 批量生成图片描述&_pro.txt文件
│   ├── count_language.py                    # 文本语言检测&分布统计
│   ├── predcict.py                          # 最终填补标签
│   └── filter_train.py                      # 筛选英语样本训练集
├── train.txt                                # 原始训练集标签文件（含guid、tag）
├── train_english_only.txt                   # 筛选后仅含英语样本的训练集
├── requirements.txt                         # Python依赖包列表
└── README.md                                # 项目说明文档
```

## 2.代码执行流程

## **_注:对于models中调用的与训练模型，由于github仓库大小限制没有上传，请到hugging face镜像网下载对应模型到文件夹后运行或者将代码中对于与训练模型的调用修改成联网使用_**

我一共设计了三个实验，执行流程如下：

### 2.1 探究去除非英语文本数据对于late_fusion的影响

我们可以先在utils下先运行 `count_language.py` 得到数据集中的数据语言分布，然后运行 `filter_train.py`，可以得到去除非英语文本数据的训练集 `train_english_only.txt` (在项目根目录中，项目中已经包含)，接下来我们可以进入 `models/late_fusion`，按顺序运行 `late_fusion_original`、`late_fuison_eng`、`late_fusion_optimized`，分别比较late fusion融合方式在采用原始数据、排除非英语数据下和优化情况下的性能差异。

### 2.2 比较优化后的late_fusion 与 early_fusion的性能差异

运行models/early_fusion/early_fusion.py ，与上一个实验的最后一个优化过的模型进行比较分析

### 2.3 比较blip与clip的性能差异

在尽可能公平的情况下，我们对于blip与clip的性能进行了比较，可以在 `/models/blip_fuison `中运行 `blip_1.py`，在 `/models/clip_fusion `中运行 `clip_1.py` 可以根据运行结果进行比较。

### 2.4 blip与earlyfusion的融合与数据增强

针对blip与early_fusion，我们将二者进行了融合，可以运行 `models/blip_fuison/blip_2.py`查看性能；同时，我们对数据进行了进一步增强，我们使用与训练的模型对于每一张图片的内容进行描述，生成了对应的增强文本，运行 `utils/generate_caption_files.py `后 可以在 `/data`中得到对应的文本增强数据，如1_pro.txt,接着我们进一步的使用增强的文本数据进行训练，请运行 `models/blip_fuison/blip_3.py`

### 2.5 预测测试集情感标签

运行 `/utils/predict.py `可以得到 测试集预测结果 `test_with_pred_lable.txt`

## 3.代码运行环境&所需库

### 3.1 系统环境

| 项⽬        | 详情                                                                                 |
| ----------- | ------------------------------------------------------------------------------------ |
| 操作系统    | Ubuntu 20.04.6 LTS (Focal Fossa)                                                     |
| 内核版本    | Linux huashida 5.4.0-212-generic #232-Ubuntu SMP Sat Mar 15 15:34:35 UTC 2025 x86_64 |
| Python 版本 | 3.8.10                                                                               |

### 3.2 硬件配置

| 项⽬            | 详情                       |
| --------------- | -------------------------- |
| 显卡型号        | 2× NVIDIA GeForce RTX 3090 |
| NVIDIA 驱动版本 | 550.90.07                  |
| 兼容 CUDA 版本  | 12.4                       |

执行下列命令以快速安装所需库

```bash
pip install -r requirements.txt
```

## 4.参考来源

### 4.1 核心模型：

- BLIP: https://github.com/salesforce/BLIP
- CLIP: https://github.com/openai/CLIP
- Hugging Face Transformers: https://github.com/huggingface/transformers

### 4.2 功能实现参考：

- 文本分类训练框架: https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification
- Matplotlib 可视化: https://matplotlib.org/stable/gallery/index.html
- PyTorch 混合精度训练: https://pytorch.org/docs/stable/amp.html
