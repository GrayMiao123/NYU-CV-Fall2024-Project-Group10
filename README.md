# Efficient Small Object Detection YOLO Model for Edge Computing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 311+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/release/python-3110/)

>NYU Courant Computer Vision Project - Fall 2024

This repository contains the implementation of an efficient YOLO-based model for small object detection, specifically optimized for edge computing applications. The model includes improvements through architecture modifications and model compression techniques including pruning and quantization.

## 👥 Authors
Jinze Huang - jh9108@nyu.edu - @RogerHuangPKX

Ning Miao - nm4543@nyu.edu - @GrayMiao123

CIMS, New York University

## 📚 Overview
This work presents an efficient adaptation of YOLOv8 for small object detection, particularly focused on traffic sign detection. Our main contributions include:
- Modified YOLOv8 architecture optimized for small object detection
- Model compression pipeline including pruning and quantization
- Evaluation on the TT100K traffic sign dataset

Video demonstration available on [YouTube](https://www.youtube.com/watch?v=ox1loJ9JUdI)

## How to use this repository

### 1. Clone the repository
```bash
git clone https://github.com/RogerHuangPKX/NYU-CV-Fall2024-Project.git
cd NYU-CV-Fall2024-Project
```

### 2. Set up the environment   
```bash
pip install -U openmim
mim install mmcv
pip install -r Modified-YOLOv8/requirements.txt
```

### 3. Run the model
#### 3.1 Pruning
```bash
python compress.py
```
#### 3.2 Quantization
```bash
python quantize.py
```
#### 3.3 Validation
```bash
python val.py
```

## 📊 Results
Our model achieved significant improvements in detection performance compared to baseline models. Here are the key metrics:

### Best Model Performance (ours)
| Metric       | Value  |
| ------------ | ------ |
| Precision    | 91.54% |
| Recall       | 81.12% |
| mAP@0.5      | 91.23% |
| mAP@0.5:0.95 | 72.59% |

### Ablation Studies

#### Attention Mechanism Comparison
| Model              | Precision  | Recall     | mAP@0.5    | mAP@0.5:0.95 |
| ------------------ | ---------- | ---------- | ---------- | ------------ |
| YOLOv8s (Baseline) | 65.11%     | 50.09%     | 57.77%     | 46.82%       |
| w/ CBAM            | 89.77%     | 75.84%     | 87.61%     | 66.58%       |
| w/ CA              | 90.31%     | 76.12%     | 87.98%     | 69.29%       |
| w/ SA              | 89.08%     | 76.62%     | 87.62%     | 67.96%       |
| w/ GAM (Ours)      | **91.54%** | **81.12%** | **91.23%** | **72.59%**   |

#### Backbone Architecture Variants
| Model             | Precision  | Recall     | mAP@0.5    | mAP@0.5:0.95 |
| ----------------- | ---------- | ---------- | ---------- | ------------ |
| EfficientViT      | 73.64%     | 59.13%     | 68.90%     | 55.25%       |
| MobileNetv4       | 81.70%     | 59.04%     | 71.47%     | 51.87%       |
| FasterNet         | 81.18%     | 59.06%     | 71.63%     | 47.71%       |
| GhostNetv2        | 83.85%     | 64.89%     | 77.16%     | 63.16%       |
| C2f-Ghost         | 89.49%     | 76.43%     | 87.81%     | 69.34%       |
| Ours (GAM_P2/3/4) | **91.54%** | **81.12%** | **91.23%** | **72.59%**   |

### Comparison with State-of-the-Art
| Model              | Precision  | Recall     | mAP@0.5    | mAP@0.5:0.95 |
| ------------------ | ---------- | ---------- | ---------- | ------------ |
| YOLOv8s (Baseline) | 65.11%     | 50.09%     | 57.77%     | 46.82%       |
| YOLOv9             | 62.84%     | 50.42%     | 57.61%     | 46.67%       |
| YOLOv10s           | 58.35%     | 48.22%     | 53.56%     | 43.60%       |
| Ours (GAM_P2/3/4)  | **91.54%** | **81.12%** | **91.23%** | **72.59%**   |

Key findings from our ablation studies:
1. GAM attention mechanism provides the best performance among all attention variants, with significant improvements across all metrics
2. Our backbone architecture outperforms other modern architectures like EfficientViT and GhostNetv2
3. The model achieves substantial improvements over the baseline:
   - +26.43% increase in precision
   - +31.03% increase in recall 
   - +33.46% improvement in mAP@0.5
   - +25.77% gain in mAP@0.5:0.95

These results demonstrate the effectiveness of our architectural modifications and optimization techniques for small object detection.


## 📂 Repository Structure
```
.
├── Modified_YOLOv8
│   ├── best.pt
│   ├── dataset
│   ├── draw_img_tools
│   ├── model_metrics.csv
│   ├── pred.py
│   ├── requirements.txt
│   ├── train.py
│   ├── trainlog_and_weights
│   ├── tt100k_example
│   └── ultralytics
├── README.md
├── model_compression
│   ├── CONTRIBUTING.md
│   ├── LICENSE
│   ├── README.md
│   ├── SAHI-detect.py
│   ├── SAHI.py
│   ├── compress.py
│   ├── configs
│   ├── docker
│   ├── docs
│   ├── examples
│   ├── figures
│   ├── logs
│   ├── mkdocs.yml
│   ├── models
│   ├── plot_channel_image.py
│   ├── pyproject.toml
│   ├── quantize.py
│   ├── requirements.txt
│   ├── runs
│   ├── train.py
│   ├── transform_weight.py
│   ├── ultralytics
│   ├── val.py
│   ├── yolov8n.pt
│   └── yolov8s_GAM_smalltarget_noP5
└── paper
    ├── NYU-CV-Fall2024-Project-Group10.pdf
    ├── arxiv.sty
    ├── images
    ├── model_metrics.csv
    ├── orcid.pdf
    ├── references.bib
    └── template.tex

20 directories, 27 files
```

## 🔧 Environment Setup
- Python: 3.11+
- PyTorch: 2.1+
- Torchvision: 0.17.2+cu121
- timm: 1.0.7
- Openmim (pip install -U openmim)
- mmcv: 2.2.0 (`mim install "mmcv>=2.0.0"`)
- torch-pruning: 1.4.1
- TensorRT: 8.6.1.6 ([Download Link](https://developer.nvidia.com/tensorrt))

## 🗃️ Dataset
### Tsinghua-Tencent-100K (TT100K)
- Official dataset: [TT100K Homepage](https://cg.cs.tsinghua.edu.cn/traffic-sign/)
- Recommended: [Kaggle version](https://www.kaggle.com/datasets/braunge/tt100k?select=mydata)

### Dataset Structure
```
📂 TT100K
├── 📂 image
│   ├── 📂 train (20.6K images)
│   └── 📂 val (3,627 images)
├── 📂 labels
│   ├── 📂 train
│   └── 📂 val
└── 📄 config files
    ├── YOLOv5.yaml
    └── YOLOv8.yaml
```

## 🚀 Model Pipeline

### 1. Base Model
Our modified YOLOv8 model incorporates:
- GAM attention mechanism
- Optimizations for small target detection
- Detection head modified to be more suitable for small object detection
- Modified backbone structure

### 2. Model Compression
We implement a two-stage compression pipeline:

#### Stage 1: Pruning
```bash
python compress.py
```
Key parameters in `compress.py`:
```python
param_dict = {
    'model': '/path/to/weights/best.pt',
    'data': '/path/to/TT100K.yaml',
    'prune_method': 'lamp',
    'global_pruning': True,
    'speed_up': 2.0
}
```

#### Stage 2: Quantization
```bash
python quantize.py
```

### 3. Model Validation
To evaluate model performance:
```bash
python val.py
```




## 📝 Thanks
We would like to thank the following projects for their contributions to this project:
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [YOLOv8 & YOLOv11](https://github.com/ultralytics/ultralytics)
- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [Training Framework](https://github.com/chaizwj/yolov8-tricks)
- [MobileOne](https://github.com/apple/ml-mobileone)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
- [RepViT-SAM](https://github.com/THU-MIG/RepViT)
- [GHOST Face Swap](https://github.com/ai-forever/ghost)
- [Huawei GhostNet](https://github.com/huawei-noah/Efficient-AI-Backbones)
- [MobileViT](https://github.com/yangyucheng000/MobileViT)
- [SAHI](https://github.com/obss/sahi)
- [External Attention](https://github.com/xmu-xiaoma666/External-Attention-pytorch)
- [YOLOv8-tricks](https://github.com/chaizwj/yolov8-tricks)
- [YOLO-Air](https://github.com/iscyy/yoloair)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments
The dataset is provided by Tsinghua University and Tencent. We do not own the dataset. We declare that we have not used any other dataset for this project, or any other commercial dataset. We do not use this dataset for any commercial purposes. 
