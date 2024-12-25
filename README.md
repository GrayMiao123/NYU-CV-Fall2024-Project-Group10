

# Inference and model compression
## 🗃️ Dataset
### Tsinghua-Tencent-100K (TT100K)
Official dataset: [TT100K Homepage](https://cg.cs.tsinghua.edu.cn/traffic-sign/)

Due to the large size of the original dataset, we recommend the [Kaggle version](https://www.kaggle.com/datasets/braunge/tt100k?select=mydata).

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


## 🔍Weights file path
```
model_compression/
│
└── yolov8s_GAM_smalltarget_noP5/
│       └── weights/
│           └── best.pt
│
└── runs/
│    └── prune/
│        ├── n-yolov8-nop234-gam-prune/
│        │   └── weights/
│        │       └── prune.pt
│        │
│        └── n-yolov8-nop234-gam-finetune2/
│            └── weights/
│                └── best.pt
│
├── val.py
|
├── compress.py
|
└── quantize.py

```

The path of our original modified model is model_compression/model_compression/yolov8s_GAM_smalltarget_noP5/weights/best.pt 

The path of our prune model is /model_compression/runs/prune/n-yolov8-nop234-gam-prune/weights/prune.pt 

The path of fine-tune model after pruning is /model_compression/runs/prune/n-yolov8-nop234-gam-finetune2/weights/best.pt

## 📉Validation
To validate each model, you need to modify val.py by adding the weight file path to class YOLO
```bash
model = YOLO('/mnt/workspace/model_compression/runs/prune/n-yolov8-nop234-gam-finetune2/weights/best.pt') #absolute path
```
and add dataset yaml file
```bash
 data="/mnt/workspace/model_compression/ultralytics/cfg/datasets/TT100K.yaml", #absolute path
```
Then run
```bash
python val.py
```

## 🚀Prune and fine-tune
To prune and fine-tune for each model, you need to modify compress.py for the first two parameters of param_dict
```bash
 param_dict = {
        # origin
        'model': '/mnt/workspace/model_compression/yolov8s_GAM_smalltarget_noP5/weights/best.pt',
        'data':'/mnt/workspace/model_compression/ultralytics/cfg/datasets/TT100K.yaml',
        'imgsz': 640,
        'epochs': 100,
        'batch': 16,
        'workers': 8,
        'cache': False,
        'optimizer': 'SGD',
        'device': '0',
        'close_mosaic': 0,
        'project':'runs/prune',
        'name':'n-yolov8-nop234-gam',
        
        # prune
        'prune_method':'lamp',
        'global_pruning': True,
        'speed_up': 2.0,
        'reg': 0.0005,
        'sl_epochs': 500,
        'sl_hyp': 'ultralytics/cfg/hyp.scratch.sl.yaml',
        'amp': False, 
        'sl_model': None,
    }, #absolute path
```
Then run
```bash
python compress.py
```

## 🎯int8 Quantization
We do quantization after pruning. For the quantization, my tensorrt version is 8.6.16 and my cuda version is 12.1. To quantization, you need to add the weight file path after fine-tuning and data yaml file path in quantize.py
```bash
 def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                      default='/mnt/workspace/model_compression/runs/prune/n-yolov8-nop234-gam-finetune2/weights/best.pt',
                      help='original path')
    parser.add_argument('--data', type=str,
                      default="/mnt/workspace/model_compression/ultralytics/cfg/datasets/TT100K.yaml",
                      help='dataset path')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--workspace', type=int, default=8, help='TensorRT workspace size in GB')
    return parser.parse_args()

```
Then run
```bash
 python quantize.py

```






















