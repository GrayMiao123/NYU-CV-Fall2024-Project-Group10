import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('/workspace/yolov8-tricks')
from ultralytics import YOLO
import torch
import gc
import math

def calculate_batch_size(memory_fraction=0.9, model_size='s', num_gpus=4):
    """
    Calculate optimal batch size for multiple GPUs
    """
    if not torch.cuda.is_available():
        return 16
    
    # 获取单个GPU显存（以GB为单位）
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    total_memory = gpu_memory * num_gpus  # 总显存
    
    # A100特定优化的每GPU基础batch size
    base_batch_sizes = {
        'n': 192,  # nano
        's': 128,   # small - 建议的优化配置
    }
    
    # 获取基础batch size
    base_batch = base_batch_sizes.get(model_size.lower(), 64)
    
    # 根据显存比例调整
    adjusted_batch = int(base_batch * memory_fraction)
    
    # 确保是8的倍数
    adjusted_batch = (adjusted_batch // 8) * 8
    
    # 限制每个GPU的最大batch size
    per_gpu_batch = min(adjusted_batch, 128)
    
    return per_gpu_batch

if __name__ == '__main__':
    # 固定配置
    import os
    os.environ['PYTHONPATH'] = '/workspace/yolov8-tricks:' + os.environ.get('PYTHONPATH', '')
    NUM_GPUS = 4
    MEMORY_FRACTION = 0.9
    DEVICE = '0,1,2,3'
    
    # 定义模型配置列表 cfg,batchsize
    MODEL_CONFIGS = {'/'

    # 遍历每个模型配置进行训练
    for model_cfg, batch_size in MODEL_CONFIGS.items():
        print(f"\nTraining model with config: {model_cfg}")
        
        # 初始化模型
        model = YOLO(model_cfg)
        model_name = model_cfg.split('/')[-1].split('.')[0]
        
        try:
            # 训练模型
            model.train(
                model=model_cfg,
                data='/workspace/mydata/YOLOv8_TT100K.yaml',
                epochs=200,
                batch=batch_size,
                imgsz=640,
                save=True,
                save_period=50,
                cache=True,
                device=[0,1,2,3],
                workers=16,
                project=f'runs/{model_name}',
                name='train_optimized',
                exist_ok=False,
                pretrained="/workspace/yolov8-tricks/yolov8s.pt",
                optimizer='Adam',
                seed=3407,
                deterministic=False,
                single_cls=False,
                rect=False,
                cos_lr=True,
                close_mosaic=10,
                amp=True,
                fraction=1.0,
                freeze=None,
                lr0=0.001 * (batch_size/ 64),
                lrf=0.01,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=5.0,
                warmup_momentum=0.8,
                warmup_bias_lr=0.1,
                box=7.5,
                cls=0.5,
                dfl=1.5,
                plots=True,
                
                # Data augmentation settings
                hsv_h=0.015,
                hsv_s=0.7,
                hsv_v=0.4,
                degrees=15,
                translate=0.1,
                scale=0.5,
                shear=5.0,
                perspective=0.0005,
                flipud=0.0,
                fliplr=0.0,
                mosaic=1.0,
                mixup=0.1,
                copy_paste=0.3,
                
            )

        except Exception as e:
            print(f"Error training model {model_cfg}: {str(e)}")
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
