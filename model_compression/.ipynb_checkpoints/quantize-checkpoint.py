from ultralytics import YOLO
import time
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                      default='/mnt/workspace/NYU-CV-Fall2024-Project/yolov10/runs/prune/n-yolov8-nop234-gam-finetune2/weights/best.pt',
                      help='原始模型路径')
    parser.add_argument('--data', type=str,
                      default="/mnt/workspace/yolov10/ultralytics/cfg/datasets/TT100K.yaml",
                      help='数据集配置文件路径')
    parser.add_argument('--batch', type=int, default=8, help='batch size')
    parser.add_argument('--workspace', type=int, default=8, help='TensorRT workspace size in GB')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载原始模型
    print(f"Loading model from {args.model}")
    model = YOLO(args.model)
    
    # 评估原始模型
    print("\nEvaluating original model...")
    original_metrics = model.val(data=args.data)
    
    # 导出TensorRT INT8模型
    print("\nExporting to TensorRT INT8...")
    model.export(
        format="engine",
        # dynamic=True,
        batch=args.batch,
        workspace=args.workspace,
        int8=True,
        data=args.data
    )
    
    # 加载TensorRT模型
    print("\nLoading TensorRT model...")
    engine_path = args.model.replace('.pt', '.engine')
    trt_model = YOLO(engine_path)
    
    # 评估TensorRT模型
    print("\nEvaluating TensorRT model...")
    trt_metrics = trt_model.val(data=args.data)
    
    # 性能对比
    print("\nPerformance Comparison:")
    print("-" * 50)
    print("Original Model:")
    print(f"mAP50-95: {original_metrics.box.map}")
    print(f"mAP50: {original_metrics.box.map50}")
    print(f"Precision: {original_metrics.box.mp}")
    print(f"Recall: {original_metrics.box.mr}")
    print(f"Speed: {original_metrics.speed['inference']:.1f}ms inference per image")
    
    print("\nTensorRT INT8 Model:")
    print(f"mAP50-95: {trt_metrics.box.map}")
    print(f"mAP50: {trt_metrics.box.map50}")
    print(f"Precision: {trt_metrics.box.mp}")
    print(f"Recall: {trt_metrics.box.mr}")
    print(f"Speed: {trt_metrics.speed['inference']:.1f}ms inference per image")
    
    # 计算加速比
    speedup = original_metrics.speed['inference'] / trt_metrics.speed['inference']
    print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()