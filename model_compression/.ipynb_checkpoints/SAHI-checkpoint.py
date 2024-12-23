import os
import cv2
import logging
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.models.yolov8 import Yolov8DetectionModel
from tabulate import tabulate
from podm.metrics import BoundingBox, get_pascal_voc_metrics, MetricPerClass
import argparse
from tqdm import tqdm
import sys

# 设置日志
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(description="SAHI-based Detection Evaluation Script")
    parser.add_argument('--filepath', type=str, 
                        default='/mnt/workspace/yolov10/datasets/tt100k_2021/images/val',
                        help='Path to the images folder')
    parser.add_argument('--annotation_folder', type=str, 
                        default='/mnt/workspace/yolov10/datasets/tt100k_2021/labels/val',
                        help='Path to the annotation folder')
    parser.add_argument('--model_path', type=str, 
                        default='/mnt/workspace/NYU-CV-Fall2024-Project/yolov10/runs/prune/n-yolov8-nop234-gam-finetune2/weights/best.pt',
                        help='Path to the model weights')
    parser.add_argument('--confidence_threshold', type=float, default=0.4)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--slice_height', type=int, default=256)
    parser.add_argument('--slice_width', type=int, default=256)
    parser.add_argument('--overlap_height_ratio', type=float, default=0.2)
    parser.add_argument('--overlap_width_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    
    # 验证路径
    if not os.path.exists(args.filepath):
        logger.error(f"Image folder not found: {args.filepath}")
        sys.exit(1)
    if not os.path.exists(args.annotation_folder):
        logger.error(f"Annotation folder not found: {args.annotation_folder}")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}")
        sys.exit(1)
        
    return args

def load_model(args):
    """加载模型并进行错误处理"""
    try:
        logger.info(f"Loading model from {args.model_path}...")
        detection_model = Yolov8DetectionModel(
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
            device=args.device
        )
        logger.info("Model loaded successfully")
        return detection_model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        sys.exit(1)

def get_image_list(filepath):
    """获取图像列表并验证"""
    logger.info(f"Scanning for images in {filepath}")
    image_files = [f for f in os.listdir(filepath) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        logger.error(f"No valid images found in {filepath}")
        sys.exit(1)
    logger.info(f"Found {len(image_files)} images")
    return image_files

def process_images(model, args, image_files):
    """处理图像并收集结果"""
    logger.info("Starting image processing...")
    labels, detections = [], []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            image_path = os.path.join(args.filepath, image_file)
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not read image: {image_file}")
                continue
                
            img_h, img_w = img.shape[:2]
            
            # SAHI预测
            result = get_sliced_prediction(
                image_path,
                model,
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap_height_ratio,
                overlap_width_ratio=args.overlap_width_ratio,
                verbose=0
            )
            
            # 处理标注和检测结果...
            # (保持原有的处理逻辑)
            
        except Exception as e:
            logger.error(f"Error processing {image_file}: {str(e)}")
            continue
            
    return labels, detections

def main():
    try:
        # 解析参数
        args = parse_args()
        
        # 加载模型
        detection_model = load_model(args)
        
        # 获取图像列表
        image_files = get_image_list(args.filepath)
        
        # 处理图像
        labels, detections = process_images(detection_model, args, image_files)
        
        # 计算指标
        logger.info("Calculating metrics...")
        results = get_pascal_voc_metrics(labels, detections, 0.5)
        map_score = MetricPerClass.mAP(results)
        logger.info(f"mAP: {map_score:.4f}")
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()