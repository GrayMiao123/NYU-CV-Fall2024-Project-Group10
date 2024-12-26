import os

import numpy as np
import torch
from tabulate import tabulate

from ultralytics import YOLO
import warnings

warnings.filterwarnings('always')

def read_label_file(label_path):
    """Read ground truth labels from txt file"""
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([int(class_id), x_center, y_center, width, height])
    return boxes

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes in xywh format"""
    # Convert to x1y1x2y2 format
    x1_1 = box1[1] - box1[3]/2
    y1_1 = box1[2] - box1[4]/2
    x2_1 = box1[1] + box1[3]/2
    y2_1 = box1[2] + box1[4]/2
    
    x1_2 = box2[1] - box2[3]/2
    y1_2 = box2[2] - box2[4]/2
    x2_2 = box2[1] + box2[3]/2
    y2_2 = box2[2] + box2[4]/2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = box1[3] * box1[4]
    area2 = box2[3] * box2[4]
    union = area1 + area2 - intersection
    
    return intersection / union

# Determine device
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(f"\nUsing device: {device}")

# Load model and data
model = YOLO("./best.pt",task="detect").to(device)
img_path = os.path.join("./tt100k_example/images/val", "2.jpg")
label_path = os.path.join("./tt100k_example/labels/val", "2.txt")

# Get ground truth
gt_boxes = read_label_file(label_path)
gt_table = [[i+1, box[0], f"({box[1]:.3f}, {box[2]:.3f})", f"({box[3]:.3f}, {box[4]:.3f})"] 
            for i, box in enumerate(gt_boxes)]
print("\n=== Ground Truth Boxes ===")
print(tabulate(gt_table, 
              headers=['ID', 'Class', 'Center (x,y)', 'Size (w,h)'],
              tablefmt='grid'))

# Get predictions
results = model(img_path)
pred_boxes = []
pred_table = []
for r in results:
    boxes = r.boxes
    # Get image size from results
    img_width = r.orig_shape[1]
    img_height = r.orig_shape[0]
    
    for i, box in enumerate(boxes):
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()
        
        # Normalize coordinates to 0-1 range
        xcenter = ((xyxy[0] + xyxy[2]) / 2) / img_width
        ycenter = ((xyxy[1] + xyxy[3]) / 2) / img_height
        width = (xyxy[2] - xyxy[0]) / img_width
        height = (xyxy[3] - xyxy[1]) / img_height
        
        pred_box = [cls, xcenter, ycenter, width, height, conf]
        pred_boxes.append(pred_box)
        pred_table.append([i+1, cls, f"{conf:.3f}", f"({xcenter:.3f}, {ycenter:.3f})", f"({width:.3f}, {height:.3f})"])

print("\n=== Predicted Boxes ===")
print(tabulate(pred_table,
              headers=['ID', 'Class', 'Conf', 'Center (x,y)', 'Size (w,h)'],
              tablefmt='grid'))

# Calculate IoU matches
iou_threshold = 0.5
iou_results = []
for i, pred_box in enumerate(pred_boxes):
    max_iou = 0
    matched_gt = None
    
    for gt_box in gt_boxes:
        iou = calculate_iou(gt_box, pred_box)
        if iou > max_iou:
            max_iou = iou
            matched_gt = gt_box
    
    status = 'MATCHED' if max_iou >= iou_threshold else 'NOT MATCHED'
    gt_class = matched_gt[0] if matched_gt else 'N/A'
    iou_results.append([
        i+1, 
        pred_box[0],
        f"{pred_box[5]:.3f}",
        gt_class,
        f"{max_iou:.3f}",
        status
    ])

print("\n=== IoU Analysis ===")
print(tabulate(iou_results,
              headers=['Pred ID', 'Pred Class', 'Confidence', 'GT Class', 'IoU', 'Status'],
              tablefmt='grid'))

# Calculate metrics
true_positives = sum(1 for pred in pred_boxes if max(calculate_iou(gt, pred) for gt in gt_boxes) >= iou_threshold)
false_positives = len(pred_boxes) - true_positives
false_negatives = sum(1 for gt in gt_boxes if max(calculate_iou(gt, pred) for pred in pred_boxes) < iou_threshold)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0

metrics_table = [
    ['IoU Threshold', iou_threshold],
    ['True Positives', true_positives],
    ['False Positives', false_positives],
    ['False Negatives', false_negatives],
    ['Precision', f"{precision:.3f}"],
    ['Recall', f"{recall:.3f}"]
]

print("\n=== Summary Metrics ===")
print(tabulate(metrics_table, tablefmt='grid'))

print(tabulate(metrics_table, tablefmt='grid'))
