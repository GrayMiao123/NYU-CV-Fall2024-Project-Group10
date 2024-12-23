from ultralytics import YOLO

model = YOLO('/mnt/workspace/NYU-CV-Fall2024-Project/yolov10/runs/prune/n-yolov8-nop234-gam-finetune2/weights/best.pt')

# Validate the model
results = model.val(
    data="/mnt/workspace/yolov10/ultralytics/cfg/datasets/TT100K.yaml",
    imgsz=640,
    project='runs/prune/validation',
    name='test1'
)


metrics = results.results_dict


print(f"\nValidation Results:")
print(f"mAP50: {metrics['metrics/mAP50(B)']:.5f}")
print(f"mAP50-95: {metrics['metrics/mAP50-95(B)']:.5f}")
print(f"Precision: {metrics['metrics/precision(B)']:.5f}")
print(f"Recall: {metrics['metrics/recall(B)']:.5f}")
