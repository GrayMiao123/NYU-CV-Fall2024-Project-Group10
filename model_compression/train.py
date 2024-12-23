import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/yolov8s_GAM_smalltarget_noP5.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(
        data='/mnt/workspace/yolov10/ultralytics/cfg/datasets/TT100K.yaml',
               batch=16,
               imgsz=640,
               epochs=200,
               patience=20,  
               augment=True,
               project='runs/v8s-s-nop5',
               name='train1',
               amp=False,
               seed=3407,
               )
    
    
    
    
