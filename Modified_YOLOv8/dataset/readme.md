# Readme
> By Jinze Huang
> Last Modified Time = {time.time:1111-1332}

[toc]


## Dataset
Tsinghua-Tencent-100K traffic sign 
> You can find it here. https://cg.cs.tsinghua.edu.cn/traffic-sign/

文件太大，数据太多。 训练finetune起来太麻烦，本proj选用的是Kaggle上面的dataset https://www.kaggle.com/datasets/braunge/tt100k?select=mydata

### Quick Review of TT100K on Kaggle
/image/train 20.6K
/image/val 3627
/labels/train
/labels/val
/YOLOv5.yaml
/YOLOv8.yaml

## Model we choose

YOLOv10

### Things to improve

1. add `attention` to structure
2. modify `data augmentation`
3. improved `loss` for `small object`
4. `large conception field` vs small 
5. `Backbone` replacement


## Ongoing things
### Benchmark
For One Stage
- [ ] yolov8
- [ ] yolov9
- [ ] yolov10
- [ ] yolov11 ?

For Two Stage
- [ ] Faster R-CNN
- [ ] R-CNn

### Backbone we can try
- [ ] GhostNet
- [ ] MobileNet
- [ ] MobileOne
- [ ] ShuffleNet
- [ ] other lightweighted models


## Paper you can read
- [ ] MobileNet v1-v3
- [ ] ShuffleNet
- [ ] MobileOne
- [ ] YOLOv10
- [ ] other small obj detection papers
