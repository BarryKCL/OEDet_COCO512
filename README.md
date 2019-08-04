# OEDet_COCO512
Single-Shot Detector with Objectness Enhancement

[image](https://github.com/BarryKCL/OEDet_COCO512/blob/master/figures/OEDet_A.png)

## Highlights

  We propose an objectness estimation module (OEM) that uses the objectness maps to improve the performance of traditional one-stage detectors.
  
  We use the Multi-scale Fusion module (MFM) for low-level features to enhance feature representation and directly predict objectness maps from the enhanced feature, thereby improving model accuracy.

  We have also proposed an efficiency model called OEDet_Lite to improve detection performance while maintaining the advantages of fast detection of the one-stage detector. Without slowing down the detection speed of the original SSD, the OEDet_Lite can increase the original SSD by 1.2 mAP on the voc2007test. 

  We evaluated the model on the challenging PASCAL VOC 2007, PASCAL VOC 2012 and MS COCO benchmarks and achieved an impressive result.

### Requirements

1. Python3
1. PyTorch 1.0 or higher
1. yacs
1. [Vizer](https://github.com/lufficc/Vizer)
1. GCC >= 4.9
1. OpenCV

### Step-by-step installation

# Optional packages

# If you want visualize loss curve. Default is enabled. Disable by using --use_tensorboard 0 when training.
pip install tensorboardX

# If you train coco dataset, must install cocoapi.
cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

### Build
If your torchvision >= 0.3.0, nms build is not needed! We also provide a python-like nms, but is very slower than build-version.
```bash
# For faster inference you need to build nms, this is needed when evaluating. Only training doesn't need this.
cd ext
python build.py build_ext develop
# Build for LightNetPlusPlus.
cd modules
sh ./make.sh
```

## Train

### Setting Up Datasets
#### Pascal VOC

For Pascal VOC dataset, make the folder structure like this:
```
VOC_ROOT
|__ VOC2007
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ VOC2012
    |_ JPEGImages
    |_ Annotations
    |_ ImageSets
    |_ SegmentationClass
|__ ...
```
Where `VOC_ROOT` default is `datasets` folder in current project, you can create symlinks to `datasets` or `export VOC_ROOT="/path/to/voc_root"`.

#### COCO

For COCO dataset, make the folder structure like this:
```
COCO_ROOT
|__ annotations
    |_ instances_valminusminival2014.json
    |_ instances_minival2014.json
    |_ instances_train2014.json
    |_ instances_val2014.json
    |_ ...
|__ train2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ val2014
    |_ <im-1-name>.jpg
    |_ ...
    |_ <im-N-name>.jpg
|__ ...
```
Where `COCO_ROOT` default is `datasets` folder in current project, you can create symlinks to `datasets` or `export COCO_ROOT="/path/to/coco_root"`.

### Single GPU training

```bash
# for example, train OEDet512:
python train.py --config-file configs/vgg_ssd512_coco_trainval35k.yaml
```

## Evaluate

### Single GPU evaluating

```bash
# for example, evaluate OEDet512:
python test.py --config-file configs/vgg_ssd512_coco_trainval35k.yaml
```
### Paper:

|         | VOC2007 test | coco test-dev2017 |
| :-----: | :----------: |   :----------:    |
| OEDet300 |     79.74     |      29.4         |
| OEDet512 |     81.70     |      32.8         |

# References:
A list of SSD and Semantic Segmentation ports that were sources of inspiration:
[lufficc/SSD](https://github.com/lufficc/SSD)       
[ansleliu/LightNetPlusPlus](https://github.com/ansleliu/LightNetPlusPlus)




