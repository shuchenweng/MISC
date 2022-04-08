# MISC: Multi-condition injection and spatially-adaptive compositing for conditional person rendering (unofficial)

## Introducation
This is the author's unofficial PyTorch MISC implementation.

We present MISC (Multi-condition Injection and Spatially-adaptive Compositing), an end-to-end trainable deep neural network for conditional person image synthesis. MISC includes a conditional person generation model and a spatially-adaptive image composition model.

<!-- ![test image size](https://github.com/shuchenweng/MISC/blob/main/setting.png){:height="50%" width="50%"} -->
 <img src="https://github.com/shuchenweng/MISC/blob/main/setting.png" width = "627" height = "434" alt="图片名称" align=center />

Given attribute (color condition), segmentation mask (geometry condition), Gaussian noise (texture condition), and background image, our model could generate the specified person in user-designated image position.

## Prerequisites
* Python 3.6
* PyTorch 1.0
* NVIDIA GPU + CUDA cuDNN

## Installation
Clone this repo: 
```
git clone https://github.com/shuchenweng/MISC.git
```
Install PyTorch and dependencies from http://pytorch.org

Install other python requirements

## Datasets
We process the [VIP person parsing](https://github.com/HCPLab-SYSU/ATEN.git) dataset for evaluation. We annotate persons in VIP with 120 attribute classes, and crop the images in VIP to keep one major person in each image. We create the training and test splits, with 42K and 6K images, respectively.

## Getting Started
Download the [proccessed VIP dataset](https://drive.google.com/file/d/13to7_krxUlW6bYiA2EojhsxJ5wGnF40s/view?usp=sharing) and copy them under DATA_DIR.

Download the [pre-trained weights](https://drive.google.com/drive/folders/1irNyIJbuPBOf03Fa1QDCw7Hocxn3djeq?usp=sharing) and copy them under PRETRAINED_DIR. 

Setting the MODEL_DIR as the storage directory for generated experimental results.

These directory parameters could be found in cfg/test_SC.yml and cfg/train_SC.yml. 

### 1) Training
```
python main.py --cfg train_SC.yml
```
### 2) Testing
```
python main.py --cfg test_SC.yml
```

## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/).

Except where otherwise noted, this content is published under a [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/) license, which means that you can copy, remix, transform and build upon the content as long as you do not use the material for commercial purposes and give appropriate credit and provide a link to the license.

## Citation
If you use this code for your research, please cite our papers [MISC: Multi-condition Injection and Spatially-adaptive Compositing for Conditional Person Image Synthesis](https://ci.idm.pku.edu.cn/CVPR20c.pdf)
```
@InProceedings{Weng_2020_CVPR,
  author = {Weng, Shuchen and Li, Wenbo and Li, Dawei and Jin, Hongxia and Shi, Boxin},
  title = {MISC: Multi-Condition Injection and Spatially-Adaptive Compositing for Conditional Person Image Synthesis},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
