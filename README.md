# ANCIS-Pytorch
Attentive Neural Cell Instance Segmentation
https://www.sciencedirect.com/science/article/pii/S1361841518308442


Please  cite  this  article  as:  
Jingru Yi,  Pengxiang Wu,  Menglin Jiang,  Qiaoying Huang, Daniel J. Hoeppner, Dimitris N. Metaxas, Attentive Neural Cell Instance Segmentation, Medical Image Analysis(2019), 
doi: https://doi.org/10.1016/j.media.2019.05.004
 

## Dependencies
Library: opencv-python, PyTorch>0.4.0, Ubuntu 14.04

## Implementation Details
To accelerate the training process, we trained the detection and segmentation modules separately.  In particular,the weights of the detection module are frozen when training the segmentation module.