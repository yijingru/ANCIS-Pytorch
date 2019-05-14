import argparse
import torch.optim as optim
from torch.optim import lr_scheduler

from seg_utils import *
from dec_utils import *
from seg_utils import seg_transforms, seg_dataset_kaggle, seg_eval_kaggle

from models import dec_net_seg, seg_net
import cv2
import os

parser = argparse.ArgumentParser(description='Detection Training (MultiGPU)')
parser.add_argument('--testDir', default="/home/grace/PycharmProjects/DataSets/kaggle/test", type=str, help='test image directory')
parser.add_argument('--annoDir', default="data/root/mask", type=str, help='annotation image directory')
parser.add_argument('--imgSuffix', default='.png', type=str, help='suffix of the input images')
parser.add_argument('--annoSuffix', default='.png', type=str, help='suffix of the annotation images')
parser.add_argument('--img_height', default=512, type=int, help='img height')
parser.add_argument('--img_width', default=512, type=int, help='img width')
parser.add_argument('--num_classes', default=2, type=int, help='dataset classes')
parser.add_argument('--top_k', default=500, type=int, help='the number of detections to keep')
parser.add_argument('--conf_thresh', default=0.3, type=float, help='confidence threshold')
parser.add_argument('--nms_thresh', default=0.3, type=float, help='nms threshold')
parser.add_argument('--seg_thresh', default=0.5, type=float, help='segmentation threshold')
parser.add_argument('--dec_weights', default="dec_weights/kaggle/end_model.pth", type=str, help='detection weights')
parser.add_argument('--seg_weights', default="seg_weights/kaggle/end_model.pth", type=str, help='segmentation weights')


def evaluation(args):
    #-----------------load detection model -------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dec_model = dec_net_seg.resnetssd50(pretrained=False, num_classes=args.num_classes)
    resume_dict = torch.load(args.dec_weights)
    resume_dict = {k[7:]: v for k, v in resume_dict.items()}
    dec_model.load_state_dict(resume_dict)
    dec_model = dec_model.to(device)
    dec_model.eval()
    #-----------------load segmentation model -------------------------
    seg_model =  seg_net.SEG_NET(num_classes=args.num_classes)
    seg_model.load_state_dict(torch.load(args.seg_weights))
    seg_model= seg_model.to(device)
    seg_model.eval()
    ##--------------------------------------------------------------
    data_transforms = seg_transforms.Compose([seg_transforms.ConvertImgFloat(),
                                       seg_transforms.Resize(args.img_height, args.img_width),
                                       seg_transforms.ToTensor()])


    dsets = seg_dataset_kaggle.NucleiCell(args.testDir, args.annoDir, data_transforms,
                                          imgSuffix=args.imgSuffix, annoSuffix=args.annoSuffix)

    # for validation data -----------------------------------
    detector = Detect(num_classes=args.num_classes,
                      top_k=args.top_k,
                      conf_thresh=args.conf_thresh,
                      nms_thresh=args.nms_thresh,
                      variance=[0.1, 0.2])
    anchorGen = Anchors(args.img_height, args.img_width)
    anchors = anchorGen.forward()

    ap_05, ap_07 = seg_eval_kaggle.do_python_eval(dsets=dsets, dec_model=dec_model, seg_model=seg_model,
                                                  detector=detector, anchors=anchors, device=device,
                                                  args=args, offline=True)

    print('Finish')


if __name__ == '__main__':
    args = parser.parse_args()
    evaluation(args)