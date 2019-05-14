from torch.utils.data.dataset import Dataset
import glob
import cv2
import numpy as np
import os
from skimage.measure import label, regionprops

def load_gt_kaggle(annoDir):
    bboxes = []
    labels = []
    masks = []
    files = os.listdir(annoDir)
    for anno_file in files:
        mask = cv2.imread(os.path.join(annoDir, anno_file), -1)
        labelImg = label(np.where(mask>0, 1., 0.))
        props = regionprops(labelImg)
        props = sorted(props, key=lambda x: x.area, reverse=True)
        r,c = np.where(labelImg==props[0].label)
        y1 = np.min(r)
        x1 = np.min(c)
        y2 = np.max(r)
        x2 = np.max(c)
        if (abs(y2 - y1) <= 1 or abs(x2 - x1) <= 1):
            continue
        bboxes.append([y1, x1, y2, x2])
        labels.append([1])
        masks.append(np.where(mask>0, 1., 0.))
    return bboxes, labels, masks

class NucleiCell(Dataset):
    def __init__(self, imgDirectory, annoDirectory, transform=None, imgSuffix='.jpg', annoSuffix='.png'):
        super(NucleiCell, self).__init__()
        self.imgDirectory = imgDirectory
        # self.annoDirectory = annoDirectory
        self.transform = transform
        self.classes = {0: 'background', 1: 'cell'}
        self.labelmap = {'cell'}
        self.imgSuffix = imgSuffix
        self.annoSuffix = annoSuffix
        self.img_files = self.load_img_ids()

    def load_img_ids(self):
        img_files = [os.path.join(self.imgDirectory, x, "images", x+self.imgSuffix)
                     for x in os.listdir(self.imgDirectory)]
        return img_files

    def load_img(self, item):
        img = cv2.imread(self.img_files[item])
        return img

    def load_annotation(self, img_file):
        barcode = img_file.split('/')[-1].split('.')[0]
        annoDir = os.path.join(self.imgDirectory, barcode, "masks")
        bboxes, labels, masks = load_gt_kaggle(annoDir)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        masks  = np.asarray(masks, dtype=np.float32)
        return bboxes, labels, masks

    def __getitem__(self, item):
        img = self.load_img(item)
        bboxes, labels, masks = self.load_annotation(self.img_files[item])
        if self.transform is not None:
            img, bboxes, labels, masks = self.transform(img, bboxes, labels, masks)
        img = img/255
        return img, bboxes, labels, masks


    def __len__(self):
        return len(self.img_files)