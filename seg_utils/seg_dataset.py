from torch.utils.data.dataset import Dataset
import glob
import cv2
import numpy as np
import os

def load_gt_mask_neural_cell(annopath):
    """
    :return: [y1, x1, y2, x2, cls] in original sizes
    """
    bboxes = []
    labels = []
    masks  = []
    mask_gt = cv2.imread(annopath)
    h,w,_ = mask_gt.shape
    cond1 = mask_gt[:, :, 0] != mask_gt[:, :, 1]
    cond2 = mask_gt[:, :, 1] != mask_gt[:, :, 2]
    cond3 = mask_gt[:, :, 2] != mask_gt[:, :, 0]

    r,c = np.where(np.logical_or(np.logical_or(cond1, cond2), cond3))
    unique_colors = np.unique(mask_gt[r, c, :], axis=0)

    for color in unique_colors:
        cond1 = mask_gt[:, :, 0] == color[0]
        cond2 = mask_gt[:, :, 1] == color[1]
        cond3 = mask_gt[:, :, 2] == color[2]
        r,c = np.where(np.logical_and(np.logical_and(cond1, cond2), cond3))
        y1 = np.min(r)
        x1 = np.min(c)
        y2 = np.max(r)
        x2 = np.max(c)
        if (abs(y2-y1)<=1 or abs(x2-x1)<=1):
            continue
        bboxes.append([y1, x1, y2, x2])   # 512 x 640
        labels.append([1])

        cur_gt_mask = np.where(np.logical_and(np.logical_and(cond1, cond2), cond3), 1., 0.)
        masks.append(cur_gt_mask)

    return bboxes, labels, masks

class NeuralCell(Dataset):
    def __init__(self, imgDirectory, annoDirectory, transform=None, imgSuffix='.jpg', annoSuffix='.png'):
        super(NeuralCell, self).__init__()
        self.imgDirectory = imgDirectory
        self.annoDirectory = annoDirectory
        self.transform = transform
        self.classes = {0: 'background', 1: 'cell'}
        self.labelmap = {'cell'}
        self.imgSuffix = imgSuffix
        self.annoSuffix = annoSuffix
        self.img_files = self.load_img_ids()

    def load_img_ids(self):
        img_files = [x for x in sorted(glob.glob(os.path.join(self.imgDirectory, "*"+self.imgSuffix)))]
        return img_files

    def load_annotation(self, img_file):
        anno_file = os.path.join(self.annoDirectory, img_file.split('/')[-1].split('.')[0] + self.annoSuffix)
        bboxes, labels, masks = load_gt_mask_neural_cell(anno_file)
        bboxes = np.asarray(bboxes, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)
        masks  = np.asarray(masks, dtype=np.float32)
        return bboxes, labels, masks

    def load_img(self, item):
        img = cv2.imread(self.img_files[item])
        return img

    def __getitem__(self, item):
        img = self.load_img(item)
        bboxes, labels, masks = self.load_annotation(self.img_files[item])
        if self.transform is not None:
            img, bboxes, labels, masks = self.transform(img, bboxes, labels, masks)
        return img, bboxes, labels, masks


    def __len__(self):
        return len(self.img_files)