import numpy as np
from numpy import random
import cv2
import torch


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes=None, labels=None):
        for t in self.transforms:
            img, bboxes, labels = t(img, bboxes, labels)
        return img, bboxes, labels


class ConvertImgFloat(object):
    def __call__(self, img, bboxes=None, labels=None):
        return img.astype(np.float32), bboxes, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, bboxes, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, bboxes, labels

class SwapChannels(object):
    def __init__(self, swaps):
        self.swaps = swaps
    def __call__(self, img):
        img = img[:, :, self.swaps]
        return img


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)
            img = shuffle(img)
        return img, bboxes, labels


class PhotometricDistort(object):
    def __init__(self):
        self.pd = RandomContrast()
        self.rb = RandomBrightness()
        self.rln = RandomLightingNoise()

    def __call__(self, img, bboxes=None, labels=None):
        img, bboxes, labels = self.rb(img, bboxes, labels)
        if random.randint(2):
            distort = self.pd#Compose(self.pd)
        else:
            distort = self.pd#Compose(self.pd)
        img, bboxes, labels = distort(img, bboxes, labels)
        img, bboxes, labels = self.rln(img, bboxes, labels)
        return img, bboxes, labels


class Expand(object):
    def __init__(self, max_scale = 2, mean = (0.485, 0.456, 0.406)):
        self.mean  = mean
        self.max_scale = max_scale

    def __call__(self, img, bboxes=None, labels=None):
        if random.randint(2):
            return img, bboxes, labels
        h,w,c = img.shape
        ratio = random.uniform(1, self.max_scale)
        y1 = random.uniform(0, h*ratio-h)
        x1 = random.uniform(0, w*ratio-w)
        expand_img = np.zeros(shape=(int(h*ratio), int(w*ratio),c),dtype=img.dtype)
        expand_img[:,:,:] = self.mean
        expand_img[int(y1):int(y1+h), int(x1):int(x1+w)] = img
        img = expand_img

        bboxes[:,0::2] += float(int(y1))
        bboxes[:,1::2] += float(int(x1))

        return img, bboxes, labels

def intersect(boxes_a, box_b):
    max_yx = np.minimum(boxes_a[:,2:], box_b[2:])
    min_yx = np.maximum(boxes_a[:,:2], box_b[:2])
    inter = np.clip((max_yx-min_yx), a_min=0., a_max=np.inf)
    return inter[:,0]*inter[:,1]

def jaccard_numpy(boxes_a, box_b):
    inter = intersect(boxes_a, box_b)
    area_a = ((boxes_a[:,2]-boxes_a[:,0])*(boxes_a[:,3]-boxes_a[:,1]))
    area_b = ((box_b[2]-box_b[0])*(box_b[3]-box_b[1]))
    union = area_a+area_b-inter
    return inter/union


class RandomSampleCrop(object):
    def __init__(self, ratio=(0.5, 1.5), min_win = 0.9):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            # (0.1, None),
            # (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )
        self.ratio = ratio
        self.min_win = min_win

    def __call__(self, img, bboxes=None, labels=None):
        height, width ,_ = img.shape
        while True:
            mode = random.choice(self.sample_options)
            if mode is None:
                return img, bboxes, labels
            min_iou, max_iou = mode

            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                current_img = img
                w = random.uniform(self.min_win*width, width)
                h = random.uniform(self.min_win*height, height)
                if h/w<self.ratio[0] or h/w>self.ratio[1]:
                    continue
                y1 = random.uniform(height-h)
                x1 = random.uniform(width-w)
                rect = np.array([int(y1), int(x1), int(y1+h), int(x1+w)])
                overlap = jaccard_numpy(bboxes, rect)
                if overlap.min()<min_iou and max_iou<overlap.max():
                    continue
                current_img = current_img[rect[0]:rect[2], rect[1]:rect[3], :]
                centers = (bboxes[:,:2]+bboxes[:,2:])/2.0
                mask1 = (rect[0]<centers[:,0])*(rect[1]<centers[:,1])
                mask2 = (rect[2]>centers[:,0])*(rect[3]>centers[:,1])
                mask = mask1*mask2
                if not mask.any():
                    continue
                current_boxes = bboxes[mask,:].copy()
                current_labels = labels[mask]
                current_boxes[:,:2] = np.maximum(current_boxes[:,:2], rect[:2])
                current_boxes[:,:2]-=rect[:2]
                current_boxes[:,2:] = np.minimum(current_boxes[:,2:], rect[2:])
                current_boxes[:,2:]-=rect[:2]
                return current_img, current_boxes, current_labels

class RandomMirror_w(object):
    def __call__(self, img, bboxes, classes):
        _,w,_ = img.shape
        if random.randint(2):
            img = img[:,::-1]
            bboxes[:,1::2] = w-bboxes[:,3::-2]
        return img, bboxes, classes

class RandomMirror_h(object):
    def __call__(self, img, bboxes, classes):
        h,_,_ = img.shape
        if random.randint(2):
            img = img[::-1,:]
            bboxes[:,0::2] = h-bboxes[:,2::-2]
        return img, bboxes, classes


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img, bboxes, classes):
        h,w,c = img.shape
        bboxes = bboxes.astype(np.float32)
        bboxes[:, 0] /= h
        bboxes[:, 1] /= w
        bboxes[:, 2] /= h
        bboxes[:, 3] /= w
        img = cv2.resize(img, dsize=(self.width, self.height))
        bboxes[:, 0] *= self.height
        bboxes[:, 1] *= self.width
        bboxes[:, 2] *= self.height
        bboxes[:, 3] *= self.width
        return img, bboxes, classes

class ToTensor(object):
    def __call__(self, img, bboxes, classes):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy().transpose((2,0,1)))

        if isinstance(bboxes, np.ndarray):
            bboxes = torch.FloatTensor(bboxes)

        if isinstance(classes, np.ndarray):
            classes = torch.Tensor(classes)

        return img, bboxes, classes


