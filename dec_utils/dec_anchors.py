import torch
from itertools import product as product
import math
import numpy as np


class Anchors(object):
    def __init__(self, img_height, img_width):
        super(Anchors, self).__init__()
        min_scale = np.array([0.04, 0.1 , 0.26, 0.42])
        max_scale = np.array([0.1 , 0.26, 0.42, 0.58])
        self.img_size = np.array([img_height, img_width])  #array([512, 640])
        self.pyramid_levels = [3, 4, 5, 6]
        self.feat_shapes = [(self.img_size + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        self.min_sizes = min_scale * self.img_size[0]  #array([ 20.48,  51.2 , 133.12])
        self.max_sizes = max_scale * self.img_size[0]  #array([ 51.2 , 133.12, 215.04])
        self.steps = [2 ** x for x in self.pyramid_levels]  #[8, 16, 32]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3]]

    def forward(self):
        coords = []
        for k, f in enumerate(self.feat_shapes):
            for i,j in product(range(f[0]), range(f[1])):
                cy = (i+0.5) * self.steps[k]
                cx = (j+0.5) * self.steps[k]
                # ar = 1
                h0 = float(self.min_sizes[k])
                w0 = float(self.min_sizes[k])
                coords.append([cy, cx, h0, w0])
                h1 = math.sqrt(h0 * float(self.max_sizes[k]))
                w1 = math.sqrt(w0 * float(self.max_sizes[k]))
                coords.append([cy, cx, h1, w1])
                # ar != 1
                for a_r in self.aspect_ratios[k]:
                    h = float(h0)/math.sqrt(a_r)
                    w = float(w0)*math.sqrt(a_r)
                    coords.append([cy, cx, h, w])
                    coords.append([cy, cx, w, h])

        output = torch.Tensor(coords).view(-1,4) # [30080, 4]
        return output #[cy, cx, h, w]

