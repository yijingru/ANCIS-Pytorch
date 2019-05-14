import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CombinationModule(nn.Module):
    def __init__(self, in_size, out_size, cat_size):
        super(CombinationModule, self).__init__()
        self.up =  nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, stride=1),
                                 nn.ReLU(inplace=True))
        self.cat =  nn.Sequential(nn.Conv2d(cat_size, out_size, kernel_size=1, stride=1),
                                 nn.ReLU(inplace=True))

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(F.upsample_bilinear(inputs2,inputs1.size()[2:]))
        return self.cat(torch.cat([inputs1, outputs2], 1))


class Attention(nn.Module):
    def __init__(self, x_channel, y_channel):
        super(Attention,self).__init__()
        self.conv1 = nn.Conv2d(x_channel, y_channel, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(y_channel, y_channel, kernel_size=1, stride=1)

    def forward(self, x, y):
        out = F.upsample_bilinear(self.conv1(x), y.size()[2:])
        out = F.relu(out+y, inplace=True)
        out = F.sigmoid(out)
        out = F.relu(self.conv2(out*y),inplace=True)
        return out

def make_attention_layers():
    layers = []
    layers += [Attention(64, 64)]
    layers += [Attention(256, 64)]
    layers += [Attention(512, 256)]
    layers += [Attention(1024, 512)]
    return layers


def make_concat_layers():
    layers = []
    layers += [CombinationModule(64, 64, 128)]
    layers += [CombinationModule(256, 64, 128)]
    layers += [CombinationModule(512, 256, 512)]
    layers += [CombinationModule(1024, 512, 1024)]
    return layers

class SEG_NET(nn.Module):
    def __init__(self, num_classes):
        super(SEG_NET,self).__init__()
        self.num_classes = num_classes
        self.layer_att = nn.ModuleList(make_attention_layers())
        self.layer_up_concat = nn.ModuleList(make_concat_layers())

        self.layer_c0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True))

        self.layer_head = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 1, kernel_size=3, padding=1))

    def get_patches(self, box, feat):
        y1, x1, y2, x2 = box
        _, h, w = feat.size()
        y1 = np.maximum(0, np.int32(np.round(y1 * h)))
        x1 = np.maximum(0, np.int32(np.round(x1 * w)))
        y2 = np.minimum(np.int32(np.round(y2 * h)), h - 1)
        x2 = np.minimum(np.int32(np.round(x2 * w)), w - 1)
        if y2<y1 or x2<x1 or y2-y1<2 or x2-x1<2:
            return None
        else:
            return (feat[:, y1:y2+1, x1:x2+1].unsqueeze(0))


    def mask_forward(self, i_x):
        pre = None
        for i in range(len(i_x)-1, -1, -1):
            if pre is None:
                pre = i_x[i]
            else:
                attents = self.layer_att[i](pre, i_x[i])
                pre = self.layer_up_concat[i](attents, pre)

        x = self.layer_head(pre)
        x = F.sigmoid(x)
        x = torch.squeeze(x, dim=0)
        x = torch.squeeze(x, dim=0)
        return x


    def forward(self, detections, feat_seg):
        feat_seg[0] = self.layer_c0(feat_seg[0])

        mask_patches = [[] for i in range(detections.size(0))]
        mask_dets = [[] for i in range(detections.size(0))]

        # iterate batch
        for i in range(detections.size(0)):
            # iterate class
            for j in range(1, detections.size(1)):
                dects = detections[i, j, :]
                mask = dects[:, 0].gt(0.).expand(5, dects.size(0)).t()
                dects = torch.masked_select(dects, mask).view(-1, 5)
                if dects.shape[0] == 0:
                    continue
                if j:
                    for box, score in zip(dects[:, 1:], dects[:, 0]):
                        i_x = []
                        y1, x1, y2, x2 = box
                        h,w = feat_seg[0].shape[2:]
                        for i_feat in range(len(feat_seg)):
                            x = self.get_patches([y1/h,x1/w,y2/h,x2/w], feat_seg[i_feat][i, :, :, :])
                            if x is None:
                                break
                            else:
                                i_x.append(x)
                        # ~~~~~~~~~~~~~~ Decoder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                        if len(i_x):
                            x = self.mask_forward(i_x)  # up pooled mask patch
                            mask_patches[i].append(x)
                            mask_dets[i].append(torch.Tensor(np.append(box,[score,j])))

        output = (mask_patches, mask_dets)

        return output