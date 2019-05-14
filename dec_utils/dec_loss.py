import torch.nn as nn
from .dec_anchors import Anchors
import torch
from .dec_encoder import encode_gt_to_anchors
import torch.nn.functional as F

class DecLoss(nn.Module):
    def __init__(self, img_height, img_width, num_classes, variances):
        super(DecLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = 3
        anchorGen = Anchors(img_height, img_width)
        self.anchors = anchorGen.forward()
        self.num_anchors = self.anchors.shape[0]
        self.match_thresh = 0.5
        self.variances = variances

    def log_sum_exp(self, x):
        """This will be used to determine un-averaged confidence loss across
        all examples in a batch.
        """
        # x: [-1, num_classes]
        x_max = x.data.max() # get the max value of all - > output one value
        return torch.log(torch.sum(torch.exp(x-x_max), dim=1, keepdim=True))+x_max


    def forward(self, predictions, gt_bboxes, gt_labels):
        """
        torch.Size([2, 30080, 4])
        torch.Size([2, 30080, 2])
        torch.Size([23360, 4])

        """
        p_locs, p_conf = predictions
        batch_size = p_conf.shape[0]

        # encode the matched groundtruth...
        t_locs = torch.FloatTensor(batch_size, self.num_anchors, 4)
        t_conf = torch.LongTensor(batch_size, self.num_anchors, 1)
        for idx in range(batch_size):
            t_boxes = gt_bboxes[idx]
            t_label = gt_labels[idx]
            d_boxes = self.anchors
            encoded_boxes, encoded_label = encode_gt_to_anchors(gt_boxes=t_boxes,
                                                                gt_label=t_label,
                                                                anchors=d_boxes,
                                                                match_thresh=self.match_thresh,
                                                                variances=self.variances)
            t_locs[idx] = encoded_boxes
            t_conf[idx] = encoded_label

        t_locs = t_locs.to(p_locs.device)
        t_conf = t_conf.to(p_conf.device)

        pos_mask = t_conf>0 # batch x num_box
        num_pos = pos_mask.long().sum(dim=1,keepdim=True)

        # locs loss
        pos_locs_mask = pos_mask.expand_as(p_locs)
        loss_locs = F.smooth_l1_loss(input = p_locs[pos_locs_mask].view(-1,4),
                                     target= t_locs[pos_locs_mask].view(-1,4),
                                     size_average=False)

        # conf loss
        # hard negtive mining
        pos_mask = pos_mask.squeeze(2)
        p_conf_batch = p_conf.view(-1, self.num_classes)
        temp = self.log_sum_exp(p_conf_batch)-p_conf_batch.gather(dim=1, index=t_conf.view(-1,1))
        temp = temp.view(batch_size, -1)
        temp[pos_mask] = 0.
        _, temp_idx = temp.sort(1, descending=True)
        _, idx_rank = temp_idx.sort(1)
        num_neg = torch.clamp(self.neg_pos_ratio*num_pos, max=pos_mask.size(1)-1).squeeze(2)
        neg_mask = idx_rank<num_neg.expand_as(idx_rank)

        # conf loss calc
        pos_conf_mask = pos_mask.unsqueeze(2).expand_as(p_conf)
        neg_conf_mask = neg_mask.unsqueeze(2).expand_as(p_conf)
        loss_conf = F.cross_entropy(input=p_conf[(pos_conf_mask+neg_conf_mask).gt(0)].view(-1,self.num_classes),
                                    target=t_conf[(pos_mask+neg_mask).gt(0)].squeeze(1),
                                    size_average=False)
        N = num_pos.data.sum()
        if N == 0:
            N = 1.
        N = N.float()
        return loss_locs/N, loss_conf/N