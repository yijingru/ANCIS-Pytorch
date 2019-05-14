import torch.nn.functional as F
import torch

## nms, decode, detect
def decode(locs, priors, variances):
    # locs:    num_priors x 4
    # priors:  num_priors x 4
    boxes = torch.cat([priors[:, :2] + locs[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(locs[:, 2:] * variances[1])], 1)
    boxes[:, 0] = boxes[:, 0]-boxes[:, 2]/2
    boxes[:, 1] = boxes[:, 1]-boxes[:, 3]/2
    boxes[:, 2] = boxes[:, 2]+boxes[:, 0]
    boxes[:, 3] = boxes[:, 3]+boxes[:, 1]
    return boxes

def nms(boxes, scores, nms_thresh=0.5, top_k=200):
    # boxes shape[-1, 4]
    # scores shape [-1,]
    scores = scores
    boxes = boxes
    keep = scores.new(scores.size(0)).zero_().long()# create a new tensor
    if boxes.numel() == 0: # return the total elements number in boxes
        return keep
    y1 = boxes[:, 0]
    x1 = boxes[:, 1]
    y2 = boxes[:, 2]
    x2 = boxes[:, 3]
    area = torch.mul(y2-y1,x2-x1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    yy1 = boxes.new()  # create a new tensor of the same type
    xx1 = boxes.new()
    yy2 = boxes.new()
    xx2 = boxes.new()
    h = boxes.new()
    w = boxes.new()
    count = 0
    while idx.numel()>0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        # doing about remains...
        # select the remaining boxes
        torch.index_select(y1, dim=0, index=idx, out=yy1)
        torch.index_select(x1, dim=0, index=idx, out=xx1)
        torch.index_select(y2, dim=0, index=idx, out=yy2)
        torch.index_select(x2, dim=0, index=idx, out=xx2)

        # calculate the inter boxes clamp with box i
        yy1 = torch.clamp(yy1, min=y1[i])
        xx1 = torch.clamp(xx1, min=x1[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        xx2 = torch.clamp(xx2, max=x2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = xx2-xx1
        h = yy2-yy1

        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        inter = w*h

        rem_areas = torch.index_select(area, dim=0, index=idx)
        union = (rem_areas-inter)+area[i]
        IoU = inter/union
        idx = idx[IoU.le(nms_thresh)]
    return keep, count

class Detect(object):
    def __init__(self, num_classes, top_k, conf_thresh, nms_thresh, variance):
        self.num_classes = num_classes
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.output = torch.zeros(1, self.num_classes, self.top_k, 5)

    def __call__(self, locs, confs, anchors):
        # locs:   batch x num_anchors x 4        torch.Size([1, 30080, 4])    cuda:0
        # confs:  batch x num_anchors x 21       torch.Size([1, 30080, 2])    cuda:0
        # anchors: num_anchors x 4 [cy,cx,h,w]    torch.Size([30080, 4])       cpu
        # self.output.zero_()

        confs = F.softmax(confs, dim=2)
        num_batch = locs.size(0)

        locs  = locs.data.cpu()
        confs = confs.data.cpu()

        output = torch.zeros(num_batch, self.num_classes, self.top_k, 5)

        # Decoding...
        for i in range(num_batch):
            decoded_boxes_i = decode(locs[i], anchors, torch.Tensor(self.variance))
            p_conf_i = confs[i]
            for cl in range(1, self.num_classes):
                cl_mask = p_conf_i[:, cl].gt(self.conf_thresh)
                p_conf_i_cl = p_conf_i[:, cl][cl_mask]
                if p_conf_i_cl.shape[0] == 0:
                    continue
                loc_mask = cl_mask.unsqueeze(1).expand_as(decoded_boxes_i)
                p_boxes_i_cl = decoded_boxes_i[loc_mask].view(-1,4)
                ids, count = nms(boxes=p_boxes_i_cl,
                                 scores=p_conf_i_cl,
                                 nms_thresh=self.nms_thresh)
                output[i, cl, :count] = torch.cat((p_conf_i_cl[ids[:count]].unsqueeze(1),
                                                   p_boxes_i_cl[ids[:count]]),1)

        return output
