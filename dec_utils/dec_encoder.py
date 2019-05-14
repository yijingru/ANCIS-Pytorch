import torch

def encode(match_boxes, priors, variances):
    c_yx = (match_boxes[:,:2]+match_boxes[:,2:]).float()/2-priors[:,:2]
    c_yx = c_yx.float()/(variances[0]*priors[:,2:])

    hw = (match_boxes[:,2:]-match_boxes[:,:2]).float()/priors[:,2:]
    hw = torch.log(hw.float())/variances[1]

    return torch.cat([c_yx, hw], 1)

def split_to_box(priors):
    return torch.cat([priors[:,:2]-priors[:,2:]/2, priors[:,:2]+priors[:,2:]/2], 1)

def intersect(boxes_a, boxes_b):
    num_a = boxes_a.size(0)
    num_b = boxes_b.size(0)
    max_xy = torch.min(boxes_a[:,2:].unsqueeze(1).expand(num_a,num_b,2),
                       boxes_b[:,2:].unsqueeze(0).expand(num_a,num_b,2))
    min_xy = torch.max(boxes_a[:,:2].unsqueeze(1).expand(num_a,num_b,2),
                       boxes_b[:,:2].unsqueeze(0).expand(num_a,num_b,2))

    inter = torch.clamp((max_xy-min_xy), min=0.)
    return inter[:,:,0]*inter[:,:,1]


def jaccard(boxes_a, boxes_b):
    inter = intersect(boxes_a, boxes_b)
    area_a = ((boxes_a[:,2]-boxes_a[::,0])*(boxes_a[:,3]-boxes_a[:,1])).unsqueeze(1).expand_as(inter)
    area_b = ((boxes_b[:,2]-boxes_b[::,0])*(boxes_b[:,3]-boxes_b[:,1])).unsqueeze(0).expand_as(inter)
    union = area_a+area_b-inter
    return inter/union

def encode_gt_to_anchors(gt_boxes, gt_label, anchors, match_thresh, variances):
    # gt_boxes:  y1,x1,y2,x2  original_sizes
    # anchors:    cy,cx,h,w    original_sizes
    # transfer:  y1,x1,y2,x2  off_sets
    anchors_box = split_to_box(anchors)
    overlaps = jaccard(gt_boxes, anchors_box) # [num_gt, num_anchors, 1]
    best_gt, best_gt_idx = overlaps.max(0, keepdim=True)
    best_gt.squeeze_(0)
    best_gt_idx.squeeze_(0)

    best_anchor, best_anchor_idx = overlaps.max(1, keepdim=True)
    best_anchor.squeeze_(1)
    best_anchor_idx.squeeze_(1)

    best_gt.index_fill_(0, best_anchor_idx, 2)

    for j in range(best_anchor_idx.size(0)): #iterate num_a
        best_gt_idx[best_anchor_idx[j]] = j
    match_boxes = gt_boxes[best_gt_idx]
    encoded_anchor_labels = gt_label[best_gt_idx]
    encoded_anchor_labels[best_gt<match_thresh] = 0.
    encoded_anchor_bboxes = encode(match_boxes,anchors,variances)
    return encoded_anchor_bboxes, encoded_anchor_labels