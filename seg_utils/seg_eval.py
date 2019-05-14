import numpy as np
import cv2


def voc_ap(rec, prec, use_07_metric=False):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :param use_07_metric: 2007 metric is 11-recall-point based AP
    :return: average precision
    """
    if use_07_metric:
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def mask_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = mask1.sum() + mask2.sum() - inter
    if union < 1.0:
        return 0
    return float(inter) / float(union)

def voc_eval_sds(dsets, dec_model, seg_model,
                 detector, anchors, device,
                 args, offline,
                 ov_thresh, use_07_metric=True):
    all_tp = []
    all_fp = []
    all_scores = []
    temp_overlaps = []
    npos = 0
    for img_idx in range(len(dsets)):
        if offline:
            print('loading {}/{} image'.format(img_idx, len(dsets)))
        inputs, gt_boxes, gt_classes, gt_masks = dsets.__getitem__(img_idx)
        x = inputs.unsqueeze(0)
        x = x.to(device)
        locs, conf, feat_seg = dec_model(x)
        detections = detector(locs, conf, anchors)
        outputs = seg_model(detections, feat_seg)
        mask_patches, mask_dets = outputs
        # For batches
        for b_mask_patches, b_mask_dets in zip(mask_patches, mask_dets):
            nd = len(b_mask_dets)
            BB_conf = []
            BB_mask = []
            # Step1: rearrange mask_patches and mask_dets
            for d in range(nd):
                d_mask = np.zeros((args.img_height, args.img_width), dtype=np.float32)
                d_mask_det = b_mask_dets[d].data.cpu().numpy()
                d_mask_patch = b_mask_patches[d].data.cpu().numpy()
                d_bbox = d_mask_det[0:4]
                d_conf = d_mask_det[4]
                d_class = d_mask_det[5]
                if d_conf < args.conf_thresh:
                    continue
                [y1, x1, y2, x2] = d_bbox
                y1 = np.maximum(0, np.int32(np.round(y1)))
                x1 = np.maximum(0, np.int32(np.round(x1)))
                y2 = np.minimum(np.int32(np.round(y2)), args.img_height - 1)
                x2 = np.minimum(np.int32(np.round(x2)), args.img_width - 1)
                d_mask_patch = cv2.resize(d_mask_patch, (x2 - x1 + 1, y2 - y1 + 1))
                d_mask_patch = np.where(d_mask_patch >= args.seg_thresh, 1., 0.)
                d_mask[y1:y2 + 1, x1:x2 + 1] = d_mask_patch
                BB_conf.append(d_conf)
                BB_mask.append(d_mask)
            BB_conf = np.asarray(BB_conf, dtype=np.float32)
            BB_mask = np.asarray(BB_mask, dtype=np.float32)
            # Step2: sort detections according to the confidences
            sorted_ind = np.argsort(-BB_conf)
            BB_mask = BB_mask[sorted_ind, :, :]
            BB_conf = BB_conf[sorted_ind]
            all_scores.extend(BB_conf)
            # Step2: intialzation of evaluations
            nd = BB_mask.shape[0]
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            BBGT_box, BBGT_label, BBGT_mask = dsets.load_annotation(dsets.img_files[img_idx])
            nd_gt = BBGT_box.shape[0]
            det_flag = [False] * nd_gt
            npos = npos + nd_gt
            for d in range(nd):
                d_BB_mask = BB_mask[d, :, :]
                # calculate max region overlap
                ovmax = -1000
                jmax = -1
                for ind2 in range(len(BBGT_mask)):
                    gt_mask = BBGT_mask[ind2]
                    overlaps = mask_iou(d_BB_mask, gt_mask)
                    if overlaps > ovmax:
                        ovmax = overlaps
                        jmax = ind2

                if ovmax > ov_thresh:
                    if not det_flag[jmax]:
                        tp[d] = 1.
                        det_flag[jmax] = 1
                        temp_overlaps.append(ovmax)
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.
            all_fp.extend(fp)
            all_tp.extend(tp)
    # step5: compute precision recall
    all_fp = np.asarray(all_fp)
    all_tp = np.asarray(all_tp)
    all_scores = np.asarray(all_scores)
    sorted_ind = np.argsort(-all_scores)
    all_fp = all_fp[sorted_ind]
    all_tp = all_tp[sorted_ind]
    all_fp = np.cumsum(all_fp)
    all_tp = np.cumsum(all_tp)
    rec = all_tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = all_tp / np.maximum(all_tp + all_fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    return ap, np.mean(temp_overlaps)


def do_python_eval(dsets, dec_model, seg_model,
                   detector, anchors, device,
                   args, offline=True):
    for i in range(len(dsets.classes)):
        cls = dsets.classes[i]
        if cls == 'background':
            continue
        ap_05, temp_overlaps = voc_eval_sds(dsets, dec_model, seg_model,
                                            detector, anchors, device,
                                            args, offline,
                                            ov_thresh=0.5)
        print('AP@0.5 for {} = {:.2f}, overlap = {:.4f}'.format(cls, ap_05 * 100, temp_overlaps))

    # aps = []
    for i in range(len(dsets.classes)):
        cls = dsets.classes[i]
        if cls == 'background':
            continue
        ap_07, temp_overlaps = voc_eval_sds(dsets, dec_model, seg_model,
                                            detector, anchors, device,
                                            args, offline,
                                            ov_thresh=0.7)
        # aps += [ap]
        print('AP@0.7 for {} = {:.2f}, overlap = {:.4f}'.format(cls, ap_07 * 100, temp_overlaps))
    # print('Mean AP@0.7 = {:.2f}'.format(np.mean(aps) * 100))
    return ap_05, ap_07

