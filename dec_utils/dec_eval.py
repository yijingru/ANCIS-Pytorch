import os
import numpy as np
import pickle

def get_voc_results_file_template(image_set, cls, output_dir):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(output_dir, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def parse_rec(annotation):
    objects = []
    bboxes, labels = annotation
    for box in bboxes:
        obj_struct = {}
        obj_struct['name'] = 'cell'
        obj_struct['bbox'] = box
        objects.append(obj_struct)
    return objects


def voc_eval(filename,
             dsets,
             cachedir,
             classname,
             ovthresh=0.5,
             use_07_metric=True):

    if not os.path.exists(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # step1: read detections
    with open(filename, 'r') as f:
        lines = f.readlines()
    f.close()
    if any(lines) == 1:
        # step2: initialization predicted detections
        splitlines = [x.strip().split(' ') for x in lines]
        all_img_files = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]
        all_img_files = [all_img_files[x] for x in sorted_ind]
        # step3: initialization ground-truth
        if not os.path.isfile(cachefile):
            # load annotations
            recs = {}
            for i, img_file in enumerate(dsets.img_files):
                recs[img_file] = parse_rec(dsets.load_annotation(img_file))
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)
        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for img_file in dsets.img_files:
            R = [obj for obj in recs[img_file] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            det = [False] * len(R)
            npos = npos + len(R)
            class_recs[img_file] = {'bbox': bbox, 'det': det}
        # step4: go down dets and mark TPs and FPs
        nd = len(all_img_files)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            R = class_recs[all_img_files[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.shape[0] > 0:
                iymin = np.maximum(BBGT[:, 0], bb[0])
                ixmin = np.maximum(BBGT[:, 1], bb[1])
                iymax = np.minimum(BBGT[:, 2], bb[2])
                ixmax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                union = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                         (BBGT[:, 2] - BBGT[:, 0]) *
                         (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / union
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
            else:
                fp[d] = 1.

        # step5: compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap

def do_python_eval(dsets, output_dir, offline=True, use_07=True):
    # aps = []
    # The PASCAL VOC metric changed in 2010
    if offline:
        print('VOC07 metric? ' + ('Yes' if use_07 else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    cachedir = os.path.join(output_dir, 'annotations_cache')
    ap05 = -1
    ap07 = -1
    if offline:
        print("AP@0.5")
    for i, cls in enumerate(dsets.labelmap):
        if offline:
            print("i:{}, cls:{}".format(i,cls))
        filename = get_voc_results_file_template('test', cls, output_dir=output_dir)
        rec, prec, ap05 = voc_eval(filename,
                                 dsets,
                                 cachedir,
                                 cls,
                                 ovthresh=0.5,
                                 use_07_metric=use_07)
        if offline:
            print('AP for {} = {:.4f}'.format(cls, ap05))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap05}, f)

    if offline:
        print("AP@0.7")
    for i, cls in enumerate(dsets.labelmap):
        if offline:
            print("i:{}, cls:{}".format(i,cls))
        filename = get_voc_results_file_template('test', cls, output_dir=output_dir)
        rec, prec, ap07 = voc_eval(filename,
                                 dsets,
                                 cachedir,
                                 cls,
                                 ovthresh=0.7,
                                 use_07_metric=use_07)
        # aps += [ap]
        if offline:
            print('AP for {} = {:.4f}'.format(cls, ap07))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap07}, f)
    return ap05, ap07

