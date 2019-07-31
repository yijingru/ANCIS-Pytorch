import argparse

from dec_utils import *
from models import dec_net
from dec_utils import dec_transforms, dec_eval, dec_dataset_kaggle
import os
import cv2
import pickle

parser = argparse.ArgumentParser(description='Detection Training (MultiGPU)')
parser.add_argument('--testDir', default="/home/grace/PycharmProjects/DataSets/kaggle/test", type=str, help='training image directory')
parser.add_argument('--annoDir', default="data/root/mask", type=str, help='annotation image directory')
parser.add_argument('--imgSuffix', default='.png', type=str, help='suffix of the input images')
parser.add_argument('--annoSuffix', default='.png', type=str, help='suffix of the annotation images')
parser.add_argument('--img_height', default=512, type=int, help='img height')
parser.add_argument('--img_width', default=512, type=int, help='img width')
parser.add_argument('--num_classes', default=2, type=int, help='dataset classes')
parser.add_argument('--top_k', default=200, type=int, help='the number of detections to keep')
parser.add_argument('--conf_thresh', default=0.3, type=float, help='confidence threshold')
parser.add_argument('--nms_thresh', default=0.3, type=float, help='nms threshold')
parser.add_argument('--resume', default="dec_weights/kaggle/end_model.pth", type=str, help='resume weights directory')
parser.add_argument('--cacheDir', default="cache", type=str, help='resume weights directory')

def load_dec_weights(dec_model, dec_weights):
    print('Resuming detection weights from {} ...'.format(dec_weights))
    dec_dict = torch.load(dec_weights)
    dec_dict_update = {}
    for k in dec_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            dec_dict_update[k[7:]] = dec_dict[k]
        else:
            dec_dict_update[k] = dec_dict[k]
    dec_model.load_state_dict(dec_dict_update, strict=True)
    return dec_model

def evaluation(args):

    data_transforms = dec_transforms.Compose([dec_transforms.ConvertImgFloat(),
                                              dec_transforms.Resize(args.img_height, args.img_width),
                                              dec_transforms.ToTensor()])

    dsets = dec_dataset_kaggle.NucleiCell(args.testDir, args.annoDir, data_transforms,
                       imgSuffix=args.imgSuffix, annoSuffix=args.annoSuffix)


    model = dec_net.resnetssd50(pretrained=True, num_classes=args.num_classes)
    model = load_dec_weights(model, args.resume)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    detector = Detect(num_classes=args.num_classes,
                      top_k=args.top_k,
                      conf_thresh=args.conf_thresh,
                      nms_thresh=args.nms_thresh,
                      variance=[0.1, 0.2])
    anchorGen = Anchors(args.img_height, args.img_width)
    anchors = anchorGen.forward()

    det_file = os.path.join(args.cacheDir, 'detections.pkl')
    if not os.path.exists(args.cacheDir):
        os.mkdir(args.cacheDir)

    all_boxes = [[[] for _ in range(len(dsets))] for _ in range(args.num_classes)]
    for img_idx in range(len(dsets)):
        print('loading {}/{} image'.format(img_idx, len(dsets)))
        ori_img = dsets.load_img(img_idx)
        h,w,c = ori_img.shape
        inputs, gt_bboxes, gt_labels = dsets.__getitem__(img_idx)  # [3, 512, 640], [3, 4], [3, 1]
        inputs = inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            locs, conf = model(inputs)
        detections = detector(locs, conf, anchors)
        for cls_idx in range(1, detections.size(1)):
            dets = detections[0, cls_idx, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.shape[0] == 0:
                continue
            pred_boxes = dets[:, 1:].cpu().numpy().astype(np.float32)
            pred_score = dets[:, 0].cpu().numpy()

            pred_boxes[:,0] /= args.img_height
            pred_boxes[:,1] /= args.img_width
            pred_boxes[:,2] /= args.img_height
            pred_boxes[:,3] /= args.img_width
            pred_boxes[:,0] *= h
            pred_boxes[:,1] *= w
            pred_boxes[:,2] *= h
            pred_boxes[:,3] *= w

            cls_dets = np.hstack((pred_boxes, pred_score[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[cls_idx][img_idx] = cls_dets

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        f.close()

    for cls_ind, cls in enumerate(dsets.labelmap):
        filename = dec_eval.get_voc_results_file_template('test', cls, args.cacheDir)
        with open(filename, 'wt') as f:
            for im_ind, index in enumerate(dsets.img_files):
                dets = all_boxes[cls_ind + 1][im_ind]
                if dets == []:
                    continue
                for k in range(dets.shape[0]):
                    # format: [img_file  confidence, y1, x1, y2, x2] save to call for multiple times
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.format(index,
                                                                               dets[k, -1],
                                                                               dets[k, 0],
                                                                               dets[k, 1],
                                                                               dets[k, 2],
                                                                               dets[k, 3]))
    ap05, ap07 = dec_eval.do_python_eval(dsets=dsets,
                                         output_dir=args.cacheDir,
                                         offline=True,
                                         use_07=True)


if __name__ == '__main__':
    args = parser.parse_args()
    evaluation(args)