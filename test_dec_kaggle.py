import argparse

from dec_utils import *
from models import dec_net
from dec_utils import dec_transforms
import cv2
from dec_utils import dec_transforms, dec_eval, dec_dataset_kaggle


parser = argparse.ArgumentParser(description='Detection Training (MultiGPU)')
parser.add_argument('--testDir', default="/home/grace/PycharmProjects/DataSets/kaggle/test", type=str, help='training image directory')
parser.add_argument('--annoDir', default="data/root/mask", type=str, help='annotation image directory')
parser.add_argument('--imgSuffix', default='.png', type=str, help='suffix of the input images')
parser.add_argument('--annoSuffix', default='.png', type=str, help='suffix of the annotation images')
parser.add_argument('--img_height', default=512, type=int, help='img height')
parser.add_argument('--img_width', default=512, type=int, help='img width')
parser.add_argument('--num_classes', default=2, type=int, help='dataset classes')
parser.add_argument('--top_k', default=500, type=int, help='the number of detections to keep')
parser.add_argument('--conf_thresh', default=0.3, type=float, help='confidence threshold')
parser.add_argument('--nms_thresh', default=0.3, type=float, help='nms threshold')
parser.add_argument('--resume', default="dec_weights/kaggle/end_model.pth", type=str, help='resume weights directory')


def test(args):

    data_transforms = dec_transforms.Compose([dec_transforms.ConvertImgFloat(),
                                       dec_transforms.Resize(args.img_height, args.img_width),
                                       dec_transforms.ToTensor()])

    dsets = dec_dataset_kaggle.NucleiCell(args.testDir, args.annoDir, data_transforms,
                       imgSuffix=args.imgSuffix, annoSuffix=args.annoSuffix)


    model = dec_net.resnetssd50(pretrained=True, num_classes=args.num_classes)
    print('Resuming training weights from {} ...'.format(args.resume))
    pretrained_dict = torch.load(args.resume)
    model_dict = model.state_dict()
    trained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
    model_dict.update(trained_dict)
    model.load_state_dict(model_dict)


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
    cv2.namedWindow('img')
    for img_idx in range(len(dsets)):
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
            dets = dets.cpu().numpy()
            for i in range(dets.shape[0]):
                box = dets[i,1:]
                score = dets[i,0]
                y1,x1,y2,x2 = box
                y1 = float(y1)/args.img_height
                x1 = float(x1)/args.img_width
                y2 = float(y2)/args.img_height
                x2 = float(x2)/args.img_width
                y1 = int(float(y1)*h)
                x1 = int(float(x1)*w)
                y2 = int(float(y2)*h)
                x2 = int(float(x2)*w)
                cv2.rectangle(ori_img, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
                cv2.putText(ori_img, "%.2f"%score, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 255))
        cv2.imshow('img', ori_img)
        k = cv2.waitKey(0)
        if k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            exit()
    cv2.destroyAllWindows()
    exit()


if __name__ == '__main__':
    args = parser.parse_args()
    test(args)