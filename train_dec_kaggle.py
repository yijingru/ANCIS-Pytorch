import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from dec_utils import *
from models import dec_net
from dec_utils import dec_transforms, dec_eval, dec_dataset_kaggle
import os
import cv2
import pickle


parser = argparse.ArgumentParser(description='Detection Training (MultiGPU)')
parser.add_argument('--trainDir', default="/home/grace/PycharmProjects/DataSets/kaggle/train", type=str, help='training image directory')
parser.add_argument('--valDir', default="/home/grace/PycharmProjects/DataSets/kaggle/val", type=str, help='validation image directory')
parser.add_argument('--annoDir', default="data/root/mask", type=str, help='annotation image directory')
parser.add_argument('--cacheDir', default="cache", type=str, help='cache save directory')
parser.add_argument('--imgSuffix', default='.png', type=str, help='suffix of the input images')
parser.add_argument('--annoSuffix', default='.png', type=str, help='suffix of the annotation images')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')
parser.add_argument('--multi_gpu', default=False, type=bool, help='use multi gpu')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=500, type=int, help='number of training epochs')
parser.add_argument('--decayEpoch', default=450, type=int, help='epoch to decay the learning rate')
parser.add_argument('--weightDst', default='dec_weights', type=str, help='weight save folder')
parser.add_argument('--img_height', default=512, type=int, help='img height')
parser.add_argument('--img_width', default=512, type=int, help='img width')
parser.add_argument('--num_classes', default=2, type=int, help='dataset classes')
parser.add_argument('--top_k', default=200, type=int, help='the number of detections to keep')
parser.add_argument('--conf_thresh', default=0.3, type=float, help='confidence threshold')
parser.add_argument('--nms_thresh', default=0.3, type=float, help='nms threshold')
parser.add_argument('--vis', default=False, type=bool, help='visualize augmented training datasets')


def collater(data):
    imgs = []
    bboxes = []
    labels = []
    for sample in data:
        imgs.append(sample[0])
        bboxes.append(sample[1])
        labels.append(sample[2])
    return torch.stack(imgs,0), bboxes, labels

def train(args):
    if not os.path.exists(args.weightDst):
        os.mkdir(args.weightDst)
    data_transforms = {
        'train': dec_transforms.Compose([dec_transforms.ConvertImgFloat(),
                                         dec_transforms.PhotometricDistort(),
                                         dec_transforms.Expand(),
                                         dec_transforms.RandomSampleCrop(),
                                         dec_transforms.RandomMirror_w(),
                                         dec_transforms.RandomMirror_h(),
                                         dec_transforms.Resize(args.img_height, args.img_width),
                                         dec_transforms.ToTensor()]),

        'val': dec_transforms.Compose([dec_transforms.ConvertImgFloat(),
                                       dec_transforms.Resize(args.img_height, args.img_width),
                                       dec_transforms.ToTensor()])
    }

    dsets = {'train': NucleiCell(args.trainDir, args.annoDir, data_transforms['train'],
                                 imgSuffix=args.imgSuffix, annoSuffix=args.annoSuffix),
             'val': NucleiCell(args.valDir, args.annoDir, data_transforms['val'],
                                 imgSuffix=args.imgSuffix, annoSuffix=args.annoSuffix)}

    dataloader = torch.utils.data.DataLoader(dsets['train'],
                                             batch_size = args.batch_size,
                                             shuffle = True,
                                             num_workers = args.num_workers,
                                             collate_fn = collater,
                                             pin_memory = True)

    model = dec_net.resnetssd50(pretrained=True, num_classes=args.num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=0.9)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[args.decayEpoch, args.num_epochs], gamma=0.1)
    criterion = DecLoss(img_height=args.img_height,
                        img_width= args.img_width,
                        num_classes=args.num_classes,
                        variances=[0.1, 0.2])

    if args.vis:
        cv2.namedWindow('img')
        for idx in range(len(dsets['train'])):
            img, bboxes, labels = dsets['train'].__getitem__(idx)
            img = img.numpy().transpose(1, 2, 0)*255
            bboxes = bboxes.numpy()
            labels = labels.numpy()
            for bbox in bboxes:
                y1, x1, y2, x2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, lineType=1)
            cv2.imshow('img', np.uint8(img))
            k = cv2.waitKey(0)
            if k & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                exit()
        cv2.destroyAllWindows()

    # for validation data -----------------------------------
    detector = Detect(num_classes=args.num_classes,
                      top_k=args.top_k,
                      conf_thresh=args.conf_thresh,
                      nms_thresh=args.nms_thresh,
                      variance=[0.1, 0.2])
    anchorGen = Anchors(args.img_height, args.img_width)
    anchors = anchorGen.forward()
    if not os.path.exists(args.cacheDir):
        os.mkdir(args.cacheDir)
    # --------------------------------------------------------
    train_loss_dict = []
    ap05_dict = []
    ap07_dict = []
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
                running_loss = 0.0
                for inputs, bboxes, labels in dataloader:
                    inputs = inputs.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss_locs, loss_conf = criterion(outputs, bboxes, labels)
                        loss = loss_locs + loss_conf
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dsets[phase])

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                train_loss_dict.append(epoch_loss)
                np.savetxt('train_loss.txt', train_loss_dict, fmt='%.6f')
                if epoch % 5 == 0:
                    torch.save(model.state_dict(),
                               os.path.join(args.weightDst, '{:d}_{:.4f}_model.pth'.format(epoch, epoch_loss)))
                torch.save(model.state_dict(), os.path.join(args.weightDst, 'end_model.pth'))

            else:
                model.eval()   # Set model to evaluate mode
                model.eval()   # Set model to evaluate mode
                det_file = os.path.join(args.cacheDir, 'detections.pkl')
                all_boxes = [[[] for _ in range(len(dsets['val']))] for _ in range(args.num_classes)]
                for img_idx in range(len(dsets['val'])):
                    ori_img = dsets['val'].load_img(img_idx)
                    h,w,c = ori_img.shape
                    inputs, gt_bboxes, gt_labels = dsets['val'].__getitem__(img_idx)  # [3, 512, 640], [3, 4], [3, 1]
                    # run model
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
                        pred_boxes = dets[:, 1:].cpu().numpy()
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

                for cls_ind, cls in enumerate(dsets['val'].labelmap):
                    filename = dec_eval.get_voc_results_file_template('test', cls, args.cacheDir)
                    with open(filename, 'wt') as f:
                        for im_ind, index in enumerate(dsets['val'].img_files):
                            dets = all_boxes[cls_ind+1][im_ind]
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
                ap05, ap07 = dec_eval.do_python_eval(dsets=dsets['val'],
                                                     output_dir=args.cacheDir,
                                                     offline=False,
                                                     use_07=True)
                print('ap05:{:.4f}, ap07:{:.4f}'.format(ap05, ap07))
                ap05_dict.append(ap05)
                np.savetxt('ap_05.txt', ap05_dict, fmt='%.6f')
                ap07_dict.append(ap07)
                np.savetxt('ap_07.txt', ap07_dict, fmt='%.6f')
    print('Finish')


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)