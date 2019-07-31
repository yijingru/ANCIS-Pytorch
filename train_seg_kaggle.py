import argparse
import torch.optim as optim
from torch.optim import lr_scheduler

from seg_utils import *
from dec_utils import *
from seg_utils import seg_transforms, seg_dataset_kaggle, seg_eval_kaggle

from models import dec_net_seg, seg_net
import cv2
import os

parser = argparse.ArgumentParser(description='Detection Training (MultiGPU)')
parser.add_argument('--trainDir', default="/home/grace/PycharmProjects/Datasets/kaggle/train", type=str, help='training image directory')
parser.add_argument('--valDir', default="/home/grace/PycharmProjects/Datasets/kaggle/val", type=str, help='validation image directory')
parser.add_argument('--annoDir', default="data/root/mask", type=str, help='annotation image directory')
parser.add_argument('--imgSuffix', default='.png', type=str, help='suffix of the input images')
parser.add_argument('--annoSuffix', default='.png', type=str, help='suffix of the annotation images')
parser.add_argument('--batch_size', default=8, type=int, help='batch size')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--init_lr', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--num_epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--weightDst', default='seg_weights', type=str, help='weight save folder')
parser.add_argument('--img_height', default=512, type=int, help='img height')
parser.add_argument('--img_width', default=512, type=int, help='img width')
parser.add_argument('--num_classes', default=2, type=int, help='dataset classes')
parser.add_argument('--top_k', default=500, type=int, help='the number of detections to keep')
parser.add_argument('--conf_thresh', default=0.3, type=float, help='confidence threshold')
parser.add_argument('--nms_thresh', default=0.3, type=float, help='nms threshold')
parser.add_argument('--seg_thresh', default=0.5, type=float, help='segmentation threshold')
parser.add_argument('--vis', default=False, type=bool, help='visualize augmented training datasets')
parser.add_argument('--dec_weights', default="dec_weights/kaggle/end_model.pth", type=str, help='detection weights')


def collater(data):
    imgs = []
    bboxes = []
    labels = []
    masks = []
    for sample in data:
        imgs.append(sample[0])
        bboxes.append(sample[1])
        labels.append(sample[2])
        masks.append(sample[3])
    return torch.stack(imgs,0), bboxes, labels, masks


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


def train(args):
    if not os.path.exists(args.weightDst):
        os.mkdir(args.weightDst)

    #-----------------load detection model -------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dec_model = dec_net_seg.resnetssd50(pretrained=False, num_classes=args.num_classes)
    dec_model = load_dec_weights(dec_model, args.dec_weights)
    dec_model = dec_model.to(device)
    #-------------------------------------------------------------------
    dec_model.eval()        # detector set to 'evaluation' mode
    for param in dec_model.parameters():
        param.requires_grad = False
    #-----------------load segmentation model -------------------------
    seg_model =  seg_net.SEG_NET(num_classes=args.num_classes)
    seg_model= seg_model.to(device)
    ##--------------------------------------------------------------
    data_transforms = {
        'train': seg_transforms.Compose([seg_transforms.ConvertImgFloat(),
                                         seg_transforms.PhotometricDistort(),
                                         seg_transforms.Expand(),
                                         seg_transforms.RandomSampleCrop(),
                                         seg_transforms.RandomMirror_w(),
                                         seg_transforms.RandomMirror_h(),
                                         seg_transforms.Resize(args.img_height, args.img_width),
                                         seg_transforms.ToTensor()]),

        'val': seg_transforms.Compose([seg_transforms.ConvertImgFloat(),
                                       seg_transforms.Resize(args.img_height, args.img_width),
                                       seg_transforms.ToTensor()])
    }


    dsets = {'train': seg_dataset_kaggle.NucleiCell(args.trainDir, args.annoDir, data_transforms['train'],
                                 imgSuffix=args.imgSuffix, annoSuffix=args.annoSuffix),
             'val': seg_dataset_kaggle.NucleiCell(args.valDir, args.annoDir, data_transforms['val'],
                                 imgSuffix=args.imgSuffix, annoSuffix=args.annoSuffix)}

    dataloader = torch.utils.data.DataLoader(dsets['train'],
                                             batch_size = args.batch_size,
                                             shuffle = True,
                                             num_workers = args.num_workers,
                                             collate_fn = collater,
                                             pin_memory = True)



    optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, seg_model.parameters()), lr=args.init_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)
    criterion = SEG_loss(height=args.img_height, width=args.img_width)


    if args.vis:
        cv2.namedWindow('img')
        for idx in range(len(dsets['train'])):
            img, bboxes, labels, masks = dsets['train'].__getitem__(idx)
            img = img.numpy().transpose(1, 2, 0).copy()*255
            print(img.shape)
            bboxes = bboxes.numpy()
            labels = labels.numpy()
            masks = masks.numpy()
            for idx in range(bboxes.shape[0]):
                y1, x1, y2, x2 = bboxes[idx,:]
                y1 = int(y1)
                x1 = int(x1)
                y2 = int(y2)
                x2 = int(x2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2, lineType=1)
                mask = masks[idx, :, :]
                img = map_mask_to_image(mask, img, color=np.random.rand(3))
            cv2.imshow('img', img)
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
                seg_model.train()
                running_loss = 0.0
                for inputs, bboxes, labels, masks in dataloader:
                    inputs = inputs.to(device)
                    with torch.no_grad():
                        locs, conf, feat_seg = dec_model(inputs)
                        detections = detector(locs, conf, anchors)

                    optimizer.zero_grad()
                    with torch.enable_grad():
                        outputs = seg_model(detections, feat_seg)
                        loss = criterion(outputs, bboxes, labels, masks)
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                epoch_loss = running_loss / len(dsets[phase])

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                train_loss_dict.append(epoch_loss)
                np.savetxt('train_loss.txt', train_loss_dict, fmt='%.6f')
                if epoch % 5 == 0:
                    torch.save(seg_model.state_dict(),
                               os.path.join(args.weightDst, '{:d}_{:.4f}_model.pth'.format(epoch, epoch_loss)))
                torch.save(seg_model.state_dict(), os.path.join(args.weightDst, 'end_model.pth'))

            else:
                if epoch % 9 == 0:
                    seg_model.eval()   # Set model to evaluate mode
                    ap_05, ap_07 = seg_eval_kaggle.do_python_eval(dsets=dsets[phase], dec_model=dec_model, seg_model=seg_model,
                                                           detector=detector, anchors=anchors, device=device,
                                                           args=args, offline=False)
                    # print('ap05:{:.4f}, ap07:{:.4f}'.format(ap05, ap07))
                    ap05_dict.append(ap_05)
                    np.savetxt('ap_05.txt', ap05_dict, fmt='%.6f')
                    ap07_dict.append(ap_07)
                    np.savetxt('ap_07.txt', ap07_dict, fmt='%.6f')

    print('Finish')


if __name__ == '__main__':
    args = parser.parse_args()
    train(args)