import cv2
import yaml
import json
import random
import logging
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path

import torch


from models.model import build_model
from utils.loss import MultiHeadLoss
from models.model import build_model
from utils.datasets import create_dataloader
from utils.torch_utils import select_device, time_synchronized
from utils.plot import plot_one_box,show_seg_result
from utils.metrics import ConfusionMatrix, SegmentationMetric, ap_per_class,\
                            output_to_target, ap_per_class
from utils.general import colorstr, increment_path, write_log,non_max_suppression,\
                        check_img_size,scale_coords,xyxy2xywh,xywh2xyxy,\
                        box_iou, data_color, AverageMeter


logger = logging.getLogger(__name__)

def test(epoch, args, hyp, val_loader, model, criterion, output_dir,
              results_file, target_name, Lane_color, DriveArea_color, logger=None, 
                                                        device='cpu'):
    """
    validata

    Inputs:
    - cfg: configurations 
    - train_loader: loder for data
    - model: 
    - criterion: (function) calculate all the loss, return 
    - writer_dict: 

    Return:
    None
    """
    # setting
    max_stride = 32
    weights = None

    save_dir = output_dir / 'visualization'
    save_dir.mkdir(parents=True, exist_ok=True)
    save_dir = str(save_dir)

     #imgsz is multiple of max_stride
    _, imgsz = [check_img_size(x, s=max_stride) for x in args.img_size]
    batch_size = args.test_batch_size
    save_hybrid=False

    #iou vector for mAP@0.5:0.95
    iouv = torch.linspace(0.5,0.95,10).to(device)    
    niou = iouv.numel()

    Det_nc, driveArea_nc, Lane_nc = hyp['nc'][0], hyp['nc'][1], hyp['nc'][2]
    seen =  0 
    confusion_matrix = ConfusionMatrix(Det_nc) #detector confusion matrix
    da_metric = SegmentationMetric(driveArea_nc) #drive area segment confusion matrix    
    ll_metric = SegmentationMetric(Lane_nc) #lane line segment confusion matrix

    Det_name, DriveArea_name, Lane_name,  = target_name[0], list(target_name[1].keys()), list(target_name[2].keys())
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in Det_name]

    
    p, r, f1, mp, mr, map50, map, t_inf, t_nms = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    
    losses = AverageMeter()

    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()

    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    T_inf = AverageMeter()
    T_nms = AverageMeter()

    # switch to train mode
    model.eval()
    jdict, stats, ap, ap_class = [], [], [], []

    for batch_i, (img, target, paths, shapes) in enumerate(tqdm(val_loader)):

        img = img.to(device, non_blocking=True)
        target = [gt.to(device) for gt in target]
        nb, _, height, width = img.shape    #batch size, channel, height, width

        with torch.no_grad():
            pad_w, pad_h = shapes[0][1][1]
            pad_w, pad_h = int(pad_w), int(pad_h)
            ratio = min(shapes[0][1][0])

            t = time_synchronized()
            det_out, da_seg_out, ll_seg_out= model(img)
            t_inf = time_synchronized() - t
            if batch_i > 0:
                T_inf.update(t_inf/img.size(0),img.size(0))

            inf_out, train_out = det_out
            
            total_loss, head_losses = criterion((train_out,da_seg_out, ll_seg_out), target, shapes,model)   #Compute loss
            losses.update(total_loss.item(), img.size(0))

            #NMS
            t = time_synchronized()
            target[0][:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [target[0][target[0][:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            output = non_max_suppression(inf_out, conf_thres=hyp['nms_conf_threshold'], iou_thres=hyp['nms_iou_threshold'], labels=lb)
            t_nms = time_synchronized() - t
            if batch_i > 0:
                T_nms.update(t_nms/img.size(0),img.size(0))
            if batch_i == 0:
                for i in range(batch_size):
                    img_test = cv2.imread(paths[i])
                    img_GT = img_test.copy()

                    da_seg_mask = da_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                    da_seg_mask = torch.nn.functional.interpolate(da_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                    _, da_seg_mask = torch.max(da_seg_mask, 1)
                    da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                    show_seg_result(img_test, da_seg_mask, palette=DriveArea_color)

                    ll_seg_mask = ll_seg_out[i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                    ll_seg_mask = torch.nn.functional.interpolate(ll_seg_mask, scale_factor=int(1/ratio), mode='bilinear')
                    _, ll_seg_mask = torch.max(ll_seg_mask, 1)
                    ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
                    show_seg_result(img_test, ll_seg_mask, palette=Lane_color)

                    det = output[i].clone()
                    if len(det):
                        det[:,:4] = scale_coords(img[i].shape[1:],det[:,:4],img_test.shape).round()
                    for *xyxy,conf,cls in reversed(det):
                        #print(cls)
                        label_det_pred = f'{Det_name[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img_test , label=label_det_pred, color=colors[int(cls)], line_thickness=2)
                    cv2.imwrite(save_dir+"/batch_{}_{}_pred.png".format(epoch,i),img_test)




                    da_gt_mask = target[1][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                    da_gt_mask = torch.nn.functional.interpolate(da_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                    _, da_gt_mask = torch.max(da_gt_mask, 1)
                    da_gt_mask = da_gt_mask.int().squeeze().cpu().numpy()
                    show_seg_result(img_GT, da_gt_mask, palette=DriveArea_color)

                    ll_gt_mask = target[2][i][:, pad_h:height-pad_h, pad_w:width-pad_w].unsqueeze(0)
                    ll_gt_mask = torch.nn.functional.interpolate(ll_gt_mask, scale_factor=int(1/ratio), mode='bilinear')
                    _, ll_gt_mask = torch.max(ll_gt_mask, 1)
                    ll_gt_mask = ll_gt_mask.int().squeeze().cpu().numpy()
                    show_seg_result(img_GT, ll_gt_mask, palette=Lane_color)
                    labels = target[0][target[0][:, 0] == i, 1:]
                    # print(labels)
                    labels[:,1:5]=xywh2xyxy(labels[:,1:5])
                    if len(labels):
                        labels[:,1:5]=scale_coords(img[i].shape[1:],labels[:,1:5],img_GT.shape).round()
                    for cls,x1,y1,x2,y2 in labels:
                        #print(Det_name)
                        #print(cls)
                        label_det_gt = f'{Det_name[int(cls)]}'
                        xyxy = (x1,y1,x2,y2)
                        plot_one_box(xyxy, img_GT , label=label_det_gt, color=colors[int(cls)], line_thickness=2)
                    cv2.imwrite(save_dir+"/batch_{}_{}_gt.png".format(epoch,i),img_GT)


        #driving area segment evaluation
        _,da_predict=torch.max(da_seg_out, 1)
        _,da_gt=torch.max(target[1], 1)
        da_predict = da_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
        da_gt = da_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

        da_metric.reset()
        da_metric.addBatch(da_predict.cpu(), da_gt.cpu())
        da_acc = da_metric.pixelAccuracy()
        da_IoU, da_mIoU = da_metric.IntersectionOverUnion()

        da_acc_seg.update(da_acc,img.size(0))
        da_IoU_seg.update(da_IoU,img.size(0))
        da_mIoU_seg.update(da_mIoU,img.size(0))

        #lane line segment evaluation
        _,ll_predict=torch.max(ll_seg_out, 1)
        _,ll_gt=torch.max(target[2], 1)
        ll_predict = ll_predict[:, pad_h:height-pad_h, pad_w:width-pad_w]
        ll_gt = ll_gt[:, pad_h:height-pad_h, pad_w:width-pad_w]

        ll_metric.reset()
        ll_metric.addBatch(ll_predict.cpu(), ll_gt.cpu())
        ll_acc = ll_metric.lineAccuracy()
        ll_IoU, ll_mIoU = ll_metric.IntersectionOverUnion()

        ll_acc_seg.update(ll_acc,img.size(0))
        ll_IoU_seg.update(ll_IoU,img.size(0))
        ll_mIoU_seg.update(ll_mIoU,img.size(0))

        # Statistics per image
        # output([xyxy,conf,cls])
        # target[0] ([img_id,cls,xyxy])
        for si, pred in enumerate(output):
            labels = target[0][target[0][:, 0] == si, 1:]     #all object in one image 
            nl = len(labels)    # num of object
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                confusion_matrix.process_batch(pred, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):                    
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        # n*m  n:pred  m:label
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))


    # Compute statistics
    # stats : [[all_img_correct]...[all_img_tcls]]
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy  zip(*) :unzip

    map70 = None
    map75 = None
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=False, save_dir=save_dir, names=Det_name)
        ap50, ap70, ap75,ap = ap[:, 0], ap[:,4], ap[:,5],ap.mean(1)  # [P, R, AP@0.5, AP@0.5:0.95]
        mp, mr, map50, map70, map75, map = p.mean(), r.mean(), ap50.mean(), ap70.mean(),ap75.mean(),ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=Det_nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t_inf, t_nms, t_inf + t_nms)) + (imgsz, imgsz, batch_size)  # tuple
    
    print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Plots
    confusion_matrix.plot(save_dir=save_dir, names=list(Det_name))
    # confusion_matrix.plot(save_dir=save_dir, names=list(Det_name.values()))


    model.float()  # for training
    maps = np.zeros(Det_nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    da_segment_result = (da_acc_seg.avg,da_IoU_seg.avg,da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg,ll_IoU_seg.avg,ll_mIoU_seg.avg)

    # print(da_segment_result)
    # print(ll_segment_result)
    detect_result = np.asarray([mp, mr, map50, map])
    t = [T_inf.avg, T_nms.avg]

    # Print results
    msg = f'Epoch: [{epoch}]    Loss({losses.avg:.3f})\nDetect:\n'

    if  Det_nc > 1 and len(stats):
        pf = '%20s' + '%13g' * 6  # print format
        msg += (('%20s' + '%13s' * 6) % ('class', 'Images', 'Labels', 'p', 'R', 'mAP@.5', 'mAP@.5:.95:')+'\n')
        msg += (pf % ('all', seen, nt.sum(), mp, mr, map50, map)+'\n')
        # Print results per class
        for i, c in enumerate(ap_class):
            msg += (pf % (Det_name[c], seen, nt[c], p[i], r[i], ap50[i], ap[i])+'\n')
    
    if  driveArea_nc > 1:
        pf = '%20s' + '%13g' # print format
        msg += 'Driving area Segment:\n'
        msg += (('%20s' + '%13s') % ('class', 'IoU')+'\n')
        for i, iou in enumerate(da_IoU_seg.avg):
            msg += (pf % (DriveArea_name[i], iou)+'\n')

    if  Lane_nc > 1:
        pf = '%20s' + '%13g' # print format
        msg += (('%20s' + '%13s') % ('class', 'IoU')+'\n')
        msg += 'Lane line Segment:\n'
        for i, iou in enumerate(ll_IoU_seg.avg):
            msg += (pf % (Lane_name[i], iou)+'\n')    

    msg += f'\n\n \
            Driving area Segment:    Acc({da_segment_result[0]:.3f})    mIOU({da_segment_result[2]:.3f})\n\
            Lane line Segment:       Acc({ll_segment_result[0]:.3f})    mIOU({ll_segment_result[2]:.3f})\n\
            Detect:    P({detect_result[0]:.3f})      R({detect_result[1]:.3f})    mAP@0.5({detect_result[2]:.3f})    mAP@0.5:0.95({detect_result[3]:.3f})\n\
            Time: inference({t[0]:.4f}s/frame)  nms({t[1]:.4f}s/frame)'
    if(logger):
        logger.info(msg)
    write_log(results_file, msg)

    
    
    return da_segment_result, ll_segment_result, detect_result, losses.avg, maps, t
        


def parse_args():
    parser = argparse.ArgumentParser(description='Test Multitask network')
    parser.add_argument('--hyp', type=str, default='hyp/hyp.scratch.yolop.yaml', 
                            help='hyperparameter path')
    parser.add_argument('--cfg', type=str, default='cfg/YOLOP_v7b3.yaml', 
                                                help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/single.yaml', 
                                            help='dataset yaml path')
    parser.add_argument('--logDir', type=str, default='runs/test',
                            help='log directory')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], 
                            help='[train, test] image sizes')
    parser.add_argument('--conf-thres', type=float, default=0.25, 
                                                help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, 
                                                    help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--weights', type=str, default='weights/last.pth', 
                                                        help='model.pth path(s)')
    parser.add_argument('--test_batch_size', type=int, default=5, 
                            help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0, 
                            help='maximum number of dataloader workers')
    # dataset
    parser.add_argument('--dataset', type=str, default='BddDataset', 
                            help='save to dataset name')

    return parser.parse_args()




if __name__ == '__main__':

    args = parse_args()

    device = select_device(args.device, batch_size=args.test_batch_size)


    # Hyperparameter
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    Det_class = data_dict['Det_names']
    DriveArea_class = data_dict['DriveArea_names']
    Lane_class = data_dict['Lane_names']
    hyp.update({'nc':[len(Det_class), len(DriveArea_class), len(Lane_class)]})
    logger.info(f"{colorstr('Det_class: ')}{Det_class}")
    logger.info(f"{colorstr('DriveArea_class: ')}{DriveArea_class}")
    logger.info(f"{colorstr('Lane_class: ')}{Lane_class}")
    target_name = (Det_class, DriveArea_class, Lane_class)
    Lane_color = data_color(Lane_class)
    DriveArea_color = data_color(DriveArea_class)


    # Directories
    args.save_dir = Path(increment_path(Path(args.logDir)/ args.dataset))  # increment run
    results_file = args.save_dir / 'results.txt'
    args.save_dir.mkdir(parents=True, exist_ok=True)


    # build up model
    #TODO anchor method
    print("begin to build up model...")
    anchors = None
    model = build_model(args.cfg, hyp['nc'], anchors).to(device)

    # loss function 
    criterion = MultiHeadLoss(hyp, device)

    # load weights
    model_dict = model.state_dict()
    checkpoint_file = args.weights
    print("=> loading checkpoint '{}'".format(checkpoint_file))
    checkpoint = torch.load(checkpoint_file)
    checkpoint_dict = checkpoint['state_dict']
    # checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
    model_dict.update(checkpoint_dict)
    model.load_state_dict(model_dict)
    print("=> loaded checkpoint '{}' ".format(checkpoint_file))

    model = model.to(device)
    model.gr = 1.0
    model.nc = hyp['nc'][0]
    print('bulid model finished')

    epoch = checkpoint['epoch'] #special for test
    # Save run settings

        # Data loading
    print("begin to load data")
    normalize = {'mean':[0.485, 0.456, 0.406], 
                 'std':[0.229, 0.224, 0.225]}
    valid_loader, valid_dataset = create_dataloader(args, hyp, data_dict, \
                            args.test_batch_size, normalize, is_train=False, \
                                                                shuffle=False)
    print('load data finished')

    with open(args.save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(args.save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)
    
    test(epoch, args, hyp, valid_loader, model, criterion,
                args.save_dir, results_file, target_name, Lane_color, DriveArea_color, 
                                                        device = device)

    print("test finish")