import os
import yaml
import math
import time
import argparse
import logging
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter


import torch
import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.backends.cudnn
from torch.cuda import amp



from models.YOLOP import get_net
from test import AverageMeter, test
from utils.autoanchor import check_anchors
from utils.loss import get_loss
from utils.datasets import create_dataloader
from utils.general import colorstr, set_logging, increment_path, write_log, val_tensorboard
from utils.metrics import fitness
from utils.torch_utils import select_device


SEG_ONLY = False           # Only train two segmentation branchs
DET_ONLY = False           # Only train detection branch
ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
DRIVABLE_ONLY = False      # Only train da_segmentation task
LANE_ONLY = False          # Only train ll_segmentation task
DET_ONLY = False  

logger = logging.getLogger(__name__)



def main(args, hyp, device, writer):
    logger.info(colorstr('hyperparameter: ') + ', '\
                                .join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, maxEpochs = Path(args.save_dir), args.epochs
    begin_epoch = 0 

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'

        # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)
  


    # build up model
    print("begin to build up model...")
    # model = Model(args.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
    model = get_net(args.cfg).to(device)

    # loss function 
    criterion = get_loss(hyp, device)


    # Optimizer
    if hyp['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),lr=hyp['lr0'],
                                momentum=hyp['momentum'], weight_decay=hyp['wd'],
                                nesterov=hyp['nesterov'])
    elif hyp['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),lr=hyp['lr0'],
                                                betas=(hyp['momentum'], 0.999))

    print("finish build model")



    # Data loading
    print("begin to load data")
    normalize = {'mean':[0.485, 0.456, 0.406], 
                 'std':[0.229, 0.224, 0.225]}
    
    train_loader, train_dataset = create_dataloader(args, hyp, \
                                    args.train_batch_size, normalize)
    num_batch = len(train_loader)
    
    valid_loader, valid_dataset = create_dataloader(args, hyp, 
                                                args.test_batch_size, normalize,  
                                                is_train=False, shuffle=False)

    print('load data finished')
    



    # AUTOANCHOR
    if args.need_autoanchor:
        logger.info("begin check anchors")
        check_anchors(train_dataset, model=model, thr=hyp['anchor_threshold'], 
                                                    imgsz=min(args.img_size))
    else:
        logger.info("anchors loaded successfully")
        det = model.model[model.detector_index]
        logger.info(str(det.anchors))


    lf = lambda x: ((1 + math.cos(x * math.pi / maxEpochs)) / 2) * \
                   (1 - hyp['lrf']) + hyp['lrf']  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # assign model params
    model.gr = 1.0
    model.nc = 1

    # training
    num_warmup = max(round(hyp['warmup_epochs'] * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    print('=> start training...')
    global_steps = 0
    for epoch in range(begin_epoch, maxEpochs+1):

        model.train()
        start = time.time()
        for i, (input, target, paths, shapes) in enumerate(train_loader):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            num_iter = i + num_batch * (epoch - 1)

            if num_iter < num_warmup:
                # warm up
                lf = lambda x: ((1 + math.cos(x * math.pi / maxEpochs)) / 2)* \
                            (1 - hyp['lrf']) + hyp['lrf']  # cosine
                xi = [0, num_warmup]
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  
                # # iou loss ratio (obj_loss = 1.0 or iou)
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, 
                    # all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(num_iter, xi, [hyp['warmup_biase_lr'] \
                                        if j == 2 else 0.0, x['initial_lr'] *\
                                                                 lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iter, xi, 
                                                [hyp['warmup_momentum'], 
                                                    hyp['momentum']])
            

            data_time.update(time.time() - start)
            input = input.to(device, non_blocking=True)
            target = [gt.to(device) for gt in target]

            # Forward
            with amp.autocast(enabled=device.type != 'cpu'):
                outputs = model(input)
                total_loss, head_losses = criterion(outputs, target, shapes,model)

            # compute gradient and do update step
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # measure accuracy and record loss
            losses.update(total_loss.item(), input.size(0))

            # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
            #                                  target.detach().cpu().numpy())
            # acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - start)
            if i % 10 == 0:
                msg = f'Epoch: [{epoch}][{i}/{len(train_loader)}]   '+\
                        f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)    '+\
                        f'peed {input.size(0)/batch_time.val:.1f} samples/s     '+\
                        f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)      '+\
                        f'Loss {losses.val:.5f} ({losses.avg:.5f})'
                logger.info(msg)
                # Write 
                write_log(results_file, msg)
                writer.add_scalar('train_loss', losses.val, global_steps)
                global_steps += 1


        lr_scheduler.step()

        # evaluate on validation set
        if (epoch > args.val_start and (epoch % args.val_freq == 0 
                                                    or epoch == maxEpochs)):
            # print('validate')
            da_segment_result, ll_segment_result, detect_result, total_loss, maps, t= test(
                epoch, args, hyp, valid_loader, model, criterion,
                save_dir,results_file, logger, device, 
            )

            fi = fitness(np.array(detect_result).reshape(1, -1))  #目标检测评价指标

            # validation result tensorboard
            val_tensorboard(writer, global_steps-1, da_segment_result, 
                            ll_segment_result, detect_result, total_loss, maps, t)

            # if perf_indicator >= best_perf:
            #     best_perf = perf_indicator
            #     best_model = True
            # else:
            #     best_model = False

            # save checkpoint model and best model
            
            savepath = wdir / f'epoch-{epoch}.pth'
            logger.info('=> saving checkpoint to {}'.format(savepath))
            ckpt = {
                'epoch': epoch,
                'model': args.name,
                'state_dict':  model.state_dict(),
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }
            torch.save(ckpt, savepath)
            del ckpt

    torch.cuda.empty_cache()
    return



    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, 
                            default='lib/data/hyp.scratch.yolop.yaml', 
                            help='hyperparameter path')
                            # yolop_backbone
    parser.add_argument('--cfg', type=str, default='cfg/yolop.yaml', 
                                            help='model.yaml path')
    parser.add_argument('--logDir', type=str, default='runs/train',
                            help='log directory')
    parser.add_argument('--saveJson', type=bool, default=False)
    parser.add_argument('--saveTxt', type=bool, default=False)
    parser.add_argument('--allplot', type=bool, default=False)
    parser.add_argument('--auto_resume', type=bool, default=False,
                            help='Resume from the last training interrupt')
    parser.add_argument('--need_autoanchor', type=bool, default=False,
                            help='Re-select the prior anchor(k-means) \
                                    When training from scratch (epoch=0), \
                                    set it to be true!')
    parser.add_argument('--device', default='', 
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=30, 
                            help='total batch size for all GPUs')
    parser.add_argument('--test_batch_size', type=int, default=40, 
                            help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0, 
                            help='maximum number of dataloader workers')
    parser.add_argument('--name', default='exp', 
                            help='save to project/name')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                            help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, 
                            help='IOU threshold for NMS')
    parser.add_argument('--num_seg_class', type=int, default=2)
    parser.add_argument('--val_start', type=int, default=20, 
                            help='start do validation')
    parser.add_argument('--val_freq', type=int, default=5, 
                            help='How many epochs do one time validation')
    # dataset   BDD100k_10k
    parser.add_argument('--dataset', type=str, default='BddDataset', 
                            help='save to dataset name')
    parser.add_argument('--dataRoot', type=str, 
                    default='F:/dataset/BDD100k_10k/bdd100k_images_10k/bdd100k/images/10k', 
                            help='the path of images folder')
    parser.add_argument('--labelRoot', type=str, 
                    default='F:/dataset/BDD100k_10k/labels/10k', 
                            help='the path of det_annotations folder')
    parser.add_argument('--maskRoot', type=str, 
                    default='F:/dataset/BDD100k_10k/labels/bdd_seg_gt', 
                            help='the path of da_seg_annotations folder')
    parser.add_argument('--laneRoot', type=str, 
                    default='F:/dataset/BDD100k_10k/labels/bdd_lane_gt', 
                            help='the path of ll_seg_annotations folder')
    parser.add_argument('--trainSet', type=str, default='train', 
                            help='IOU threshold for NMS')
    parser.add_argument('--testSet', type=str, default='val', 
                            help='IOU threshold for NMS')
    parser.add_argument('--dataFormat', type=str, default='jpg', 
                            help='Data Format')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], 
                            help='[train, test] image sizes')
    parser.add_argument('--org_img_size', nargs='+', type=int, default=[720, 1280], 
                            help='[train, test] original image sizes')

    parser.add_argument('--pretrain', type=str, default='', 
                            help='all branch pretrain')
    parser.add_argument('--pretrain_det', type=str, default='', 
                            help='detection branch pretrain')
   
    # Cudnn related params
    parser.add_argument('--cudnn_benchmark', type=bool, default=True,  
                                help='Use GPUs to speed up network training')
    parser.add_argument('--cudnn_deterministic', type=bool, default=False, 
                                help='only use deterministic convolution algorithms')
    parser.add_argument('--cudnn_enabled', type=bool, default=True,  
                                help='controls whether cuDNN is enabled')
    return parser.parse_args()

if __name__ == '__main__':
    # Set DDP variables
    args = parse_args()
    set_logging()

    # cudnn related setting
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    device = select_device(args.device, batch_size=args.train_batch_size)


    # Hyperparameter
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    hyp.update({'seg_only':SEG_ONLY,'det_only':DET_ONLY,
                'enc_seg_only':ENC_SEG_ONLY, 'enc_det_only':ENC_DET_ONLY,
                'drivable_only':DRIVABLE_ONLY, 'lane_only':LANE_ONLY,
                'det_only':DET_ONLY})


    args.save_dir = increment_path(Path(args.logDir)/ args.dataset) 

    # Train
    logger.info(args)
    prefix = colorstr('tensorboard: ')
    logger.info(f"{prefix}Start with 'tensorboard --logdir {args.logDir}'"+\
                                        ", view at http://localhost:6006/")
    writer = SummaryWriter(args.save_dir)  # Tensorboard
    main(args, hyp, device, writer)
 


