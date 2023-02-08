import os
import yaml
import math
import time
import argparse
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.utils.data
from torch.cuda import amp
import torch.backends.cudnn



from utils.loss import get_loss
from utils.metrics import fitness
from test import test
from utils.autoanchor import check_anchors
from utils.torch_utils import select_device
from utils.datasets import create_dataloader
from models.YOLOP import get_optimizer, Model
from utils.general import colorstr, set_logging, increment_path, write_log,\
                         val_tensorboard, data_color, AverageMeter




logger = logging.getLogger(__name__)



def main(args, hyp, device, writer):
    begin_epoch = 1
    global_steps = 0

    # Directories
    logger.info(colorstr('hyperparameter: ') + ', '\
                                .join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, maxEpochs = Path(args.save_dir), args.epochs
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'

  

    # Get class and class number
    with open(args.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    Det_class = data_dict['Det_names']
    Lane_class = data_dict['Lane_names']
    DriveArea_class = data_dict['DriveArea_names']
    hyp.update({'nc':[len(Det_class), len(Lane_class), len(DriveArea_class)]})
    logger.info(f"{colorstr('Det_class: ')}{Det_class}")
    logger.info(f"{colorstr('Lane_class: ')}{Lane_class}")
    logger.info(f"{colorstr('DriveArea_class: ')}{DriveArea_class}")
    Lane_color = data_color(Lane_class)
    DriveArea_color = data_color(DriveArea_class)
    # Save run settings(hyp, args)
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    # build up model
    print("begin to build up model...")
    # model = Model(args.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create

    #TODO anchor method
    anchors = None
    model = Model(args.cfg, hyp['nc'], anchors).to(device)

    # loss function 
    criterion = get_loss(hyp, device)

    # Optimizer
    optimizer = get_optimizer(hyp, model)                               

    # resume 
    if(args.resume):
        checkpoint = torch.load(args.resume)
        begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        global_steps = checkpoint['global_steps']+1
        msg = f'{colorstr("=> loaded checkpoint")} "{args.resume}"(epoch {begin_epoch})'
        logger.info(msg)
        write_log(results_file, msg)
    print("finish build model")



    # Data loading
    print("begin to load data")
    normalize = {'mean':[0.485, 0.456, 0.406], 
                 'std':[0.229, 0.224, 0.225]}
    
    train_loader, train_dataset = create_dataloader(args, hyp, data_dict, \
                                                args.train_batch_size, normalize)
    num_batch = len(train_loader)
    
    valid_loader, valid_dataset = create_dataloader(args, hyp, data_dict,\
                                                args.train_batch_size, normalize, \
                                                    is_train=False, shuffle=False)

    print('load data finished')
    



    # AUTOANCHOR
    if args.need_autoanchor:
        logger.info("begin check anchors")
        check_anchors(train_dataset, model=model, thr=hyp['anchor_threshold'], 
                                                    imgsz=min(args.img_size))
    else:
        logger.info("anchors loaded successfully")
        det = model.model[model.HeadOut[0]]
        logger.info(str(det.anchors))


    lf = lambda x: ((1 + math.cos(x * math.pi / maxEpochs)) / 2) * \
                   (1 - hyp['lrf']) + hyp['lrf']  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # # assign model params
    model.gr = 1.0
    model.nc = hyp['nc'][0]

    # training
    num_warmup = max(round(hyp['warmup_epochs'] * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    
    print(colorstr('=> start training...'))
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
        if (epoch >= args.val_start and (epoch % args.val_freq == 0 
                                                    or epoch == maxEpochs)):
                                                    
            savepath = wdir / f'epoch-{epoch}.pth'
            logger.info(f'{colorstr("=> saving checkpoint")} to {savepath}')
            ckpt = {
                'epoch': epoch,
                'state_dict':  model.state_dict(),
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                'global_steps':global_steps-1,
            }
            torch.save(ckpt, savepath)
            del ckpt

            # print('validate')
            da_segment_result, ll_segment_result, detect_result, total_loss, maps, t= test(
                epoch, args, hyp, valid_loader, model, criterion,save_dir,results_file, 
                                            Lane_color, DriveArea_color,logger, device)

            fi = fitness(np.array(detect_result).reshape(1, -1))  #目标检测评价指标

            # validation result tensorboard
            val_tensorboard(writer, global_steps-1, da_segment_result, 
                            ll_segment_result, detect_result, total_loss, maps, t)

    torch.cuda.empty_cache()
    return



    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, 
                            default='hyp/hyp.scratch.yolop.yaml', 
                            help='hyperparameter path')
                            # yolop_backbone
    parser.add_argument('--cfg', type=str, default='cfg/test.yaml', 
                                            help='model yaml path')
    parser.add_argument('--data', type=str, default='data/muti.yaml', 
                                            help='dataset yaml path')
    parser.add_argument('--logDir', type=str, default='runs/train',
                            help='log directory')
    parser.add_argument('--saveJson', type=bool, default=False)
    parser.add_argument('--saveTxt', type=bool, default=False)
    parser.add_argument('--allplot', type=bool, default=False)
    parser.add_argument('--resume', type=str, default='',
                            help='Resume the weight  runs/train/BddDataset/')
    parser.add_argument('--need_autoanchor', type=bool, default=False,
                            help='Re-select the prior anchor(k-means) \
                                    When training from scratch (epoch=0), \
                                    set it to be true!')
    parser.add_argument('--device', default='', 
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=10, 
                            help='total batch size for all GPUs')
    parser.add_argument('--test_batch_size', type=int, default=10, 
                            help='total batch size for all GPUs')
    parser.add_argument('--workers', type=int, default=0, 
                            help='maximum number of dataloader workers')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                            help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, 
                            help='IOU threshold for NMS')
    parser.add_argument('--val_start', type=int, default=20, 
                            help='start do validation')
    parser.add_argument('--val_freq', type=int, default=5, 
                            help='How many epochs do one time validation')
    # dataset   BDD100k_10k
    parser.add_argument('--dataset', type=str, default='BddDataset', 
                            help='save to dataset name')

    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], 
                            help='[train, test] image sizes')
    parser.add_argument('--org_img_size', nargs='+', type=int, default=[720, 1280], 
                            help='[train, test] original image sizes')

    parser.add_argument('--pretrain', type=str, default='', 
                            help='all branch pretrain')
   
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

    if(args.resume):
        args.save_dir = Path('./')
        for p in args.resume.split('/')[:-2]:
            args.save_dir= args.save_dir / p
    else:
        args.save_dir = increment_path(Path(args.logDir)/ args.dataset) 
    print(args.save_dir)

    # Train
    logger.info(args)
    logger.info(f"{colorstr('tensorboard: ')}Start with 'tensorboard --logdir {args.logDir}'"+\
                                        ", view at http://localhost:6006/")
    writer = SummaryWriter(args.save_dir)  # Tensorboard
    
    main(args, hyp, device, writer)
 


