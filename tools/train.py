import argparse
import logging
import os, sys
import yaml
from pathlib import Path
import math
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import pprint
import time
import torch
import torch.nn.parallel
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist
import torch.backends.cudnn
import torch.optim
import torch.utils.data


import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm


from lib.core.loss import get_loss
from lib.core.function import train, validate
from lib.core.general import fitness
from lib.utils.utils import save_checkpoint, \
                            get_optimizer, is_parallel, \
                            torch_distributed_zero_first \
                            
from lib.utils.autoanchor import run_anchor

from models.YOLOP import get_net
from utils.datasets import create_dataloader
from utils.general import colorstr, set_logging, increment_path

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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def main(args, hyp, device, tb_writer):
    logger.info(colorstr('hyperparameters: ') + ', '\
                                .join(f'{k}={v}' for k, v in hyp.items()))
    save_dir, epochs,  rank = Path(args.save_dir), args.epochs, args.global_rank
    begin_epoch = 0 

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    # last = wdir / 'last.pt'
    # best = wdir / 'best.pt'
    results_file = save_dir / 'results.txt'

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(save_dir / 'args.yaml', 'w') as f:
        yaml.dump(vars(args), f, sort_keys=False)

  


    # build up model
    print("begin to build up model...")
    model = get_net().to(device)

    # define loss function (criterion) and optimizer
    criterion = get_loss(hyp, device)
    optimizer = get_optimizer(hyp, model)
    print("finish build model")


    if rank == -1 and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # # DDP mode
    if rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,find_unused_parameters=True)



    # Data loading
    print("begin to load data")
    normalize = {'mean':[0.485, 0.456, 0.406], 
                 'std':[0.229, 0.224, 0.225]}
    train_loader, train_dataset = \
                create_dataloader(args, hyp, normalize)
    num_batch = len(train_loader)
    if rank in [-1, 0]:
        valid_loader, valid_dataset = \
                create_dataloader(args, hyp, normalize, is_train=False, shuffle=False)
    print('load data finished')
    



    # AUTOANCHOR
    if rank in [-1, 0]:
        if args.need_autoanchor:
            logger.info("begin check anchors")
            run_anchor(logger,train_dataset, model=model, thr=hyp['anchor_threshold'], imgsz=min(args.img_size))
        else:
            logger.info("anchors loaded successfully")
            det = model.module.model[model.module.detector_index] if is_parallel(model) \
                else model.model[model.detector_index]
            logger.info(str(det.anchors))


    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * \
                   (1 - hyp['lrf']) + hyp['lrf']  # cosine
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)


    # assign model params
    model.gr = 1.0
    model.nc = 1

    # training
    num_warmup = max(round(hyp['warmup_epochs'] * num_batch), 1000)
    scaler = amp.GradScaler(enabled=device.type != 'cpu')
    print('=> start training...')
    for epoch in range(begin_epoch+1, epochs+1):
        if rank != -1:
            train_loader.sampler.set_epoch(epoch)

        model.train()
        start = time.time()
        for i, (input, target, paths, shapes) in enumerate(train_loader):
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter()
            num_iter = i + num_batch * (epoch - 1)

            if num_iter < num_warmup:
                # warm up
                lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * \
                            (1 - hyp['lrf']) + hyp['lrf']  # cosine
                xi = [0, num_warmup]
                # model.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(num_iter, xi, [hyp['warmup_biase_lr'] \
                                        if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(num_iter, xi, [hyp['warmup_momentum'], 
                                                                    hyp['momentum']])

            data_time.update(time.time() - start)
            if not args.debug:
                input = input.to(device, non_blocking=True)
                assign_target = []
                for tgt in target:
                    assign_target.append(tgt.to(device))
                target = assign_target
            with amp.autocast(enabled=device.type != 'cpu'):
                outputs = model(input)
                total_loss, head_losses = criterion(outputs, target, shapes,model)
                # print(head_losses)

            # compute gradient and do update step
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if rank in [-1, 0]:
                # measure accuracy and record loss
                losses.update(total_loss.item(), input.size(0))

                # _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                #                                  target.detach().cpu().numpy())
                # acc.update(avg_acc, cnt)

                # measure elapsed time
                batch_time.update(time.time() - start)
                if i % 10 == 0:
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                        'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                        'Speed {speed:.1f} samples/s\t' \
                        'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            speed=input.size(0)/batch_time.val,
                            data_time=data_time, loss=losses)
                    logger.info(msg)
                    # Write
                    with open(results_file, 'a') as f:
                        f.write(msg+'\n')  

                    writer = tb_writer['writer']
                    global_steps = tb_writer['train_global_steps']
                    writer.add_scalar('train_loss', losses.val, global_steps)
                    # writer.add_scalar('train_acc', acc.val, global_steps)
                    tb_writer['train_global_steps'] = global_steps + 1


        # # train for one epoch
        # train(args, hyp, train_loader, model, criterion, optimizer, scaler,
        #       epoch, num_batch, num_warmup, tb_writer, logger, device, rank)
        
        lr_scheduler.step()

        # evaluate on validation set
        if (epoch > args.val_start and (epoch % args.val_freq == 0 or epoch == epochs)) and args.global_rank in [-1, 0]:
            # print('validate')
            da_segment_results,ll_segment_results,detect_results, total_loss,maps, times = validate(
                epoch, args, hyp, valid_loader, valid_dataset, model, criterion,
                save_dir, tb_writer,
                logger, device, rank
            )

            fi = fitness(np.array(detect_results).reshape(1, -1))  #目标检测评价指标
            
            msg = 'Epoch: [{0}]    Loss({loss:.3f})\n' \
                      'Driving area Segment: Acc({da_seg_acc:.3f})    IOU ({da_seg_iou:.3f})    mIOU({da_seg_miou:.3f})\n' \
                      'Lane line Segment: Acc({ll_seg_acc:.3f})    IOU ({ll_seg_iou:.3f})  mIOU({ll_seg_miou:.3f})\n' \
                      'Detect: P({p:.3f})  R({r:.3f})  mAP@0.5({map50:.3f})  mAP@0.5:0.95({map:.3f})\n'\
                      'Time: inference({t_inf:.4f}s/frame)  nms({t_nms:.4f}s/frame)'.format(
                          epoch,  loss=total_loss, da_seg_acc=da_segment_results[0],da_seg_iou=da_segment_results[1],da_seg_miou=da_segment_results[2],
                          ll_seg_acc=ll_segment_results[0],ll_seg_iou=ll_segment_results[1],ll_seg_miou=ll_segment_results[2],
                          p=detect_results[0],r=detect_results[1],map50=detect_results[2],map=detect_results[3],
                          t_inf=times[0], t_nms=times[1])
            logger.info(msg)
            with open(results_file, 'a') as f:
                        f.write(msg+'\n')  

            # if perf_indicator >= best_perf:
            #     best_perf = perf_indicator
            #     best_model = True
            # else:
            #     best_model = False

        # save checkpoint model and best model
        if rank in [-1, 0]:
            savepath = os.path.join(wdir, f'epoch-{epoch}.pth')
            logger.info('=> saving checkpoint to {}'.format(savepath))
            save_checkpoint(
                epoch=epoch,
                name=args.name,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=wdir,
                filename=f'epoch-{epoch}.pth'
            )
            save_checkpoint(
                epoch=epoch,
                name=args.name,
                model=model,
                # 'best_state_dict': model.module.state_dict(),
                # 'perf': perf_indicator,
                optimizer=optimizer,
                output_dir=os.path.join(args.logDir, args.dataset),
                filename='checkpoint.pth'
            )

    # save final model
    if rank in [-1, 0]:
        final_model_state_file = os.path.join(
            wdir, 'final_state.pth'
        )
        logger.info('=> saving final model state to {}'.format(
            final_model_state_file)
        )
        model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
        torch.save(model_state, final_model_state_file)
        tb_writer['writer'].close()
    else:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyp', type=str, default='lib/data/hyp.scratch.yolop.yaml', 
                            help='hyperparameter path')
    parser.add_argument('--logDir', type=str, default='runs/train',
                            help='log directory')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--debug', type=bool, default=False)

    parser.add_argument('--saveJson', type=bool, default=False)
    parser.add_argument('--saveTxt', type=bool, default=False)

    parser.add_argument('--auto_resume', type=bool, default=False,
                            help='Resume from the last training interrupt')
    parser.add_argument('--need_autoanchor', type=bool, default=False,
                            help='Re-select the prior anchor(k-means) \
                                    When training from scratch (epoch=0), \
                                    set it to be true!')
    parser.add_argument('--device', default='', 
                            help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--batch_size', type=int, default=11, 
                            help='total batch size for all GPUs')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], 
                            help='[train, test] image sizes')
    parser.add_argument('--org_img_size', nargs='+', type=int, default=[720, 1280], 
                            help='[train, test] original image sizes')
    parser.add_argument('--workers', type=int, default=0, 
                            help='maximum number of dataloader workers')
    parser.add_argument('--name', default='exp', 
                            help='save to project/name')
    parser.add_argument('--sync-bn', action='store_true', 
                            help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, 
                            help='DDP parameter, do not modify')
    parser.add_argument('--conf_thres', type=float, default=0.001,
                            help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.6, 
                            help='IOU threshold for NMS')
    parser.add_argument('--num_seg_class', type=int, default=2)
    parser.add_argument('--val_start', type=int, default=0, 
                            help='start do validation')
    parser.add_argument('--val_freq', type=int, default=1, 
                            help='How many epochs do one time validation')
    # dataset
    parser.add_argument('--dataset', type=str, default='BddDataset', 
                            help='save to dataset name')
    parser.add_argument('--dataRoot', type=str, default='F:/dataset/BDD100k_10k/bdd100k_images_10k/bdd100k/images/10k', 
                            help='the path of images folder')
    parser.add_argument('--labelRoot', type=str, default='F:/dataset/BDD100k_10k/labels/10k', 
                            help='the path of det_annotations folder')
    parser.add_argument('--maskRoot', type=str, default='F:/dataset/BDD100k_10k/labels/bdd_seg_gt', 
                            help='the path of da_seg_annotations folder')
    parser.add_argument('--laneRoot', type=str, default='F:/dataset/BDD100k_10k/labels/bdd_lane_gt', 
                            help='the path of ll_seg_annotations folder')
    parser.add_argument('--trainSet', type=str, default='train', 
                            help='IOU threshold for NMS')
    parser.add_argument('--testSet', type=str, default='val', 
                            help='IOU threshold for NMS')
    parser.add_argument('--dataFormat', type=str, default='jpg', 
                            help='Data Format')

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
    args = parser.parse_args()


    # Set DDP variables
    args.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' \
                                                    in os.environ else 1
    args.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    set_logging(args.global_rank)

    # cudnn related setting
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = args.cudnn_deterministic

    # DDP mode
    args.total_batch_size = args.batch_size
    device = select_device(args.device, batch_size=args.batch_size)
    if args.local_rank != -1:
        assert torch.cuda.device_count() > args.local_rank
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend
        assert args.batch_size % args.world_size == 0, '--batch-size must be multiple of CUDA device count'
        args.batch_size = args.total_batch_size // args.world_size


    # Hyperparameter
    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    hyp.update({'seg_only':SEG_ONLY,'det_only':DET_ONLY,
                'enc_seg_only':ENC_SEG_ONLY, 'enc_det_only':ENC_DET_ONLY,
                'drivable_only':DRIVABLE_ONLY, 'lane_only':LANE_ONLY,
                'det_only':DET_ONLY})


    args.save_dir = increment_path(Path(args.logDir)/ args.dataset)  # increment run

    # Train
    logger.info(args)
    tb_writer = None  # init loggers
    if args.global_rank in [-1, 0]:
        prefix = colorstr('tensorboard: ')
        logger.info(f"{prefix}Start with 'tensorboard --logdir {args.logDir}', view at http://localhost:6006/")
        tb_writer = SummaryWriter(args.save_dir)  # Tensorboard
        tb_writer ={
            'writer':tb_writer,
            'train_global_steps':0
        }
    main(args, hyp, device, tb_writer)
 


