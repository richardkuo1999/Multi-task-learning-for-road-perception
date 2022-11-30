import glob
import datetime
import logging
import os
import time
from collections import namedtuple
import platform
import subprocess
from pathlib import Path

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from contextlib import contextmanager
import re

logger = logging.getLogger(__name__)

def clean_str(s):
    # Cleans a string by replacing special characters with underscore _
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl="_", string=s)



# def create_logger(dataset, cfg_path, phase='train', rank=-1):
#     # set up logger dir
#     dataset = dataset
#     dataset = dataset.replace(':', '_')
#     cfg_path = os.path.basename(cfg_path).split('.')[0]

#     if rank in [-1, 0]:
#         time_str = time.strftime('%Y-%m-%d-%H-%M')
#         log_file = '{}_{}_{}.log'.format(cfg_path, time_str, phase)
#         # set up tensorboard_log_dir
#         tensorboard_log_dir = Path(cfg_path) / dataset  / time_str
#         final_output_dir = tensorboard_log_dir
#         if not tensorboard_log_dir.exists():
#             print('=> creating {}'.format(tensorboard_log_dir))
#             tensorboard_log_dir.mkdir(parents=True)

#         final_log_file = tensorboard_log_dir / log_file
#         head = '%(asctime)-15s %(message)s'
#         logging.basicConfig(filename=str(final_log_file),
#                             format=head)
#         logger = logging.getLogger()
#         logger.setLevel(logging.INFO)
#         console = logging.StreamHandler()
#         logging.getLogger('').addHandler(console)

#         return logger, str(final_output_dir), str(tensorboard_log_dir)
#     else:
#         return None, None, None




def get_optimizer(hyp, model):
    optimizer = None
    if hyp['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=hyp['lr0'],
            momentum=hyp['momentum'],
            weight_decay=hyp['wd'],
            nesterov=hyp['nesterov']
        )
    elif hyp['optimizer'] == 'adam':
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            #model.parameters(),
            lr=hyp['lr0'],
            betas=(hyp['momentum'], 0.999)
        )

    return optimizer


def save_checkpoint(epoch, name, model, optimizer, output_dir, filename, is_best=False):
    model_state = model.module.state_dict() if is_parallel(model) else model.state_dict()
    checkpoint = {
            'epoch': epoch,
            'model': name,
            'state_dict': model_state,
            # 'best_state_dict': model.module.state_dict(),
            # 'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }
    torch.save(checkpoint, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in checkpoint:
        torch.save(checkpoint['best_state_dict'],
                   os.path.join(output_dir, 'model_best.pth'))


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
        # elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()

