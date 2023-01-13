import os
import sys
import math
import logging
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.metrics import  SegmentationMetric
from utils.autoanchor import check_anchor_order
from utils.torch_utils import time_synchronized, initialize_weights, model_info,\
                            select_device
from utils.general import make_divisible
from torch.nn import Upsample
from models.common import *


logger = logging.getLogger(__name__)

class Model(nn.Module):
    def __init__(self, cfg, nc, ch=3, anchors=None):
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml
            with open(cfg, 'r', encoding='utf-8') as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)  # load cfg

        # Define model
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        self.HeadOut = self.yaml['HeadOut']
        self.Det_nc = self.yaml['Det_nc']
        if nc:
            logger.info(f"Overriding model.yaml Det_nc={self.yaml['Det_nc']} with nc={nc[0]}")
            logger.info(f"Overriding model.yaml Lane_nc={self.yaml['Lane_nc']} with nc={nc[1]}")
            logger.info(f"Overriding model.yaml driveArea_nc={self.yaml['driveArea_nc']} with nc={nc[2]}")
            self.yaml['Det_nc'] = nc[0]  # override yaml value
            self.yaml['Lane_nc'] = nc[1]  # override yaml value
            self.yaml['driveArea_nc'] = nc[2]  # override yaml value
        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.Det_nc)]  # default names
        # print([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[self.HeadOut[0]]  # Detect()
        if isinstance(m, Detect) or isinstance(m, IDetect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            # with torch.no_grad():
            model_out = self.forward(torch.zeros(1, 3, s, s))
            detects, _, _= model_out
            m.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            m.anchors /= m.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(m)
            self.stride = m.stride
            if nc[0] > 1:
                self._initialize_biases()# only run once
            # print('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        # print(x.size())
        for i, block in enumerate(self.model):
            print(i, block)
            if block.f != -1:
                x = cache[block.f] if isinstance(block.f, int) else [x if j == -1 else cache[j] for j in block.f]       #calculate concat detect
            x = block(x)
            # try:
            #     print(x.size())
            # except:
            #     pass
            if i in self.HeadOut[1:]:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
            if i == self.HeadOut[0]:
                det_out = x
            cache.append(x if block.i in self.save else None)
        out.insert(0,det_out)
        return out ##det_out, da_seg_out,ll_seg_out
            
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.HeadOut[0]]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

def parse_model(d, ch):
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, gd, gw = d['anchors'], d['depth_multiple'], d['width_multiple']
    Det_nc, Lane_nc, driveArea_nc = d['Det_nc'], d['Lane_nc'], d['driveArea_nc']
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (Det_nc + 5)  # number of outputs = anchors * (classes + 5)

    
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['Neck'] + \
                d['Det_Head']  + d['DriveArea_Head']  + d['Lane_Head']):
        # print(i,f, n, m, args)
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass
        
        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [nn.Conv2d, Conv, RepConv, SPP, SPPCSPC, Focus, 
                Bottleneck, BottleneckCSP]:
            c1, c2 = ch[f], args[0]
            # if c2 != no:  # if not output
            #     c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [SPPCSPC]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m in [Detect, IDetect]:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]) if n > 1 else m(*args)
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    return nn.Sequential(*layers), sorted(save)


def get_optimizer(hyp, model):
    if hyp['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),lr=hyp['lr0'],
                                momentum=hyp['momentum'], weight_decay=hyp['wd'],
                                nesterov=hyp['nesterov'])
    elif hyp['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),lr=hyp['lr0'],
                                                betas=(hyp['momentum'], 0.999))   
    return optimizer


if __name__ == "__main__":
    cfg = 'F:/ITRI/YOLOP/cfg/YOLOP_v7bT2_ReConv.yaml'
    device = select_device('', batch_size=1)


    Det_class = 1
    Lane_class = 1
    DriveArea_class = 1
    nc = [Det_class, Lane_class, DriveArea_class]

    model = Model(cfg, nc).to(device)
    
    input = torch.randn((1, 3, 640, 640)).to(device, non_blocking=True)
    model_out = model(input)
    print(model_out)
    detects, dring_area_seg, lane_line_seg = model_out
    print('detects:', len(detects),detects[0].size())
    print('dring_area_seg:', dring_area_seg.size())
    print('lane_line_seg:', lane_line_seg.size())
