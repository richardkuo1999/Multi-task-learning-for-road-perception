import os
import sys
import yaml
import torch
import torch.nn as nn
import math
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.metrics import  SegmentationMetric
from utils.autoanchor import check_anchor_order
from utils.torch_utils import time_synchronized, initialize_weights
from torch.nn import Upsample
from models.common import MP, Conv, SP, SPP, SPPCSPC, Bottleneck,\
                         BottleneckCSP, Focus, Concat, Detect, RepConv,\
                         IDetect, SharpenConv


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            # print(i, block)
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block in [Detect,IDetect]:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect) or isinstance(Detector, IDetect):
            s = 128  # 2x min stride
            # for x in self.forward(torch.zeros(1, 3, s, s)):
            #     print (x.shape)
            # with torch.no_grad():
            model_out = self.forward(torch.zeros(1, 3, s, s))
            detects, _, _= model_out
            Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            # print("stride"+str(Detector.stride ))
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()# only run once
            # print('Strides: %s' % m.stride.tolist())
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        Da_fmap = []
        LL_fmap = []
        # print(x.size())
        for i, block in enumerate(self.model):
            print(i, block)
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            # try:
            #     print(x.size())
            # except:
            #     pass
            if i in self.seg_out_idx:     #save driving area segment result
                m=nn.Sigmoid()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out ##det_out, da_seg_out,ll_seg_out
            
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def parse_model(cfg):
    for i, (f, m, args) in enumerate(cfg):
        # print(i,m)
        cfg[i][-2] = eval(m) if isinstance(m, str) else m
        if( isinstance(cfg[i][-1], list)):
            for j in range(len(cfg[i][-1])):
                cfg[i][-1][j] = None if cfg[i][-1][j] == 'None' else cfg[i][-1][j]
    return cfg


def get_net(cfg, **kwargs): 
    with open(cfg, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)  # load cfg

    m_block_cfg = parse_model(cfg['YOLOP'])
    return MCnet(m_block_cfg, **kwargs)


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
    # from torch.utils.tensorboard import SummaryWriter
    cfg = 'F:\ITRI\YOLOP\cfg\yolop.yaml'
    model = get_net(cfg)
    
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out,SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
 
