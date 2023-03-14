import math
from collections import OrderedDict

import torch
import torch
from torch import nn
import torch.nn.functional as F

from models.common import Conv, E_ELAN, MPConv, OverlapPatchEmbed, MLP, resize, IDetect

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        print('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

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


class UNext(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.layer_1 = nn.Sequential(OrderedDict([
                            ('conv1', Conv(3,32,3,1)),
                            ('conv2', Conv(32,64,3,2)),
                            ]))    
        self.layer_2 = nn.Sequential(OrderedDict([
                            ('conv1', Conv(64,64,3,1)),
                            ('conv2', Conv(64,128,3,2)),
                            ('E_ELAN', E_ELAN(128,256)),
                            ('conv3', Conv(256,256,1,1)),
                            ]))  
        self.layer_3 = nn.Sequential(OrderedDict([
                            ('MPConv', MPConv(256,256)),
                            ('E_ELAN', E_ELAN(256,512)),
                            ('conv', Conv(512,512,1,1)),
                            ]))  
        self.layer_4 = nn.Sequential(OrderedDict([
                            ('MPConv', MPConv(512,512)),
                            ('E_ELAN', E_ELAN(512,1024)),
                            ('conv', Conv(1024,1024,1,1)),
                            ]))  
                            
        # img_size embed_dims
        # self.patch_embed1 = OverlapPatchEmbed(img_size=640 // 16, patch_size=3, stride=2, in_chans=1024,
        #                                       embed_dim=1024)
        # self.patch_embed2 = OverlapPatchEmbed(img_size=640 // 8, patch_size=3, stride=2, in_chans=512,
        #                                       embed_dim=1024)
        # self.patch_embed3 = OverlapPatchEmbed(img_size=640 // 4, patch_size=3, stride=2, in_chans=256,
        #                                       embed_dim=1024)
        
        self.linear_c4 = MLP(input_dim=1024, embed_dim=1024)
        self.linear_c3 = MLP(input_dim=512, embed_dim=1024)
        self.linear_c2 = MLP(input_dim=256, embed_dim=1024)
        self.linear_c1 = MLP(input_dim=64, embed_dim=1024)
        self.linear_c = MLP(input_dim=4096, embed_dim=2048)
        self.linear_conv1 = Conv(2048,1024,1,1)

        self.de4_conv = Conv(1024,1024,1,1)
        self.linear_c5 = MLP(input_dim=2048, embed_dim=512)
        
        self.de3_conv = Conv(512,512,1,1)
        self.linear_c6 = MLP(input_dim=1024, embed_dim=256)

        self.de2_conv = Conv(256,256,1,1)
        # driveable
        self.da_linear_c7 = MLP(input_dim=512, embed_dim=64)

        self.da_de1_conv = Conv(64,64,1,1)
        self.da_linear_c8 = MLP(input_dim=128, embed_dim=32)

        self.da_pred = Conv(32,nc[1],1,1)

        # lane line
        self.ll_linear_c7 = MLP(input_dim=512, embed_dim=64)

        self.ll_de1_conv = Conv(64,64,1,1)
        self.ll_linear_c8 = MLP(input_dim=128, embed_dim=32)

        self.ll_pred = Conv(32,nc[2],1,1)

        # object detection
        self.anchors = [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]]
        self.obj_conv1 = nn.Sequential(OrderedDict([
                            ('conv1', Conv(2048,2048,3,2)),
                            ('conv2', Conv(2048,1024,1,1)),
                            ]))   
        self.obj_conv2 = Conv(1024,512,1,1)
        self.obj_conv3 = Conv(512, 256,1,1)
        self.HeadOut = IDetect(nc=nc[0] ,anchors=self.anchors, ch=[256,512, 1024])

    def forward(self, x):
        # encoder
        out = self.layer_1(x)
        c1 = out

        out = self.layer_2(out)
        c2 = out

        out = self.layer_3(out)
        c3 = out

        out = self.layer_4(out)
        c4 = out

        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c4.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c4.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = resize(_c1, size=c4.size()[2:],mode='bilinear',align_corners=False)

        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_c(_c).permute(0,2,1).reshape(n, -1, _c.shape[2], _c.shape[3])
        # _c = resize(_c, size=c4.size()[2:],mode='bilinear',align_corners=False)
# ======================================================================
        # decoder
        c5 = torch.cat([self.linear_conv1(_c),self.de4_conv(c4)], dim=1)
        c5 = self.linear_c5(c5).permute(0,2,1).reshape(n, -1, c5.shape[2], c5.shape[3])
        c5 = resize(c5, size=c3.size()[2:],mode='bilinear',align_corners=False)
        
        c6 = torch.cat([c5,self.de3_conv(c3)], dim=1)
        c6 = self.linear_c6(c6).permute(0,2,1).reshape(n, -1, c6.shape[2], c6.shape[3])
        c6 = resize(c6, size=c2.size()[2:],mode='bilinear',align_corners=False)



        out = torch.cat([c6,self.de2_conv(c2)], dim=1)

        # driveable area
        da_out = self.da_linear_c7(out).permute(0,2,1).reshape(n, -1, out.shape[2], out.shape[3])
        da_out = resize(da_out, size=c1.size()[2:],mode='bilinear',align_corners=False)

        da_out = torch.cat([da_out,self.da_de1_conv(c1)], dim=1)
        da_out = self.da_linear_c8(da_out).permute(0,2,1).reshape(n, -1, da_out.shape[2], da_out.shape[3])
        da_out = resize(da_out, size=x.size()[2:],mode='bilinear',align_corners=False)
        
        # lane line
        ll_out = self.ll_linear_c7(out).permute(0,2,1).reshape(n, -1, out.shape[2], out.shape[3])
        ll_out = resize(ll_out, size=c1.size()[2:],mode='bilinear',align_corners=False)

        ll_out = torch.cat([ll_out,self.ll_de1_conv(c1)], dim=1)
        ll_out = self.ll_linear_c8(ll_out).permute(0,2,1).reshape(n, -1, ll_out.shape[2], ll_out.shape[3])
        ll_out = resize(ll_out, size=x.size()[2:],mode='bilinear',align_corners=False)
        
        # object detection
        deout = self.HeadOut([self.obj_conv3(c3),self.obj_conv2(c4),self.obj_conv1(_c)])

        return [deout, self.da_pred(da_out),self.ll_pred(ll_out)]
    


class Model(nn.Module):
  def __init__(self, cfg, nc, anchors=None, ch=3, ):
    super(Model, self).__init__()
    self.model = UNext(nc)

    # Build strides, anchors
    m = self.model.HeadOut  # Detect()
    if isinstance(m, IDetect):
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
        # if nc[0] > 1:
        self._initialize_biases()# only run once
        # print('Strides: %s' % m.stride.tolist())

    # Init weights, biases
    initialize_weights(self)
  def forward(self, x):
        return self.model(x)

  def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model.HeadOut  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
  def forward(self, x):         
    return self.model(x)



if __name__ == '__main__':
  model = Model(cfg=' ', nc=[8,2,9]).cuda()
  input = torch.ones(1,3,640,640).cuda()
  print(model)
#   model.eval()
  output = model(input)
  print(len(output[0]))
  print(output[0][0].shape)
  print(output[1].shape)
  print(output[2].shape)
#   print(output[0][0].shape)
#   print(output[1].shape)
#   print(output[2].shape)
