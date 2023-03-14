import torch
import torch.nn as nn

from utils.general import bbox_iou
from utils.metrics import  SegmentationMetric
from utils.postprocess import build_targets


class MultiHeadLoss(nn.Module):
    """
    collect all the loss we need
    """
    def __init__(self, hyp, device):
        """
        Inputs:
        - losses: (list)[nn.Module, nn.Module, ...]
        - cfg: config object
        """
        super().__init__()
        
        self.hyp = hyp
        self.nc = hyp['nc']


        # class loss criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['cls_pos_weight']])).to(device)
        # object loss criteria
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['obj_pos_weight']])).to(device)
        # drivable area segmentation loss criteria
        da_w = torch.ones(self.nc[1])
        da_w[0] = 0.05
        daseg = (nn.CrossEntropyLoss(weight=da_w) if self.nc[1] > 2 else  \
                nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['seg_pos_weight']]))).to(device)
        # lane line segmentation loss criteria
        ll_w = torch.ones(self.nc[2])
        ll_w[0] = 0.05
        llseg = (nn.CrossEntropyLoss(weight=ll_w) if self.nc[2] > 2 else \
                nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([hyp['seg_pos_weight']]))).to(device)
        # Focal loss
        gamma = hyp['fl_gamma']  # focal loss gamma
        if gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

        losses = [BCEcls, BCEobj, daseg, llseg]

        self.losses = nn.ModuleList(losses)
        

    def forward(self, head_fields, head_targets, shapes, model):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head
        - model:

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]

        """
        total_loss, head_losses = self._forward_impl(head_fields, head_targets, shapes, model)

        return total_loss, head_losses

    def _forward_impl(self, predictions, targets, shapes, model):
        """

        Args:
            predictions: predicts of [[det_head1, det_head2, det_head3], drive_area_seg_head, lane_line_seg_head]
            targets: gts [det_targets, segment_targets, lane_targets]
            model:

        Returns:
            total_loss: sum of all the loss
            head_losses: list containing losses

        """
        hyp = self.hyp
        device = targets[0].device
        _, _, daseg, llseg = self.losses

        lbox, lobj, lcls, no = self.det_loss(predictions[0], targets[0], model, device)
        lseg_da = daseg(predictions[1], targets[1])
        lseg_ll = llseg(predictions[2], targets[2])
        
        s = 3 / no  # output count scaling

        lcls *= hyp['cls_gain'] * s
        lobj *= hyp['obj_gain'] * s * (1.4 if no == 4 else 1.)
        lbox *= hyp['box_gain'] * s

        lseg_da *= hyp['da_seg_gain']
        lseg_ll *= hyp['ll_seg_gain']

        loss = lbox + lobj + lcls + lseg_da + lseg_ll
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), loss.item())

    def det_loss(self, predictions, targets, model, device):
        hyp = self.hyp
        BCEcls, BCEobj, _, _ = self.losses

        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = build_targets(hyp, predictions, targets, model)  # targets

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        cp, cn = smooth_BCE(eps=0.0)

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions)  # number of outputs
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)  # iou ratio

                # Classification
                # print(model.nc)
                if model.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], cn, device=device)  # targets
                    t[range(n), tcls[i]] = cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss
        return lbox, lobj, lcls, no



def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss