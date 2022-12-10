import cv2
import time
import shutil
import os, sys
import argparse
from pathlib import Path
from numpy import random

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms




from models.YOLOP import get_net
from utils.datasets import LoadImages, LoadStreams
from utils.plot import plot_one_box,show_seg_result
from utils.torch_utils import select_device, time_synchronized
from utils.postprocess import morphological_process, connect_lane
from utils.general import increment_path, write_log, non_max_suppression,\
                        scale_coords, AverageMeter





normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])


def detect(args, device):
    save_dir = args.save_dir

    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    results_file = save_dir / 'results.txt'
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(args.cfg)
    checkpoint = torch.load(args.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if args.source.isnumeric():
        cudnn.benchmark = True  
        dataset = LoadStreams(args.source, img_size=args.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(args.source, img_size=args.img_size)
        bs = 1  # batch_size


    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]


    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, args.img_size, args.img_size), device=device)  
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()
    
    for i, (path, img, img_det, vid_cap,shapes) in enumerate(dataset):
        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out= model(img)
        t2 = time_synchronized()
        # if i == 0:
        #     print(det_out)
        inf_out, _ = det_out
        inf_time.update(t2-t1,img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=args.conf_thres, 
                                        iou_thres=args.iou_thres, classes=None, 
                                        agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det=det_pred[0]

        save_path = save_dir / Path(path).name \
            if dataset.mode != 'stream' else save_dir / "web.mp4"

        _, _, height, width = img.shape
        h,w,_=img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(da_predict, 
                                        scale_factor=int(1/ratio), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        
        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, 
                                        scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # Lane line post-processing
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, 
                                                            is_demo=True)

        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, 
                                        color=colors[int(cls)], line_thickness=2)
        
        if dataset.mode == 'image':
            cv2.imwrite(str(save_path),img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_=img_det.shape
                vid_writer = cv2.VideoWriter(save_path, 
                                    cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)
        
        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    # print('Results saved to %s' % Path(args.save_dir))
    # print('Done. (%.3f s)' % (time.time() - t0))
    # print('inf : (%.4f s/frame)   nms : (%.4fs/frame)' % (inf_time.avg,nms_time.avg))
    # print('fps : (%.4f frame/s)' % (1/(inf_time.avg+nms_time.avg)))
    msg = f'{str(args.weights)} , {str(args.cfg)}\n'+\
          f'Results saved to {str(args.save_dir)}\n'+\
          f'Done. ({(time.time() - t0)} s)\n'+\
          f'inf : ({inf_time.avg} s/frame)   nms : ({nms_time.avg}s/frame)\n'+\
          f'fps : ({(1/(inf_time.avg+nms_time.avg))} frame/s)'
    print(msg)
    write_log(results_file, msg)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logDir', type=str, default='runs/demo',
                            help='log directory')
    parser.add_argument('--weights', type=str, default='weights/epoch-295.pth', 
                                                    help='model.pth path(s)')
    parser.add_argument('--cfg', type=str, default='cfg/YOLOP_v7bT2.yaml', 
                                                    help='model.yaml path')
    parser.add_argument('--source', type=str, default='inference/images', 
                                                    help='source')  
    parser.add_argument('--img-size', type=int, default=640, 
                                                    help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, 
                                                help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, 
                                                    help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', 
                                    help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--dataset', type=str, default='BddDataset', 
                            help='save to dataset name')
    parser.add_argument('--save-dir', type=str, default='inference/output', 
                                                help='directory to save results')
    parser.add_argument('--augment', action='store_true',help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    args = parser.parse_args()

    device = select_device(args.device)
    
    # test_yaml = ['cfg/yolop.yaml','cfg/yolop.yaml','cfg/YOLOP_v7b1.yaml','cfg/YOLOP_v7b2.yaml','cfg/YOLOP_v7bT1.yaml','cfg/yolop.yaml']
    # test_weight = ['weights/epoch-240.pth','weights/epoch-116.pth','weights/epoch-280.pth','weights/epoch-200.pth','weights/epoch-295.pth','weights/epoch-182.pth',]
    # for i, test_file in enumerate(zip(test_yaml,test_weight)):
    #     print(i, test_file[0],test_file[1])
    #     args.cfg = test_file[0]
    #     args.weights = test_file[1]
        
    #     args.save_dir = increment_path(Path(args.logDir)/ args.dataset)  # increment run
    #     with torch.no_grad():
    #         detect(args, device)

    args.save_dir = increment_path(Path(args.logDir)/ args.dataset)  # increment run
    with torch.no_grad():
        detect(args, device)