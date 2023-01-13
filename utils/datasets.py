import os
import cv2
import yaml
import glob
import json
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from prefetch_generator import BackgroundGenerator

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from utils.general import xyxy2xywh, convert
from utils.augmentations import augment_hsv, random_perspective, letterbox,\
                                 letterbox_for_img

id_dict = {'person': 0, 'rider': 1, 'car': 2, 'bus': 3, 'truck': 4, 
'bike': 5, 'motor': 6, 'tl_green': 7, 'tl_red': 8, 
'tl_yellow': 9, 'tl_none': 10, 'traffic sign': 11, 'train': 12}
id_dict_single = {'car': 0, 'bus': 1, 'truck': 2,'train': 3}
# id_dict = {'car': 0, 'bus': 1, 'truck': 2}

single_cls = True       # just detect vehicle


def create_dataloader(args, hyp, data_dict, batch_size, normalize, is_train=True, shuffle=True):
    normalize = transforms.Normalize(
            normalize['mean'], normalize['std']
        )

    datasets = eval(args.dataset)(
        args=args,
        hyp=hyp,
        data_dict=data_dict,
        is_train=is_train,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    loader = DataLoaderX(
        datasets,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=AutoDriveDataset.collate_fn
    )
    return loader, datasets

class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class AutoDriveDataset(Dataset):
    """
    A general Dataset for some common function
    """
    def __init__(self, args, hyp, data_dict, is_train, transform=None):
        """
        initial all the characteristic

        Inputs:
        -args: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize
        
        Returns:
        None
        """
        self.is_train = is_train
        self.hyp = hyp
        self.data_dict = data_dict
        self.transform = transform
        self.inputsize = args.img_size
        self.num_seg_class = args.num_seg_class
        self.shapes = np.array(args.org_img_size)

        # Data Root
        self.img_root = Path(data_dict[0])
        self.label_root = Path(data_dict[1])
        self.mask_root = Path(data_dict[2])
        self.lane_root = Path(data_dict[3])
            
        self.img_list = self.img_root.iterdir()
        self.Tensor = transforms.ToTensor()
    
    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, args, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError
    
    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and ground-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        seg_label = cv2.imread(data["mask"])
        lane_label = cv2.imread(data["lane"])
        # TODO
        # print(lane_label.shape)
        # print(seg_label.shape)
        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw
        r = resized_shape / max(h0, w0)  # resize image to img_size
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]
        
        (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), resized_shape, auto=True, scaleup=self.is_train)
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)
        
        det_label = data["label"]
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]
            
        if self.is_train:
            combination = (img, seg_label, lane_label)
            (img, seg_label, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.hyp['rot_factor'],
                translate=self.hyp['translate'],
                scale=self.hyp['scale_factor'],
                shear=self.hyp['shear']
            )
            #print(labels.shape)
            augment_hsv(img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])
            # img, seg_label, labels = cutout(combination=combination, labels=labels)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                lane_label = np.fliplr(lane_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.flipud(seg_label)
                lane_label = np.flipud(lane_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]
        
        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])
        # TODO
        if self.num_seg_class == 3:
            _,seg0 = cv2.threshold(seg_label[:,:,0],128,255,cv2.THRESH_BINARY)
            _,seg1 = cv2.threshold(seg_label[:,:,1],1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        else:
            _,seg1 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY)
            _,seg2 = cv2.threshold(seg_label,1,255,cv2.THRESH_BINARY_INV)
        _,lane1 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY)
        _,lane2 = cv2.threshold(lane_label,1,255,cv2.THRESH_BINARY_INV)
        # _,seg2 = cv2.threshold(seg_label[:,:,2],1,255,cv2.THRESH_BINARY)
        # # seg1[cutout_mask] = 0
        # # seg2[cutout_mask] = 0
        
        # seg_label /= 255
        # seg0 = self.Tensor(seg0)
        # TODO
        if self.num_seg_class == 3:
            seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        # seg1 = self.Tensor(seg1)
        # seg2 = self.Tensor(seg2)
        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)
        # TODO
        # seg_label = torch.stack((seg2[0], seg1[0]),0)
        if self.num_seg_class == 3:
            seg_label = torch.stack((seg0[0],seg1[0],seg2[0]),0)
        else:
            seg_label = torch.stack((seg2[0], seg1[0]),0)
            
        lane_label = torch.stack((lane2[0], lane1[0]),0)
        # _, gt_mask = torch.max(seg_label, 0)
        # _ = show_seg_result(img, gt_mask, idx, 0, save_dir='debug', is_gt=True)
        

        target = [labels_out, seg_label, lane_label]
        img = self.transform(img)

        return img, target, data["image"], shapes

    def filter_data(self, db):
        """
        finished on children dataset
        """
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes= zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_lane.append(l_lane)
        return torch.stack(img, 0), [torch.cat(label_det, 0), torch.stack(label_seg, 0), torch.stack(label_lane, 0)], paths, shapes

class BddDataset(AutoDriveDataset):
    def __init__(self, args, hyp, data_dict, is_train, transform=None):
        super().__init__(args, hyp, data_dict, is_train, transform)
        self.db = self.__get_db()
        
    def __get_db(self):
        """get database from the annotation file

        Returns:
            gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'label':, 'mask':,'lane':}
            image: image path
            label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
            mask: path of the driver area segmentation label path
            lane: path of the lane segmentation label path
        """
        gt_db = []
        for ImageName in tqdm(list(self.img_list)):
            name = ImageName.name.split('.')[0]
            label_path = str(self.label_root / (name+'.json'))

            rec = {
                'image': str(self.img_root / (name+'.jpg')),
                'label': self.__GetOD_GT(label_path),
                'mask': str(self.mask_root / (name+'.png')),
                'lane': str(self.lane_root / (name+'.png'))
            }

            gt_db.append(rec)
        return gt_db

    def __GetOD_GT(self, label_path):
        height, width = self.shapes
        with open(label_path, 'r') as f:
                label = json.load(f)    
        data = label['frames'][0]['objects']
        data = self.filter_data(data)
        gt = np.zeros((len(data), 5))
        for idx, obj in enumerate(data):
            category = obj['category']
            if obj['category'] in id_dict.keys():
                x1 = float(obj['box2d']['x1'])
                y1 = float(obj['box2d']['y1'])
                x2 = float(obj['box2d']['x2'])
                y2 = float(obj['box2d']['y2'])

                gt[idx][0] = 0 if single_cls else id_dict[category]
                box = convert((width, height), (x1, x2, y1, y2))
                gt[idx][1:] = list(box)
        return gt

    def filter_data(self, data: list)->list:
        # TODO deal with dataset(don't convert evert time)
        """Filter useless image in the dataset

        Args:
            data (list): database

        Returns:
            remain (list): filtered dataset
        """
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if single_cls:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain



img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
vid_formats = ['.mov', '.avi', '.mp4', '.mpg', '.mpeg', '.m4v', '.wmv', '.mkv']
class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if os.path.splitext(x)[-1].lower() in img_formats]
        videos = [x for x in files if os.path.splitext(x)[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()
            h0, w0 = img0.shape[:2]

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
            #img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ')
            h0, w0 = img0.shape[:2]

        # Padded resize
        img, ratio, pad = letterbox_for_img(img0, new_shape=self.img_size, auto=True)
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Convert
        #img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)


        # cv2.imwrite(path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
        return path, img, img0, self.cap, shapes

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files


