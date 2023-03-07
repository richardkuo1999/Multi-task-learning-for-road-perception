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

from utils.general import xyxy2xywh, xywh2xyxy, xywhn2xyxy, convert, one_hot_it_v11_dice
from utils.augmentations import augment_hsv, random_perspective, letterbox,\
                                 letterbox_for_img



def create_dataloader(args, hyp, data_dict, batch_size, normalize, is_train=True, shuffle=True):
    normalize = transforms.Normalize(
            normalize['mean'], normalize['std']
        )
    
    datasets = eval(args.dataset)(
        args=args,
        hyp=hyp,
        data_dict=data_dict,
        dataSet=data_dict['train'] if is_train else data_dict['val'],
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
    def __init__(self, args, hyp, data_dict, dataSet, is_train, transform=None):
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

        self.Tensor = transforms.ToTensor()

        # Data Root
        self.img_root = Path(dataSet[0])
        self.label_root = Path(dataSet[1])
        self.mask_root = Path(dataSet[2])
        self.lane_root = Path(dataSet[3])
        self.label_Lane_info = data_dict['Lane_names']
        self.label_drivable_info = data_dict['DriveArea_names']

        self.mask_list = self.mask_root.iterdir()

        self.db = []

    
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
        hyp = self.hyp
        data = self.db[idx]
        resized_shape = max(self.inputsize) if isinstance(self.inputsize, list) \
                                            else self.inputsize

        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]  # orig hw

        drivable_label = cv2.imread(data["mask"])

        lane_label = cv2.imread(data["lane"])
        lane_label = cv2.cvtColor(lane_label, cv2.COLOR_BGR2RGB)


        #resize
        (img, drivable_label, lane_label), ratio, pad = letterbox((img, drivable_label, lane_label),\
                                         resized_shape, auto=True, scaleup=self.is_train)
        h, w = img.shape[:2]


        with open(data["label"], 'r') as f:
                label = json.load(f)
        obj_data = label['frames'][0]['objects']
        obj_data = self.filter_data(obj_data)
        gt = np.zeros((len(obj_data), 5))


        for idx, obj in enumerate(obj_data):
            category = obj['category']
            # if category == "traffic light":
            #     color = obj['attributes']['trafficLightColor']
            #     category = "tl_" + color
            if category in self.data_dict['Det_names']:
                x1 = float(obj['box2d']['x1'])
                y1 = float(obj['box2d']['y1'])
                x2 = float(obj['box2d']['x2'])
                y2 = float(obj['box2d']['y2'])

                gt[idx][0] = self.data_dict['Det_names'].index(category) 
                box = convert((w0, h0), (x1, x2, y1, y2))
                gt[idx][1:] = list(box)

        det_label = gt
        
        labels=[]
        
        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1:5] = xywhn2xyxy(labels[:, 1:5], ratio[0] * w0, ratio[1] * h0, \
                                        padw=pad[0], padh=pad[1])

        from utils.plot import plot_one_box
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]    
        for cls,x1,y1,x2,y2 in labels:
            xyxy = (x1,y1,x2,y2)
            plot_one_box(xyxy, img, color=colors[int(cls)], label=str(int(cls)), line_thickness=1)
        cv2.imwrite("./batch_1_1_det_gt.png",img)

        if self.is_train:
            combination = (img, drivable_label, lane_label)
            (img, drivable_label, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=hyp['rot_factor'],
                translate=hyp['translate'],
                scale=hyp['scale_factor'],
                shear=hyp['shear']
            )
            #print(labels.shape)
            augment_hsv(img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])
            # img, drivable_label, labels = cutout(combination=combination, labels=labels)

        if len(labels):
            # convert xyxy to xywh
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

            # Normalize coordinates 0 - 1
            labels[:, [2, 4]] /= img.shape[0]  # height
            labels[:, [1, 3]] /= img.shape[1]  # width

        if self.is_train:
        # random left-right flip
            if random.random() < hyp['fliplr']:
                img = np.fliplr(img)
                drivable_label = np.fliplr(drivable_label)
                lane_label = np.fliplr(lane_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            if random.random() < hyp['flipud']:
                img = np.flipud(img)
                drivable_label = np.flipud(drivable_label)
                lane_label = np.flipud(lane_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]

        labels_out = torch.zeros((len(labels), 6))
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(labels)
        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # drivable_label = np.ascontiguousarray(drivable_label)
        # if idx == 0:
        #     print(drivable_label[:,:,0])
        
        
        

        drivable_label = one_hot_it_v11_dice(drivable_label, self.label_drivable_info)
        lane_label = one_hot_it_v11_dice(lane_label, self.label_Lane_info)

        # from PIL import Image
        # aaa = img.copy()
        # drivable_label_bool = drivable_label.copy().astype(dtype=bool)
        # for i in range(0,len(drivable_label_bool[0,0])):
        #     aaa[drivable_label_bool[:,:,i]] = self.label_drivable_info[list(self.label_drivable_info)[i]][:3]

        # lane_label_bool = lane_label.copy().astype(dtype=bool)
        # for i in range(1,len(lane_label_bool[0,0])):
        #     aaa[lane_label_bool[:,:,i]] = self.label_Lane_info[list(self.label_Lane_info)[i]][:3]
        # aaa = Image.fromarray(aaa, "RGB")
        # aaa.save(f'{idx}.png')
        # cv2.imwrite(f'{idx}.jpg',img)

        drivable_label = self.Tensor(drivable_label)
        lane_label = self.Tensor(lane_label)


        img = self.transform(img)
        target = [labels_out, drivable_label, lane_label]
        shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling

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
    def __init__(self, args, hyp, data_dict, dataSet, is_train, transform=None):
        super().__init__(args, hyp, data_dict, dataSet, is_train, transform)
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
        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), 
                                str(self.label_root)).replace(".png", ".json")
            image_path = mask_path.replace(str(self.mask_root), 
                                str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.mask_root), 
                                str(self.lane_root))


            rec = [{
                'image': image_path,
                'label': label_path,
                'mask': mask_path,
                'lane': lane_path
            }]

            gt_db += rec
        return gt_db

    def filter_data(self, data: list)->list:
        """Filter useless image in the dataset

        Args:
            data (list): database

        Returns:
            remain (list): filtered dataset
        """
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if obj['category'] in self.data_dict['Det_names']:
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


