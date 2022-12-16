    # #   resume
    # if rank in [-1, 0]:
    #     Encoder_para_idx = [str(i) for i in range(0, 17)]
    #     Det_Head_para_idx = [str(i) for i in range(17, 25)]
    #     Da_Seg_Head_para_idx = [str(i) for i in range(25, 34)]
    #     Ll_Seg_Head_para_idx = [str(i) for i in range(34,43)]
    #     checkpoint_file = os.path.join(
    #         os.path.join(args.logDir, args.dataset), 'checkpoint.pth'
    #     )
    #     if os.path.exists(args.pretrain):
    #         logger.info("=> loading model '{}'".format(args.pretrain))
    #         checkpoint = torch.load(args.pretrain)
    #         begin_epoch = checkpoint['epoch']
    #         # best_perf = checkpoint['perf']
    #         last_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         logger.info("=> loaded checkpoint '{}' (epoch {})".format(
    #             args.pretrain, checkpoint['epoch']))
    #         #args.need_autoanchor = False     #disable autoanchor
        
    #     if os.path.exists(args.pretrain_det):
    #         logger.info("=> loading model weight in det branch from '{}'".format(args.pretrain))
    #         det_idx_range = [str(i) for i in range(0,25)]
    #         model_dict = model.state_dict()
    #         checkpoint_file = args.pretrain_det
    #         checkpoint = torch.load(checkpoint_file)
    #         begin_epoch = checkpoint['epoch']
    #         last_epoch = checkpoint['epoch']
    #         checkpoint_dict = {k: v for k, v in checkpoint['state_dict'].items() if k.split(".")[1] in det_idx_range}
    #         model_dict.update(checkpoint_dict)
    #         model.load_state_dict(model_dict)
    #         logger.info("=> loaded det branch checkpoint '{}' ".format(checkpoint_file))
        
    #     if args.auto_resume and os.path.exists(checkpoint_file):
    #         logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
    #         checkpoint = torch.load(checkpoint_file)
    #         begin_epoch = checkpoint['epoch']
    #         # best_perf = checkpoint['perf']
    #         last_epoch = checkpoint['epoch']
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         logger.info("=> loaded checkpoint '{}' (epoch {})".format(
    #             checkpoint_file, checkpoint['epoch']))
    #         #args.need_autoanchor = False     #disable autoanchor
    #     # model = model.to(device)

    #     if hyp['seg_only']:  #Only train two segmentation branch
    #         logger.info('freeze encoder and Det head...')
    #         for k, v in model.named_parameters():
    #             v.requires_grad = True  # train all layers
    #             if k.split(".")[1] in Encoder_para_idx + Det_Head_para_idx:
    #                 print('freezing %s' % k)
    #                 v.requires_grad = False

    #     if hyp['det_only']:  #Only train detection branch
    #         logger.info('freeze encoder and two Seg heads...')
    #         # print(model.named_parameters)
    #         for k, v in model.named_parameters():
    #             v.requires_grad = True  # train all layers
    #             if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
    #                 print('freezing %s' % k)
    #                 v.requires_grad = False

    #     if hyp['enc_seg_only']:  # Only train encoder and two segmentation branch
    #         logger.info('freeze Det head...')
    #         for k, v in model.named_parameters():
    #             v.requires_grad = True  # train all layers 
    #             if k.split(".")[1] in Det_Head_para_idx:
    #                 print('freezing %s' % k)
    #                 v.requires_grad = False

    #     if hyp['enc_det_only'] or hyp['det_only']:    # Only train encoder and detection branch
    #         logger.info('freeze two Seg heads...')
    #         for k, v in model.named_parameters():
    #             v.requires_grad = True  # train all layers
    #             if k.split(".")[1] in Da_Seg_Head_para_idx + Ll_Seg_Head_para_idx:
    #                 print('freezing %s' % k)
    #                 v.requires_grad = False


    #     if hyp['lane_only']: 
    #         logger.info('freeze encoder and Det head and Da_Seg heads...')
    #         # print(model.named_parameters)
    #         for k, v in model.named_parameters():
    #             v.requires_grad = True  # train all layers
    #             if k.split(".")[1] in Encoder_para_idx + Da_Seg_Head_para_idx + Det_Head_para_idx:
    #                 print('freezing %s' % k)
    #                 v.requires_grad = False

    #     if hyp['drivable_only']:
    #         logger.info('freeze encoder and Det head and Ll_Seg heads...')
    #         # print(model.named_parameters)
    #         for k, v in model.named_parameters():
    #             v.requires_grad = True  # train all layers
    #             if k.split(".")[1] in Encoder_para_idx + Ll_Seg_Head_para_idx + Det_Head_para_idx:
    #                 print('freezing %s' % k)
    #                 v.requires_grad = False






# class HustDataset(AutoDriveDataset):
#     def __init__(self, cfg, is_train, inputsize, transform=None):
#         super().__init__(cfg, is_train, inputsize, transform)
#         self.db = self._get_db()
#         self.cfg = cfg

#     def _get_db(self):
#         """
#         get database from the annotation file

#         Inputs:

#         Returns:
#         gt_db: (list)database   [a,b,c,...]
#                 a: (dictionary){'image':, 'information':, ......}
#         image: image path
#         mask: path of the segmetation label
#         label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
#         """
#         print('building database...')
#         gt_db = []
#         height, width = self.shapes
#         for mask in tqdm(list(self.mask_list)):
#             mask_path = str(mask)
#             label_path = self.label_root
#             # label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
#             image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
#             lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
#             with open(label_path, 'r') as f:
#                 label = json.load(f)
#             data = label[int(os.path.basename(image_path)[:-4])]["labels"]
#             data = self.filter_data(data)
#             gt = np.zeros((len(data), 5))
#             for idx, obj in enumerate(data):
#                 category = obj['category']
#                 if category == "traffic light":
#                     color = obj['attributes']['Traffic Light Color'][0]
#                     category = "tl_" + color
#                 if category in id_dict.keys():
#                     x1 = float(obj['box2d']['x1'])
#                     y1 = float(obj['box2d']['y1'])
#                     x2 = float(obj['box2d']['x2'])
#                     y2 = float(obj['box2d']['y2'])
#                     cls_id = id_dict[category]
#                     if single_cls:
#                          cls_id=0
#                     gt[idx][0] = cls_id
#                     box = convert((width, height), (x1, x2, y1, y2))
#                     gt[idx][1:] = list(box)
                

#             rec = [{
#                 'image': image_path,
#                 'label': gt,
#                 'mask': mask_path,
#                 'lane': lane_path
#             }]

#             gt_db += rec
#         print('database build finish')
#         return gt_db

#     def filter_data(self, data):
#         remain = []
#         for obj in data:
#             if 'box2d' in obj.keys():  # obj.has_key('box2d'):
#                 if single_cls:
#                     if obj['category'] in id_dict_single.keys():
#                         remain.append(obj)
#                 else:
#                     remain.append(obj)
#         return remain

#     def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
#         """  
#         """
#         pass










# class LoadStreams:  # multiple IP or RTSP cameras
#     def __init__(self, sources='streams.txt', img_size=640, auto=True):
#         self.mode = 'stream'
#         self.img_size = img_size

#         if os.path.isfile(sources):
#             with open(sources, 'r') as f:
#                 sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
#         else:
#             sources = [sources]

#         n = len(sources)
#         self.imgs, self.fps, self.frames, self.threads = [None] * n, [0] * n, [0] * n, [None] * n
#         self.sources = [clean_str(x) for x in sources]  # clean source names for later
#         self.auto = auto
#         for i, s in enumerate(sources):  # index, source
#             # Start thread to read frames from video stream
#             print(f'{i + 1}/{n}: {s}... ', end='')
#             s = eval(s) if s.isnumeric() else s  # i.e. s = '0' local webcam
#             cap = cv2.VideoCapture(s)
#             assert cap.isOpened(), f'Failed to open {s}'
#             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#             self.fps[i] = max(cap.get(cv2.CAP_PROP_FPS) % 100, 0) or 30.0  # 30 FPS fallback
#             self.frames[i] = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback

#             _, self.imgs[i] = cap.read()  # guarantee first frame
#             self.threads[i] = Thread(target=self.update, args=([i, cap]), daemon=True)
#             print(f" success ({self.frames[i]} frames {w}x{h} at {self.fps[i]:.2f} FPS)")
#             self.threads[i].start()
#         print('')  # newline

#         # check for common shapes

#         s = np.stack([letterbox_for_img(x, self.img_size, auto=self.auto)[0].shape for x in self.imgs], 0)  # shapes
#         self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
#         if not self.rect:
#             print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

#     def update(self, i, cap):
#         # Read stream `i` frames in daemon thread
#         n, f, read = 0, self.frames[i], 1  # frame number, frame array, inference every 'read' frame
#         while cap.isOpened() and n < f:
#             n += 1
#             # _, self.imgs[index] = cap.read()
#             cap.grab()
#             if n % read == 0:
#                 success, im = cap.retrieve()
#                 self.imgs[i] = im if success else self.imgs[i] * 0
#             time.sleep(1 / self.fps[i])  # wait time

#     def __iter__(self):
#         self.count = -1
#         return self

#     def __next__(self):
#         self.count += 1
#         if not all(x.is_alive() for x in self.threads) or cv2.waitKey(1) == ord('q'):  # q to quit
#             cv2.destroyAllWindows()
#             raise StopIteration

#         # Letterbox
#         img0 = self.imgs.copy()

#         h0, w0 = img0[0].shape[:2]
#         img, _, pad = letterbox_for_img(img0[0], self.img_size, auto=self.rect and self.auto)

#         # Stack
#         h, w = img.shape[:2]
#         shapes = (h0, w0), ((h / h0, w / w0), pad)

#         # Convert
#         #img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
#         img = np.ascontiguousarray(img)

#         return self.sources, img, img0[0], None, shapes

#     def __len__(self):
#         return len(self.sources)  # 1E12 frames = 32 streams at 30 FPS for 30 years


# if args.source.isnumeric():
    #     cudnn.benchmark = True  
    #     dataset = LoadStreams(args.source, img_size=args.img_size)
    #     bs = len(dataset)  # batch_size
    # else: