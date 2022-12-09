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
