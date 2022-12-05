## main command

### train
python train.py --cfg cfg/YOLOP_v7bT1.yaml

### Test (not test)
python test.py --weights weights/epoch-240.pth

### Demo (not test)

python demo.py --cfg cfg/YOLOP_v7b1.yaml  --source inference/images --weights weights/epoch-116.pth

python demo.py --source F:/CEX_EXT/Map/data/real_image/images

## TOOL
### Tensorboard
tensorboard --logdir=runs