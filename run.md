## main command

### train
python tools/train.py

### Test (not test)
python tools/test.py --weights weights/epoch-240.pth

### Demo (not test)

python tools/demo.py --source inference/images --weights weights/epoch-116.pth

python tools/demo.py --source F:/CEX_EXT/Map/data/real_image/images

## TOOL
### Tensorboard
tensorboard --logdir=runs