# Multi-task-learning-for-road-perception

## Datasets
  - Panoramic Perception: [BDD100k](https://www.vis.xyz/bdd100k/)
  - Partial Perception: [YOLOP](https://github.com/hustvl/YOLOP)

## Requirement
  This codebase has been developed with
  ```
    Python 3.9
    Cuda 12
    Pytorch 2.0.1
  ```
  See requirements.txt for additional dependencies and version requirements.
  ```shell
    pip install -r requirements.txt
  ```

## main command
  You can change the data use, Path, and Classes, and merge some classes from [here](/data).
  You can revise the network from [here](/cfg).

  ### Train
  ```shell
  python train.py
  ```
  ### Test
  ```shell
  python test.py
  ```
  ### Predict
  ```shell
  python demo.py
  ```
  ### Tensorboard
  ```shell
    tensorboard --logdir=runs
  ```

## Argument
  ### Train
| Source           |   Argument                  |     Type    | Notes                                                                        |
| :---             |    :----:                   |     :----:  |   ---:                                                                       |
| hyp              | 'hyp/hyp.scratch.yolop.yaml'| str         | hyperparameter path                                                          |
| cfg              | 'cfg/YOLOP_v7b3.yaml'       | str         | model yaml path                                                              |
| data             | 'data/multi.yaml'           | str         | dataset yaml path                                                            |
| logDir           | 'runs/train'                | str         | log directory                                                                |
| resume           | ''                          | str         | Resume the weight                                                            |
| epochs           | 500                         | int         | number of epochs to train for                                                |
| val_start        | 20                          | int         | start do validation                                                          |
| val_freq         | 5                           | int         | How many epochs do one time validation                                       |
| train_batch_size | 5                           | int         | train batch size                                                             |
| test_batch_size  | 5                           | int         | test batch size                                                              |
| workers          | 6                           | int         | maximum number of dataloader workers                                         |
| device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         |
  ### Test
  | Source           |   Argument                  |     Type    | Notes                                                                        |
  | :---             |    :----:                   |     :----:  |   ---:                                                                       |
  | hyp              | 'hyp/hyp.scratch.yolop.yaml'| str         | hyperparameter path                                                          |
  | cfg              | 'cfg/YOLOP_v7b3.yaml'       | str         | model yaml path                                                              |
  | weights          | './weights/epoch-200.pth'   | str         | model.pth path(s)                                                            |
  | data             | 'data/multi.yaml'           | str         | dataset yaml path                                                            |
  | logDir           | 'runs/train'                | str         | log directory                                                                |
  | test_batch_size  | 5                           | int         | test batch size                                                              |
  | device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         |
  ### Predict
  | Source           |   Argument                  |     Type    | Notes                                                                        |
  | :---             |    :----:                   |     :----:  |   ---:                                                                       |
  | hyp              | 'hyp/hyp.scratch.yolop.yaml'| str         | hyperparameter path                                                          |
  | cfg              | 'cfg/YOLOP_v7b3.yaml'       | str         | model yaml path                                                              |
  | weights          | './weights/epoch-200.pth'   | str         | model.pth path(s)                                                            |
  | data             | 'data/multi.yaml'           | str         | dataset yaml path                                                            |
  | source           | './inference/36_GH015559'   | str         | inference file path                                                          |
  | logDir           | 'runs/train'                | str         | log directory                                                                |
  | test_batch_size  | 5                           | int         | test batch size                                                              |
  | device           | ''                          | None        | device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu         |
  | draw             | 'True'                      | bool        | save the Predict result                                                      |
  | savelabel        | 'True'                      | bool        | save the Predict result as label                                             |