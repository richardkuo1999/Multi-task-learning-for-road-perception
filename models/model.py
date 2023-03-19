import torch
from pathlib import Path

def build_model(cfg, nc, anchors=None, ch=3,):
    if len(cfg.split('.')) > 1:
        if(cfg.split('.')[-1] == 'yaml' and Path(cfg).exists):
           from models.YOLOP import Model
        else:
            raise Exception(f'{cfg} not exist')

    elif Path('models', cfg+'.py').exists:
        if cfg == 'UNext':
          from models.UNext import Model
        elif cfg == 'Newmodel':
          from models.Newmodel import Model
        else:
            raise Exception(f'mmodel {cfg} not exist')
    model = Model(cfg, nc, anchors)
    print(parameter_count_table(model))
    return model

def get_optimizer(hyp, model):
    if hyp['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),lr=hyp['lr0'],
                                momentum=hyp['momentum'], weight_decay=hyp['wd'],
                                nesterov=hyp['nesterov'])
    elif hyp['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),lr=hyp['lr0'],
                                                betas=(hyp['momentum'], 0.999))   
    return optimizer