import torchvision.transforms as transforms
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

import lib.dataset as dataset

class DataLoaderX(DataLoader):
    """prefetch dataloader"""
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def create_dataloader(args, hyp, batch_size, normalize, is_train=True, shuffle=True):
    normalize = transforms.Normalize(
            normalize['mean'], normalize['std']
        )

    datasets = eval('dataset.' + args.dataset)(
        args=args,
        hyp=hyp,
        is_train=is_train,
        inputsize=args.img_size,
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
        collate_fn=dataset.AutoDriveDataset.collate_fn
    )
    return loader, datasets