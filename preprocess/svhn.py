import torch
from torchvision import datasets, transforms
import config

def get_svhn(train):

    pre_process = transforms.Compose([transforms.Resize(28),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])
    if train:
        svhn_dataset = datasets.SVHN(root=config.data_root, split='train', transform=pre_process, download=True)
    else:
        svhn_dataset = datasets.SVHN(root=config.data_root, split='test', transform=pre_process, download=True)

    svhn_data_loader = torch.utils.data.DataLoader(dataset=svhn_dataset, batch_size=config.batch_size, shuffle=True)

    return svhn_data_loader

