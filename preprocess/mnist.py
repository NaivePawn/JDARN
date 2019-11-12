import torch
from torchvision import datasets, transforms
import config

def get_mnist(train):

    pre_process = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                      ])
    mnist_dataset = datasets.MNIST(root=config.data_root, train=train, transform=pre_process, download=True)

    mnist_data_loader = torch.utils.data.DataLoader(dataset=mnist_dataset, batch_size=config.batch_size, shuffle=True)

    return mnist_data_loader
