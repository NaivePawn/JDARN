import torch
from torchvision import transforms, datasets


def load_images(root_path, dir, batch_size, is_train=True, resize_size=256, crop_size=224):
    if is_train:
        transformer = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transformer = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    images = datasets.ImageFolder(root=root_path + dir, transform=transformer)
    images_loader = torch.utils.data.DataLoader(images,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                # drop_last=True,
                                                num_workers=4)

    return images_loader