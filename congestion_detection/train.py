import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CongestionDetector

detector = CongestionDetector(use_gpu=True)

criterion = nn.NLLLoss()
optimizer = optim.Adam(detector.model.parameters())

img_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

PATH = 'data'

train_directory = os.path.join(PATH, 'train')
val_directory = os.path.join(PATH, 'val')

data = {
    'train': datasets.ImageFolder(root=train_directory, transform=img_transforms['train']),
    'val': datasets.ImageFolder(root=val_directory, transform=img_transforms['val'])
}

train_dataloader = DataLoader(data['train'], batch_size=32, shuffle=True)
val_dataloader = DataLoader(data['val'], batch_size=32, shuffle=True)

detector.train(criterion=criterion, optimizer=optimizer, n_epochs=5,
               train_dataloader=train_dataloader, val_dataloader=val_dataloader)

detector.save_weights('models/model.pt')
