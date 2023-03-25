import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CongestionDetector

img_transform = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

PATH = 'data'

test_directory = os.path.join(PATH, 'test')

test_data = datasets.ImageFolder(root=test_directory, transform=img_transform)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

detector = CongestionDetector()
detector.load_weights('models/model.pt')

detector.evaluate(test_dataloader=test_dataloader)
