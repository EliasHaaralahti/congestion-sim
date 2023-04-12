"""Script containing a class for creating the congestion detection model."""
import copy
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import alexnet, AlexNet_Weights
from torchvision import transforms
from PIL import Image

class CongestionDetector:
    """
    Class that represents the pre-trained AlexNet model that is fine-tuned to detect if
    an image of traffic (taken by RSU) is congested or not.
    """
    def __init__(self, use_gpu: bool=False):
        self.n_classes = 2
        self.classes = ['congested', 'not_congested']
        self.device = torch.device('cuda:0' if use_gpu else 'cpu')
        self.model = alexnet(weights=AlexNet_Weights.DEFAULT)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, self.n_classes)
        self.model.classifier.add_module('7', nn.LogSoftmax(dim=1))
        if use_gpu:
            self.model.cuda()

    def train(self, criterion: nn.Module, optimizer: nn.Module, n_epochs: int,
              train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
        """
        Train the binary classifier. Early stopping is used to save the model with
        the highest validation accuracy.
        """
        start = time.time()
        train_data_size = len(train_dataloader.dataset)
        val_data_size = len(val_dataloader.dataset)
        best_val_accuracy = 0
        best_model_weights = copy.deepcopy(self.model.state_dict())
        for epoch in range(n_epochs):
            epoch_start = time.time()
            print(f'Epoch: {epoch+1}/{n_epochs}')
            self.model.train()
            train_loss = 0
            train_accuracy = 0
            val_loss = 0
            val_accuracy = 0
            for inputs, labels in train_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs.data, 1)
                correct_counts = preds.eq(labels.data.view_as(preds))
                accuracy = torch.mean(correct_counts.type(torch.FloatTensor))
                train_accuracy += accuracy.item() * inputs.size(0)
            with torch.no_grad():
                self.model.eval()
                for inputs, labels in val_dataloader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += accuracy.item() * inputs.size(0)
                    _, preds = torch.max(outputs.data, 1)
                    correct_counts = preds.eq(labels.data.view_as(preds))
                    accuracy = torch.mean(correct_counts.type(torch.FloatTensor))
                    val_accuracy += accuracy.item() * inputs.size(0)
            avg_train_loss = train_loss / train_data_size
            avg_train_accuracy = train_accuracy / train_data_size
            print(f'Training loss: {avg_train_loss}, accuracy: {avg_train_accuracy}')
            avg_val_loss = val_loss / val_data_size
            avg_val_accuracy = val_accuracy / val_data_size
            print(f'Validation loss: {avg_val_loss}, accuracy: {avg_val_accuracy}')
            if avg_val_accuracy > best_val_accuracy:
                best_val_accuracy = avg_val_accuracy
                best_model_weights = copy.deepcopy(self.model.state_dict())
            epoch_time = time.time() - epoch_start
            print(f'Epoch finished in {epoch_time // 60} minutes, {epoch_time % 60} seconds')
        self.model.load_state_dict(best_model_weights)
        training_time = time.time() - start
        print(f'Training finished in {training_time // 60} minutes, {training_time % 60} seconds')
        print(f'Best validation accuracy: {best_val_accuracy}')

    def predict(self, img: Image) -> dict:
        """Make predictions using the model. Returns a dictionary with the class probabilities."""
        img_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        img_tensor = img_transform(img).view(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            self.model.eval()
            output = self.model(img_tensor)
            exp_output = torch.exp(output)
            prediction = {}
            prediction['congested'] = exp_output.cpu().numpy()[0][0]
            prediction['not_congested'] = exp_output.cpu().numpy()[0][1]
            return prediction

    def evaluate(self, test_dataloader: DataLoader) -> None:
        """Evaluate the model using a separate test set."""
        test_data_size = len(test_dataloader.dataset)
        test_accuracy = 0
        with torch.no_grad():
            self.model.eval()
            for inputs, labels in test_dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = torch.max(outputs.data, 1)
                correct_counts = preds.eq(labels.data.view_as(preds))
                accuracy = torch.mean(correct_counts.type(torch.FloatTensor))
                test_accuracy += accuracy.item() * inputs.size(0)
        print(f'Test accuracy: {test_accuracy / test_data_size}')

    def save_weights(self, filename: str) -> None:
        """Save model weights."""
        torch.save(self.model.state_dict(), filename)

    def load_weights(self, filename: str) -> None:
        """Load model weights."""
        self.model.load_state_dict(torch.load(filename))
