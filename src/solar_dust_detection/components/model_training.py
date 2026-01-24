# Components update
import os
import urllib.request as request
from zipfile import ZipFile
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets, models
from solar_dust_detection import logger
import time

class MapDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.dataset)

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def get_base_model(self):
        self.model = models.resnet18(weights=None)
        
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, self.config.params_classes)
        
        checkpoint = torch.load(self.config.updated_base_model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        
        # 4. Move to GPU/CPU
        self.model.to(self.device)
        
    def train_valid_generator(self):
        
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_transforms = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]), # Resize to (224, 224)
            transforms.ToTensor(),
            normalize
        ])

        if self.config.params_is_augmentation:
            train_transforms = transforms.Compose([
                transforms.Resize(self.config.params_image_size[:-1]),
                transforms.RandomRotation(40),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
                transforms.ToTensor(),
                normalize
            ])
        else:
            train_transforms = val_transforms       

        # ImageFolder expects structure: data/class_a/img1.jpg, data/class_b/img2.jpg
        full_dataset = datasets.ImageFolder(root=self.config.training_data)
        
        val_size = int(len(full_dataset) * 0.20)
        train_size = len(full_dataset) - val_size
        
        train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

        train_dataset = MapDataset(train_subset, train_transforms)
        val_dataset = MapDataset(val_subset, val_transforms)

        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.params_batch_size, 
            shuffle=True, 
            num_workers=0 
        )
        
        self.valid_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.params_batch_size, 
            shuffle=False,
            num_workers=0
        )

    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)

    def train(self):
        # 1. Define Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.params_learning_rate)

        print(f"Training on {self.device} with {len(self.train_loader.dataset)} samples.")

        # 2. The Training Loop 
        for epoch in range(self.config.params_epochs):
            self.model.train() 
            running_loss = 0.0
            correct = 0
            total = 0

            # --- Batch Loop ---
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # Zero gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Metrics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # --- Epoch Metrics ---
            epoch_acc = 100 * correct / total
            logger.info(f"Epoch [{epoch+1}/{self.config.params_epochs}] "
                  f"Loss: {running_loss/len(self.train_loader):.4f} "
                  f"Acc: {epoch_acc:.2f}%")


        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )
        logger.info(f"Model saved to {self.config.trained_model_path}")