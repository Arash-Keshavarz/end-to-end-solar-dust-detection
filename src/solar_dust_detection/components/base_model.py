import os
from pathlib import Path
import torch
import torchvision.models as models
import torch.nn as nn
from solar_dust_detection import logger
from solar_dust_detection.entity.config_entity import BaseModelConfig



class BaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config
    
    def get_base_model(self):
        if self.config.params_weights == "imagenet":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        
        self.model = models.resnet18(weights=weights)
        self.save_model(path=self.config.base_model_path, model=self.model)
        logger.info(f"Base model saved at {self.config.root_dir}")

        for param in self.model.parameters():
            param.requires_grad = False
        
        num_features = self.model.fc.in_features
        
        self.model.fc = nn.Linear(num_features, self.config.params_classes)
        
        self.save_model(path=self.config.updated_base_model_path, model=self.model)
        
        logger.info(f"Updated model (classes={self.config.params_classes}) saved to {self.config.updated_base_model_path}")
    
    @staticmethod
    def save_model(path: Path, model: nn.Module):
        torch.save(model.state_dict(), path)