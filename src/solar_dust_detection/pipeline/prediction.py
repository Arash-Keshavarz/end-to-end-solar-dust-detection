import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

        #  Load Model Architecture
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2) 

        # Load Weights
        model_path = os.path.join("artifacts", "training", "model.pt") 
        
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        
        model.eval() 

        #  Load and Preprocess Image
        imagename = self.filename
        image_data = Image.open(imagename).convert('RGB')
        
        input_tensor = preprocess(image_data)
        input_batch = input_tensor.unsqueeze(0) 

        # Prediction
        with torch.no_grad():
            output = model(input_batch)
            result_index = torch.argmax(output, dim=1).item()
            print(f"Predicted Class Index: {result_index}")

        if result_index == 1:
            prediction = 'Dusty'
            return [{ "image" : prediction}]
        else:
            prediction = 'Clean'
            return [{ "image" : prediction}]