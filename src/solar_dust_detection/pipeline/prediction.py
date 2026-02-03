import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from solar_dust_detection import logger


class PredictionPipeline:
    """
    Lightweight inference wrapper.

    Notes:
    - The model is loaded once during init (better latency than loading per request).
    - Model path can be provided via MODEL_PATH env var or constructor argument.
    """

    def __init__(self, filename: str, model_path: Optional[str] = None):
        self.filename = filename
        self.device = torch.device("cpu")
        self.model_path = self._resolve_model_path(model_path)
        self.preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.model = self._load_model(self.model_path)

    def _resolve_model_path(self, model_path: Optional[str]) -> Path:
        if model_path:
            return Path(model_path)

        env_path = os.getenv("MODEL_PATH")
        if env_path:
            return Path(env_path)

        # Prefer the DVC training output; fall back to the "model/" directory.
        candidates = [
            Path("artifacts/training/model.pt"),
            Path("model/model.pt"),
        ]
        for p in candidates:
            if p.exists():
                return p

        raise FileNotFoundError(
            "No model weights found. Expected one of: "
            "'artifacts/training/model.pt' or 'model/model.pt', "
            "or set MODEL_PATH env var."
        )

    def _load_model(self, path: Path) -> nn.Module:
        logger.info("Loading model weights from: %s", path)
        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)

        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.to(self.device)
        model.eval()
        return model

    def predict(self):
        # Load and preprocess image
        image_data = Image.open(self.filename).convert("RGB")
        input_tensor = self.preprocess(image_data)
        input_batch = input_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(input_batch)
            result_index = int(torch.argmax(output, dim=1).item())

        prediction = "Dusty" if result_index == 1 else "Clean"
        return [{"image": prediction}]