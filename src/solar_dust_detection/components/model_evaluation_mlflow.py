import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, datasets, models
from pathlib import Path
import mlflow
import mlflow.pytorch
from urllib.parse import urlparse
from dotenv import load_dotenv
from solar_dust_detection.entity.config_entity import EvaluationConfig
from solar_dust_detection.utils.common import save_json
from solar_dust_detection import logger

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

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    def _load_env(self):
        for parent in Path(__file__).resolve().parents:
            env_path = parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                return
        load_dotenv()

    def _valid_generator(self):
        # 1. Define Transforms
        # Standard ImageNet normalization
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_transforms = transforms.Compose([
            transforms.Resize(self.config.params_image_size[:-1]), # Resize to (224, 224)
            transforms.ToTensor(),
            normalize
        ])

        # 2. Load Data
        full_dataset = datasets.ImageFolder(root=self.config.training_data)
        
        # 3. Create Split
        torch.manual_seed(42) 
        
        val_size = int(len(full_dataset) * 0.30)
        train_size = len(full_dataset) - val_size
        
        _, val_subset = random_split(full_dataset, [train_size, val_size])

        # 4. Apply Transforms
        val_dataset = MapDataset(val_subset, val_transforms)

        # 5. Create Loader
        self.valid_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.params_batch_size, 
            shuffle=False,
            num_workers=0 
        )

    def load_model(self, path: Path) -> nn.Module:

        model = models.resnet18(weights=None)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, self.config.params_classes) 
        
        # Load weights
        checkpoint = torch.load(path, map_location=self.device)
        model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval() 
        return model

    def evaluation(self):
        self.model = self.load_model(self.config.path_of_model)
        self._valid_generator()
        
        criterion = nn.CrossEntropyLoss()
        
        running_loss = 0.0
        correct = 0
        total = 0

        # No gradient needed for evaluation
        with torch.no_grad():
            for images, labels in self.valid_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate averages
        avg_loss = running_loss / len(self.valid_loader)
        avg_acc = correct / total
        
        self.score = [avg_loss, avg_acc]
        self.save_score()

    def save_score(self):
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)
        logger.info(f"Scores saved: {scores}")

    def log_into_mlflow(self):
        self._load_env()
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", self.config.mlflow_uri)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)

        if not os.getenv("MLFLOW_TRACKING_USERNAME"):
            dagshub_user = os.getenv("DAGSHUB_USER")
            if dagshub_user:
                os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_user

        if not os.getenv("MLFLOW_TRACKING_PASSWORD"):
            dagshub_token = os.getenv("DAGSHUB_TOKEN")
            if dagshub_token:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        if tracking_uri.startswith("https://dagshub.com"):
            if not os.getenv("MLFLOW_TRACKING_USERNAME") or not os.getenv("MLFLOW_TRACKING_PASSWORD"):
                raise RuntimeError(
                    "DagsHub MLflow requires credentials. "
                    "Set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD "
                    "or DAGSHUB_USER and DAGSHUB_TOKEN in your .env file."
                )

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"loss": self.score[0], "accuracy": self.score[1]}
            )
            
            # Switch to mlflow.pytorch
            if tracking_url_type_store != "file":
                mlflow.pytorch.log_model(self.model, "model", registered_model_name="ResNet18Model")
            else:
                mlflow.pytorch.log_model(self.model, "model")
