from dataclasses import dataclass
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzipped_data_dir: Path
    
    
@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    
    # Model parameters
    params_image_size: List[int]
    params_learning_rate: float
    params_weights: str
    params_classes: int