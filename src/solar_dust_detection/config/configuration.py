from solar_dust_detection.constants import *
from solar_dust_detection.entity.config_entity import DataIngestionConfig, BaseModelConfig, TrainingConfig, EvaluationConfig
from solar_dust_detection.utils.common import read_yaml, create_directories
from pathlib import Path
import os


class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([Path(self.config.artifacts_root)])
        
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzipped_data_dir=Path(config.unzipped_data_dir),
        )
        return data_ingestion_config
    
    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model
        
        create_directories([Path(config.root_dir)])
        
        base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
        )
        
        return base_model_config
    
    def get_training_config(self) -> TrainingConfig:
        training_config = self.config.training

        root_dir = Path(training_config.root_dir)
        trained_model_path = Path(training_config.trained_model_path)
        updated_base_model_path = Path(self.config.base_model.updated_base_model_path)
        training_data = os.path.join(self.config.data_ingestion.unzipped_data_dir, "Detect_solar_dust")
        create_directories([root_dir])
        
        params_epochs = self.params.EPOCHS
        params_batch_size = self.params.BATCH_SIZE
        params_is_augmentation = self.params.AUGMENTATION
        params_image_size = self.params.IMAGE_SIZE
        params_learning_rate = self.params.LEARNING_RATE
        params_classes = self.params.CLASSES

        training_config = TrainingConfig(
            root_dir=root_dir,
            trained_model_path=trained_model_path,
            updated_base_model_path=updated_base_model_path,
            training_data=training_data,
            params_epochs=params_epochs,
            params_batch_size=params_batch_size,
            params_is_augmentation=params_is_augmentation,
            params_image_size=params_image_size,
            params_learning_rate=params_learning_rate,
            params_classes=params_classes,
        )

        return training_config
    
    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model= "artifacts/training/model.pt",
            training_data= "artifacts/data_ingestion/Detect_solar_dust",
            all_params = self.params,
            mlflow_uri= "https://dagshub.com/Arash-keshavarz/end-to-end-solar-dust-detection.mlflow",
            params_image_size = self.params.IMAGE_SIZE,
            params_batch_size = self.params.BATCH_SIZE,
            params_classes= self.params.CLASSES
        )
        return eval_config