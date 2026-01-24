from solar_dust_detection import logger
from solar_dust_detection.pipeline.stage_01_data_ingestion import STAGE, DataIngestionTrainingPipeline
from solar_dust_detection.pipeline.stage_02_base_model import BaseModelTrainingPipeline
from solar_dust_detection.pipeline.stage_03_model_training import ModelTrainingPipeline


#----------------------------------------------------------------
STAGE_NAME = "Data Ingestion Stage" 

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e

#----------------------------------------------------------------
STAGE_NAME = "Base Model Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = BaseModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e

#----------------------------------------------------------------
STAGE_NAME = "Model Training Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<\n\nx================x")
except Exception as e:
    logger.exception(e)
    raise e


#----------------------------------------------------------------