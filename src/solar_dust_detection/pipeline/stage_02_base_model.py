from solar_dust_detection.config.configuration import ConfigurationManager
from solar_dust_detection import logger
from solar_dust_detection.components.base_model import BaseModel



STAGE = "Base Model Stage"

class BaseModelTrainingPipeline:
    def __init__(self):
        pass 
    
    def main(self):
        
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = BaseModel(config=base_model_config)
        base_model.get_base_model()

            
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE} started <<<<<")
        obj = BaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE} completed <<<<<\n\nx================x")
    except Exception as e:
        logger.exception(e)
        raise e