from solar_dust_detection.config.configuration import ConfigurationManager
from solar_dust_detection import logger
from solar_dust_detection.components.model_training import Training



STAGE = "Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass 
    
    def main(self):
        
        config = ConfigurationManager()
        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train()


            
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE} started <<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE} completed <<<<<\n\nx================x")
    except Exception as e:
        logger.exception(e)
        raise e



