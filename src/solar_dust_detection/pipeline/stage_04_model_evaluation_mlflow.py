from solar_dust_detection.config.configuration import ConfigurationManager
from solar_dust_detection import logger
from solar_dust_detection.components.model_evaluation_mlflow import Evaluation



STAGE = "Evaluation Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass 
    
    def main(self):
        config = ConfigurationManager()
        evaluation_config = config.get_evaluation_config()
        evaluation = Evaluation(config=evaluation_config)
        evaluation.evaluation()
        #evaluation.log_into_mlflow()





            
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE} started <<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE} completed <<<<<\n\nx================x")
    except Exception as e:
        logger.exception(e)
        raise e



