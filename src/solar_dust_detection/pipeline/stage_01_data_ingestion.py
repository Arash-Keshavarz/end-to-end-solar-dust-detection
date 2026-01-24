from solar_dust_detection.config.configuration import ConfigurationManager
from solar_dust_detection import logger
from solar_dust_detection.components.data_ingestion import DataIngestion



STAGE = "Data Ingestion Stage" 

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass 
    
    def main(self):
        
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_data()
        data_ingestion.extract_zip_file()

        
        
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>> stage {STAGE} started <<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>> stage {STAGE} completed <<<<<\n\nx================x")
    except Exception as e:
        logger.exception(e)
        raise e