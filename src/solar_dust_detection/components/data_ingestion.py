import os
import gdown
import zipfile
from solar_dust_detection import logger
from solar_dust_detection.entity.config_entity import DataIngestionConfig
from solar_dust_detection.utils.common import get_size

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> str:
        """Download data from Google Drive."""
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Created directory at: artifacts/data_ingestion")

            file_id = dataset_url.split("/")[-2]
            prefix_url = "https://drive.google.com/uc?export=download&id="
            gdown.download(prefix_url + file_id, str(zip_download_dir))
            logger.info(f"Downloading data from {dataset_url} to {zip_download_dir}")

        except Exception as e:
            logger.error(f"Error occurred while downloading data: {e}")
            raise e

    def extract_zip_file(self) -> None:
        """Extract the zip file to the specified directory."""
        unzip_path = self.config.unzipped_data_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted data to {unzip_path}")