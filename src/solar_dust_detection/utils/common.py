import os
from box.exceptions import BoxValueError
import yaml
from solar_dust_detection import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file.

    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e from e  
    
@ensure_annotations
def create_directories(path_to_directories: list[Path], verbose=True) -> None:
    """Creates directories if they do not exist.

    Args:
        path_to_directories (list[Path]): List of directory paths to create.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")
            

@ensure_annotations
def save_json(path: Path, data: dict[str, Any]) -> None:
    """Saves a dictionary as a JSON file.

    Args:
        path (Path): The path to the JSON file.
        data (dict[str, Any]): The data to save.
    """
    with open(path, "w") as json_file:
        json.dump(data, json_file, indent=4)
    logger.info(f"JSON file saved at: {path}")
    
    
@ensure_annotations
def load_json(path: Path) -> dict[str, Any]:
    """Loads a JSON file and returns its contents as a dictionary.

    Args:
        path (Path): The path to the JSON file.

    Returns:
        dict[str, Any]: The contents of the JSON file as a dictionary.
    """
    with open(path, "r") as json_file:
        content = json.load(json_file)
    logger.info(f"JSON file loaded from: {path}")
    
    return ConfigBox(content)

@ensure_annotations
def save_bin(data: Any, path: Path) -> None:
    """Saves data to a binary file using joblib.

    Args:
        data (Any): The data to save.
        path (Path): The path to the binary file.
    """
    joblib.dump(data, path)
    logger.info(f"Binary file saved at: {path}")    
    
@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads data from a binary file using joblib.

    Args:
        path (Path): The path to the binary file.

    Returns:
        Any: The data loaded from the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    
    return data 

@ensure_annotations
def get_size(path: Path) -> str:
    """Gets the size of a file in kilobytes (KB).

    Args:
        path (Path): The path to the file.

    Returns:
        str: The size of the file in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    logger.info(f"File size for {path} is {size_in_kb} KB")
    return f"{size_in_kb} KB"   

def encodeImageIntoBase64(image_path: Path) -> str:
    """Encodes an image file into a base64 string.

    Args:
        image_path (Path): The path to the image file.

    Returns:
        str: The base64 encoded string of the image.
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    logger.info(f"Image at {image_path} encoded into base64 string")
    return encoded_string

def decodeBase64ToImage(base64_string: str, output_path: Path) -> None:
    """Decodes a base64 string back into an image file.

    Args:
        base64_string (str): The base64 encoded string of the image.
        output_path (Path): The path to save the decoded image file.
    """
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(base64_string))
        image_file.close()
    logger.info(f"Base64 string decoded and saved as image at {output_path}")