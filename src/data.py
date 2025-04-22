import os
import shutil
from urllib.parse import urlparse
import kagglehub


def process_url(url: str):
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    if "datasets" in parts:
        index = parts.index("datasets")
        return "/".join(parts[index + 1 :])
    return None


def download_kaggle_dataset(dataset_url, destination_folder):
    """
    Downloads a dataset from Kaggle using kagglehub and optionally copies to specified location

    Parameters:
    dataset_name (str): Name of dataset in format 'username/dataset-name'
    download_path (str): Path where to copy the downloaded files

    Returns:
    tuple: (cache_path, list of files)
    """
    download_path = os.path.join(".", "data", destination_folder)
    dataset = process_url(dataset_url)
    if not dataset:
        print("Invalid Kaggle dataset URL")
        raise FileNotFoundError("Invalid Kaggle dataset URL")
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    try:
        cache_path = kagglehub.dataset_download(dataset)

        if cache_path:
            print(f"Dataset cached at: {cache_path}")
            
            downloaded_files = []
            for root, _, files in os.walk(cache_path):
                for file in files:
                    
                    file_path = os.path.join(root, file)
                    downloaded_files.append(file_path)
                    
                    dest_path = os.path.join(download_path, file)
                    shutil.copy2(file_path, dest_path)
                    print(f"Copied {file} to {download_path}")
        else:
            print("Dataset not found")
            raise FileNotFoundError("Dataset not found")

    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
