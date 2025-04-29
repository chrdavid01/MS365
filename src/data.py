import os
import shutil
from urllib.parse import urlparse
from typing import Optional
import kagglehub


def process_kaggle_url(url: str) -> Optional[str]:
    """
    Process a Kaggle URL to extract the dataset path.

    Args:
        url (str): The Kaggle dataset URL to process

    Returns:
        Optional[str]: The dataset path in format 'username/dataset-name' if valid,
                       None if the URL is invalid or doesn't contain 'datasets'
    """
    parsed = urlparse(url)
    parts = parsed.path.split("/")
    if "datasets" in parts:
        index = parts.index("datasets")
        return "/".join(parts[index + 1 :])
    return None


def download_kaggle_dataset(dataset_url: str, download_path: str) -> None:
    """
    Downloads a dataset from Kaggle using kagglehub and copies to specified location

    Parameters:
        dataset_url (str): Name of dataset in format 'username/dataset-name'
        destination_folder (str): Path where to copy the downloaded files

    Raises:
        FileNotFoundError: If the dataset URL is invalid or dataset is not found.
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    if os.listdir(download_path):
        print(
            f"{download_path} contains data. Delete the file(s) if you want to download again."
        )
        return

    dataset = process_kaggle_url(dataset_url)
    if not dataset:
        print("Invalid Kaggle dataset URL")
        raise FileNotFoundError("Invalid Kaggle dataset URL")

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
