import os
import requests
from datasets import load_dataset
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the data directory if it doesn't exist
data_dir = "data"
Path(data_dir).mkdir(exist_ok=True)

# Load the dataset
try:
    dataset = load_dataset("yandex/alchemist", split="train")
except Exception as e:
    logger.error(f"Failed to load dataset: {e}")
    exit(1)

# Iterate through the dataset
for item in dataset:
    img_key = item["img_key"]
    url = item["url"]
    
    # Define the file path using img_key
    file_name = f"{img_key}.jpg"  # Assuming images are JPEG; adjust if needed
    file_path = os.path.join(data_dir, file_name)
    
    # Skip if file already exists
    if os.path.exists(file_path):
        logger.info(f"File {file_name} already exists, skipping.")
        continue
    
    # Download the image
    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Successfully downloaded {file_name}")
        else:
            logger.warning(f"Failed to download {url} for {file_name}, status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Error downloading {url} for {file_name}: {e}")

print("Download complete!")

