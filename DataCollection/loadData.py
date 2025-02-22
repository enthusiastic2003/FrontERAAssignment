import kagglehub
import os
import shutil

# Define dataset
InfantCryBaby = "warcoder/infant-cry-audio-corpus"

# Download dataset
path = kagglehub.dataset_download(InfantCryBaby)
print("Infant Cry Baby data downloaded to:", path)

# Set environment variable
os.environ["INFANT_CRY_DATA_LOCATION"] = path

# Get current working directory
current_dir = os.getcwd()

tempData = "./tempData"
if not os.path.exists(tempData):
    os.makedirs(tempData)

pasteData = os.path.join(current_dir, tempData)

# Define destination path in current directory
destination_path = os.path.join(pasteData, os.path.basename(path))

# Copy dataset to current directory
if not os.path.exists(destination_path):
    shutil.copytree(path, destination_path)
    print(f"Dataset copied to: {destination_path}")
else:
    print(f"Dataset already exists in: {destination_path}")


################################################################

from git import Repo
import os

# Repository URL
repo_url = "https://github.com/QingyuLiu0521/ICSD.git"

# Clone to current working directory
ICSDCryPath = os.path.join(os.getcwd(), "tempData/ICSD")

# Clone the repo
if not os.path.exists(ICSDCryPath):
    print(f"Cloning into {ICSDCryPath}...")
    Repo.clone_from(repo_url, ICSDCryPath)
    print("Repository cloned successfully.")
else:
    print(f"Repository already exists at {ICSDCryPath}.")

# Set environment variable
os.environ["ICSD_CRY_DATA_LOCATION"] = ICSDCryPath

##################################################################

import requests
import os
import shutil

# URL of the ZIP file
url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hbppd883sd-1.zip"

# Directory to save the file
save_dir = "tempData"
os.makedirs(save_dir, exist_ok=True)

# File path to save the ZIP
zip_path = os.path.join(save_dir, "hbppd883sd-1.zip")

# Download the file
print(f"Downloading file from {url}...")
response = requests.get(url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    with open(zip_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print(f"File downloaded and saved to {zip_path}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")


# Unzip the file
import zipfile

# Directory to extract the files
extract_dir = os.path.join(save_dir, "./MendelyCryData")

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)


# Extract the ZIP file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)
    print(f"Files extracted to {extract_dir}")


############################################################

import requests
import os
import shutil

# URL of the .tar.gz file
url = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz"

# Directory to save the file
save_dir = "./tempData"
os.makedirs(save_dir, exist_ok=True)

# File path to save the .tar.gz
tar_gz_path = os.path.join(save_dir, "features.tar.gz")

# Download the file
print(f"Downloading file from {url}...")
#response = requests.get(url, stream=True)

# Check if the request was successful
if response.status_code == 200:
    with open(tar_gz_path, 'wb') as f:
        shutil.copyfileobj(response.raw, f)
    print(f"File downloaded and saved to {tar_gz_path}")
else:
    print(f"Failed to download file. Status code: {response.status_code}")

# Extract the .tar.gz file
import tar
import gzip

# Directory to extract the files

extract_dir = os.path.join(save_dir, "./tempData/GoogleCry")

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir)

# Extract the .tar.gz file
with tar.open(tar_gz_path, 'r:gz') as tar_ref:
    tar_ref.extractall(extract_dir)
    print(f"Files extracted to {extract_dir}")

