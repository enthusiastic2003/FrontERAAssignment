import os
import shutil
import requests
import zipfile
import tarfile
import kagglehub
from typing import Optional

class DatasetDownloader:
    def __init__(self, base_dir: str = "./tempData"):
        """Initialize the downloader with a base directory for all downloads."""
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _ensure_directory(self, directory: str) -> None:
        """Create directory if it doesn't exist."""
        if not os.path.exists(directory):
            os.makedirs(directory)

    def download_file(self, url: str, filename: str) -> Optional[str]:
        """
        Download a file from URL and save it to the specified location.

        Args:
            url: URL to download from
            filename: Name to save the file as

        Returns:
            Path to downloaded file or None if download failed
        """
        filepath = os.path.join(self.base_dir, filename)
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            print(f"Downloaded file to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Failed to download file: {str(e)}")
            return None

    def extract_archive(self, archive_path: str, extract_dir: str) -> Optional[str]:
        """
        Extract a compressed archive (zip or tar.gz).

        Args:
            archive_path: Path to the archive file
            extract_dir: Directory to extract to

        Returns:
            Path to extracted directory or None if extraction failed
        """
        try:
            full_extract_path = os.path.join(self.base_dir, extract_dir)
            self._ensure_directory(full_extract_path)

            if archive_path.endswith('.zip'):
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(full_extract_path)
            elif archive_path.endswith('.tar.gz'):
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(full_extract_path)

            print(f"Extracted to: {full_extract_path}")
            return full_extract_path
        except Exception as e:
            print(f"Failed to extract archive: {str(e)}")
            return None

    def download_kaggle_dataset(self, dataset_name: str, local_dir: str) -> Optional[str]:
        """
        Download a dataset from Kaggle and copy it to local directory.

        Args:
            dataset_name: Name of the Kaggle dataset
            local_dir: Local directory name to copy to

        Returns:
            Path to local copy of dataset or None if failed
        """
        try:
            # Download from Kaggle
            kaggle_path = kagglehub.dataset_download(dataset_name)

            # Set up local path
            destination_path = os.path.join(self.base_dir, local_dir)


            # Copy to local directory if it doesn't exist
            if not os.path.exists(destination_path):
                shutil.copytree(kaggle_path, destination_path)
                print(f"Kaggle dataset copied to: {destination_path}")
            else:
                print(f"Dataset already exists in: {destination_path}")

            return destination_path
        except Exception as e:
            print(f"Failed to download Kaggle dataset: {str(e)}")
            return None

def main():
    """Main function to demonstrate usage of the DatasetDownloader class."""
    downloader = DatasetDownloader()
    #
    # # Download and process Infant Cry dataset from Kaggle
    # infant_cry_path = downloader.download_kaggle_dataset(
    #     "warcoder/infant-cry-audio-corpus",
    #     "InfantCryData"
    # )
    #
    # # Download and process Mendely dataset
    # mendely_url = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/hbppd883sd-1.zip"
    # mendely_zip = downloader.download_file(mendely_url, "mendely_cry.zip")
    # if mendely_zip:
    #     downloader.extract_archive(mendely_zip, "MendelyCryData")
    #
    # ## NOW DOWNLOAD SCREAM DATASETS
    #
    # kaggleHumanScream = downloader.download_kaggle_dataset(
    #     "whats2000/human-screaming-detection-dataset",
    #     "KaggleHumanScreamData"
    # )
    #
    # # Copy /home/sirjanh/Downloads/cv-corpus-20.0-delta-2024-12-06-en.tar.gz to tempData
    # # Dyanmically set the path to the downloads folder
    #
    userDownloads = os.path.expanduser("~") + "/Downloads"

    cv_corpus_url = f"{userDownloads}/cv-corpus-20.0-delta-2024-12-06-en.tar.gz"
    cv_corpus_tar = cv_corpus_url
    if cv_corpus_tar:
        downloader.extract_archive(cv_corpus_tar, "MozillaControlData")
    
    libriSpeech_url = "https://us.openslr.org/resources/12/test-clean.tar.gz"
    libriSpeech_tar = downloader.download_file(libriSpeech_url, "test-clean.tar.gz")
    if libriSpeech_tar:
         downloader.extract_archive(libriSpeech_tar, "LibriControlData")

if __name__ == "__main__":
    main()
