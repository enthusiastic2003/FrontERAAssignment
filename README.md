# Audio Classification with Wav2Vec2 and YAMNet

This project involves fine-tuning the Wav2Vec2 and YAMNet models for audio classification tasks using custom datasets. Follow the steps below to set up the environment, download datasets, and train the models.

## Installation

### 1. Install FFmpeg

First, you need to install **FFmpeg** to handle audio processing. Use the following command to install it:

```bash
# On Ubuntu/Debian:
sudo apt-get install ffmpeg

# On macOS:
brew install ffmpeg
```

### 2. Install Python Requirements

Next, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset Download and Preparation

### 1. Download Datasets

To download and prepare the datasets, run the following commands:

```bash
cp ./DataCollection/* ./
python3 google_datasets_download.py
python3 download_datasets.py
python3 convert_to_required_format.py
python3 csv_creator.py
```

## Training the Models

Once the datasets are prepared, you can fine-tune the models by running the following Jupyter notebooks:

- `FineTune_Wav2Vec2.ipynb`
- `FineTuning_yamnetModel.ipynb`

These notebooks guide you through the fine-tuning process for both Wav2Vec2 and YAMNet models using the prepared datasets.
