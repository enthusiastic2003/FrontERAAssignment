# Installation
Install ffmpeg and install python requirements via:

> pip install -r requirements.txt

# Download Datasets via:
> cp ./DataCollection/* ./
> python3 google_datasets_download.py
> python3 download_datasets.py
> python3 convert_to_required_format.py
> python3 csv_creator.py

Next, you may train the models again by running the jupyter notebooks : FineTune_Wav2Vec2.ipynb and FineTuning_yamnetModel.ipynb


