import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# -----------------------------
# 1. Feature Extraction Setup
# -----------------------------
prepend_path = "./dataClassified/"
# Load wav2vec 2.0 pretrained model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to("cuda")
wav2vec_model.eval()

# Audio Augmentation Pipeline
augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

# -----------------------------
# 2. Custom PyTorch Dataset
# -----------------------------

class SoundDatasetFromCSV(Dataset):
    def __init__(self, csv_file, augment_audio=False):
        self.data = pd.read_csv(csv_file)
        self.augment_audio = augment_audio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = prepend_path + self.data.iloc[idx]['file_path']
        label = int(self.data.iloc[idx]['label'])

        # Load and preprocess audio
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(waveform)
        waveform = waveform.squeeze()

        # Apply augmentation
        if self.augment_audio:
            waveform = augment(samples=waveform.numpy(), sample_rate=16000)
            waveform = torch.tensor(waveform)

        waveform.to("cuda")
        # Preprocess for wav2vec2
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs.to("cuda")
        # Extract embeddings
        with torch.no_grad():
            outputs = wav2vec_model(**inputs)
            embeddings = outputs.last_hidden_state  # (batch_size, seq_length, hidden_size)

        # Mean pooling
        pooled_embedding = torch.mean(embeddings, dim=1).squeeze()  # (hidden_size,)

        return pooled_embedding.to("cpu").numpy(), label
