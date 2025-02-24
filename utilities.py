import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import  tensorflow_io as tfio
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import pandas as pd
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

### Constants
train_split = 0.8
val_split = 0.1
test_split = 0.1
batch_size = 1
num_epochs = 10
learning_rate = 0.001
num_classes = 3
model_path = "models/model.h5"
model_name = "model.h5"
model_dir = "models"
data_dir = "dataClassified"
data_csv = "dataClassified/data.csv"
cry_loc = "dataClassified/cry_data"
scream_loc = "dataClassified/scream_data"
control_loc = "dataClassified/control_data"
###


augmentor = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_shift=-0.5, max_shift=0.5, p=0.5),
    ])

@tf.function
def load_wav_16k_mono(filename, label, augment=False):
    """Load a WAV file, convert to float tensor, resample to 16 kHz, apply augmentation if needed."""
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)  # Remove extra dimension
    wav.set_shape([None])  # Ensure the shape is defined

    if augment:
        # Apply augmentation using tf.py_function
        wav = tf.py_function(func=apply_augmentation, inp=[wav], Tout=tf.float32)
        wav.set_shape([None])  # Set shape again after py_function

    return wav, label

def apply_augmentation(wav):
    # Ensure the input is a NumPy array
    wav_np = wav.numpy() if isinstance(wav, tf.Tensor) else wav

    # Apply audiomentations augmentor
    augmented = augmentor(samples=wav_np, sample_rate=16000)

    return augmented.astype(np.float32)





import tensorflow as tf
import pandas as pd

class CustomDataset(tf.data.Dataset):
    def __new__(cls, csv_file, prepend_path, data_type="train", batch_size=batch_size, train_split=train_split, val_split=val_split):
        # Load CSV
        df = pd.read_csv(csv_file)

        # Split dataset
        if data_type == "train":
            df = df.iloc[:int(len(df) * train_split)]
        elif data_type == "val":
            df = df.iloc[int(len(df) * train_split):int(len(df) * (train_split + val_split))]
        elif data_type == "test":
            df = df.iloc[int(len(df) * (train_split + val_split)):]
        else:
            raise ValueError("type must be train, val, or test")

        file_paths = prepend_path + df["file_path"].values
        labels = df["label"].values

        # Create dataset
        #dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

        # Map loading function but pass in augment=False if this is the test set or validation set 
        augment = True if data_type == "train" else False
        dataset = dataset.map(lambda x, y: load_wav_16k_mono(x, y, augment), num_parallel_calls=tf.data.AUTOTUNE)

        # No batching, return samples 1 by 1
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

