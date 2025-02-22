import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import  tensorflow_io as tfio
import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile
import pandas as pd


### Constants
train_split = 0.7
val_split = 0.15
test_split = 0.15
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


@tf.function
def load_wav_16k_mono(filename , label):
    """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(
          file_contents,
          desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    return wav, label


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
        dataset = dataset.map(load_wav_16k_mono, num_parallel_calls=tf.data.AUTOTUNE)

        # No batching, return samples 1 by 1
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset

