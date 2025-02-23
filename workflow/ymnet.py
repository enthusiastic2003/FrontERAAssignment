import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import torch

class YAMNetManager:
    def __init__(self, h5_model_path: str):
        """
        Initialize YAMNet model manager

        Args:
            h5_model_path (str): Path to the trained H5 model
        """
        # Load YAMNet
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

        # Load H5 model
        self.classifier = tf.keras.models.load_model(h5_model_path)

    def extract_embedding(self, waveform, sample_rate=16000):
        """
        Extract YAMNet embeddings from audio

        Args:
            waveform (numpy.ndarray or torch.Tensor): Input audio waveform
            sample_rate (int): Sample rate of the input audio

        Returns:
            numpy.ndarray: YAMNet embedding
        """
        # Convert to numpy if needed
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()

        # Ensure correct shape and type
        waveform = waveform.squeeze()
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        # Resample if necessary
        if sample_rate != 16000:
            waveform = tfio.audio.resample(
                waveform,
                rate_in=sample_rate,
                rate_out=16000
            )

        # Get embeddings
        scores, embeddings, spectrogram = self.yamnet_model(waveform)
        return embeddings.numpy()

    def get_predictions(self, waveform, sample_rate=16000):
        """
        Get predictions from the YAMNet model

        Args:
            waveform (numpy.ndarray or torch.Tensor): Input audio waveform
            sample_rate (int): Sample rate of the input audio

        Returns:
            dict: Dictionary containing predictions and embeddings
        """
        embedding = self.extract_embedding(waveform, sample_rate)

        # Convert embeddings to the right shape if needed
        if len(embedding.shape) == 2:
            # If we have multiple frames, we might want to aggregate them
            embedding = np.mean(embedding, axis=0, keepdims=True)
        elif len(embedding.shape) == 1:
            # If we have a single embedding vector, add batch dimension
            embedding = np.expand_dims(embedding, 0)

        # Get predictions using the H5 model
        predictions = self.classifier.predict(embedding, verbose=0)

        return {
            'predictions': predictions,
            'embedding': embedding
        }

# Example usage:
"""
# Initialize the YAMNet manager
yamnet_manager = YAMNetManager(h5_model_path='path/to/model.h5')

# Load and process audio
waveform, sample_rate = torchaudio.load('path/to/audio.wav')
results = yamnet_manager.get_predictions(waveform, sample_rate)
"""
