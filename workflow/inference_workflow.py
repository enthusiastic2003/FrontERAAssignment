import numpy as np
import torch
from w2v2 import Wav2Vec2Manager
from ymnet import YAMNetManager
import soundfile as sf

class AudioClassifier:
    def __init__(self, w2v2_path='w2v2_nn.pth', yamnet_path='model_yamnet_final.h5'):
        self.w2v2_manager = Wav2Vec2Manager(classifier_path=w2v2_path)
        self.yamnet_manager = YAMNetManager(h5_model_path=yamnet_path)

    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))
        return audio_data

    def classify(self, audio_data: np.ndarray, sample_rate: int) -> str:
        audio_data = self.preprocess_audio(audio_data)
        
        # Convert audio to tensor for Wav2Vec2
        audio_data = np.mean(audio_data, axis=1)
        waveform = torch.from_numpy(audio_data).float()

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        # Wav2Vec2 predictions
        w2v2_results = self.w2v2_manager.get_predictions(waveform, sample_rate=sample_rate)
        w2v2_preds = w2v2_results['predictions'][0]

        # YAMNet predictions
        yamnet_results = self.yamnet_manager.get_predictions(audio_data, sample_rate=sample_rate)
        yamnet_preds = yamnet_results['predictions'][0]

        # Combine predictions
        combined_predictions = {
            f"class_{i}": (w2v2_preds[i] + yamnet_preds[i]) / 2
            for i in range(len(w2v2_preds))
        }

        # Get class with highest confidence
        best_class = max(combined_predictions, key=combined_predictions.get)
        confidence = combined_predictions[best_class]

        return f"Predicted Class: {best_class}, Confidence: {confidence:.2f}"

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python audio_classify.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]
    audio_data, sample_rate = sf.read(audio_file)

    classifier = AudioClassifier()
    result = classifier.classify(audio_data, sample_rate)
    print(result)
