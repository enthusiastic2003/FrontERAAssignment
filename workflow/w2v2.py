import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class Wav2Vec2Manager:
    def __init__(self, classifier_path: str):
        """
        Initialize Wav2Vec2 model manager
        
        Args:
            classifier_path (str): Path to the trained classifier model
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Wav2Vec2 model and processor
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
        self.wav2vec_model.eval()
        
        # Load the neural network classifier
        self.classifier = SimpleNN(768, 128, 3).to(self.device)
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.classifier.eval()
    
    def extract_embedding(self, waveform, sample_rate=16000):
        """
        Extract Wav2Vec2 embeddings from audio
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            sample_rate (int): Sample rate of the input audio
            
        Returns:
            torch.Tensor: Pooled embedding vector
        """
        # Resample if necessary
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=16000
            )(waveform)
        
        waveform = waveform.squeeze().to(self.device)
        
        # Preprocess for wav2vec2
        inputs = self.processor(
            waveform, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.wav2vec_model(**inputs)
            embeddings = outputs.last_hidden_state
            pooled_embedding = torch.mean(embeddings, dim=1).squeeze()
            
        return pooled_embedding
    
    def get_predictions(self, waveform, sample_rate=16000):
        """
        Get predictions from the Wav2Vec2 model
        
        Args:
            waveform (torch.Tensor): Input audio waveform
            sample_rate (int): Sample rate of the input audio
            
        Returns:
            dict: Dictionary containing predictions and embeddings
        """
        embedding = self.extract_embedding(waveform, sample_rate)
        
        with torch.no_grad():
            predictions = self.classifier(embedding.unsqueeze(0))
            probabilities = torch.softmax(predictions, dim=1)
        
        return {
            'predictions': probabilities.cpu().numpy(),
            'embedding': embedding.cpu().numpy()
        }

# Example usage:
"""
# Initialize the Wav2Vec2 manager
w2v2_manager = Wav2Vec2Manager(classifier_path='path/to/classifier.pth')

# Load and process audio
waveform, sample_rate = torchaudio.load('path/to/audio.wav')
results = w2v2_manager.get_predictions(waveform, sample_rate)
"""