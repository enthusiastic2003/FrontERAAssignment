import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from torch import nn
import torch.optim as optim
from sklearn.metrics import classification_report
from tqdm import tqdm


class AudioClassifier(nn.Module):
    def __init__(self, num_classes, pretrained_model="facebook/wav2vec2-base"):
        super().__init__()
        
        # Load pretrained wav2vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(pretrained_model)
        
        # Add classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),  # 768 is wav2vec2-base hidden size
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features using wav2vec2
        features = self.wav2vec2(x).last_hidden_state
        
        # Pool features (mean pooling)
        features = torch.mean(features, dim=1)
        
        # Classify
        output = self.classifier(features)
        return output
    

class AudioDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.data = pd.read_csv(csv_file)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_path = "./dataClassified/" + self.data.iloc[idx]['file_path']
        label = self.data.iloc[idx]['label']
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize the waveform
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-6)
        
        # Process with wav2vec2
        input_values = self.processor(waveform, sampling_rate=sample_rate, 
                                    padding="max_length", max_length=16000,
                                    truncation=True, return_tensors="pt").input_values
        
        return input_values.squeeze(), torch.tensor(label)

def create_data_loaders(train_df, val_df, processor, batch_size=8):
    # Create datasets
    train_dataset = AudioDataset(train_df, processor)
    val_dataset = AudioDataset(val_df, processor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * correct / total
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss/len(train_loader):.3f} | Train Acc: {train_acc:.3f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.3f} | Val Acc: {val_acc:.3f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
    
    # Create datasets
    from sklearn.model_selection import train_test_split
    
    # Read and split data
    csv_file = './dataClassified/data.csv'
    df = pd.read_csv(csv_file)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, 
                                       stratify=df['label'])
    
    # Save splits to temporary CSVs
    train_df.to_csv('train_temp.csv', index=False)
    val_df.to_csv('val_temp.csv', index=False)
    
    # Create datasets and dataloaders
    train_dataset = AudioDataset('train_temp.csv', processor)
    val_dataset = AudioDataset('val_temp.csv', processor)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2)
    
    # Initialize model
    model = AudioClassifier(num_classes=3).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device)
    
    # Evaluate on validation set
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
          target_names=['Scream', 'Cry', 'Control']))

if __name__ == "__main__":
    main()