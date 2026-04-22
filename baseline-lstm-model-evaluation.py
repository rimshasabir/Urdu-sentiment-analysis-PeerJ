```python
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import torch.nn.functional as F

# Load dataset
df = pd.read_csv(r'C:\Users\RIMSHA\PycharmProjects\RESEARCH\train_data.csv')
print("CSV loaded successfully. Columns found:", df.columns.tolist())

# Preprocessing
df = df.dropna(subset=['Sentiment', 'TokenizedText'])
df['Sentiment'] = df['Sentiment'].str.strip().str.lower()
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment'] = df['Sentiment'].map(label_map)
df['TokenizedText'] = df['TokenizedText'].astype(str)

# Split dataset
train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Vocabulary creation
def build_vocab(texts):
    word_counts = Counter()
    for text in texts:
        words = text.split()
        word_counts.update(words)
    vocab = {word: idx+2 for idx, (word, count) in enumerate(word_counts.items())}  # +2 for padding and unknown
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

vocab = build_vocab(train_df['TokenizedText'])
vocab_size = len(vocab)

# Dataset class
class LSTMDataset(Dataset):
    def __init__(self, df, vocab):
        self.texts = df['TokenizedText'].tolist()
        self.labels = df['sentiment'].tolist()
        self.vocab = vocab
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        words = text.split()
        indices = [self.vocab.get(word, self.vocab['<UNK>']) for word in words]
        return {
            'indices': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Collate function for DataLoader
def collate_fn(batch):
    indices = [item['indices'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.tensor([len(idx) for idx in indices], dtype=torch.long)
    padded_indices = pad_sequence(indices, batch_first=True, padding_value=0)
    return {
        'indices': padded_indices,
        'labels': labels,
        'lengths': lengths
    }

# LSTM Model
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                            dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text, text_lengths):
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# Hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 3
N_LAYERS = 2
DROPOUT = 0.5
BATCH_SIZE = 32
EPOCHS = 10

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SentimentLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, DROPOUT).to(device)
optimizer = Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Create datasets and dataloaders
train_dataset = LSTMDataset(train_df, vocab)
val_dataset = LSTMDataset(val_df, vocab)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        indices = batch['indices'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)
        
        predictions = model(indices, lengths)
        loss = criterion(predictions, labels)
        
        acc = accuracy_score(labels.cpu(), predictions.argmax(1).cpu())
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc
    
    # Validation
    model.eval()
    val_loss = 0
    val_acc = 0
    
    with torch.no_grad():
        for batch in val_loader:
            indices = batch['indices'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)
            
            predictions = model(indices, lengths)
            loss = criterion(predictions, labels)
            
            acc = accuracy_score(labels.cpu(), predictions.argmax(1).cpu())
            
            val_loss += loss.item()
            val_acc += acc
    
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {epoch_loss/len(train_loader):.3f} | Train Acc: {epoch_acc/len(train_loader)*100:.2f}%')
    print(f'\t Val. Loss: {val_loss/len(val_loader):.3f} |  Val. Acc: {val_acc/len(val_loader)*100:.2f}%')

# Predict on full dataset
full_dataset = LSTMDataset(df, vocab)
full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

model.eval()
all_predictions = []
with torch.no_grad():
    for batch in full_loader:
        indices = batch['indices'].to(device)
        lengths = batch['lengths'].to(device)
        predictions = model(indices, lengths)
        all_predictions.extend(predictions.argmax(1).cpu().numpy())

# Save results
label_map_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}
df['LSTM_Sentiment'] = [label_map_reverse[p] for p in all_predictions]
df.to_csv(r'D:\model\LSTM_Train.csv', index=False)

# Save model
torch.save(model.state_dict(), r'D:\model\lstm_model.pt')
print("LSTM model training completed and saved successfully!")
```
