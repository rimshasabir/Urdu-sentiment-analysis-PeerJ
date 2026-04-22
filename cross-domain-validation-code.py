import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, cohen_kappa_score, confusion_matrix)
import gc
from tqdm.auto import tqdm

# --- 1. GPU & Data Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Load Dataset
# Note: Ensure 'imdb_urdu_reviews.csv' is in the same directory
df = pd.read_csv("/kaggle/input/datasets/rimshasabir/originaldataset/wholedataset.csv")

# Clean data: Use 'review' and 'sentiment' columns from your file
df = df.dropna(subset=['sentiment', 'review']).copy()

# Map labels: positive -> 1, negative -> 0
label_map = {'negative': 0, 'positive': 1}
df['sentiment_label'] = df['sentiment'].str.strip().str.lower().map(label_map)
df = df.dropna(subset=['sentiment_label'])

# Create Holdout Test Set (10%)
main_df, holdout_df = train_test_split(
    df, test_size=0.10, stratify=df['sentiment_label'], random_state=42
)

# --- 2. Dataset Class ---
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['review'].astype(str).tolist()
        self.labels = df['sentiment_label'].tolist()
        self.tokenizer = tokenizer

    def __len__(self): 
        return len(self.texts)

    def __getitem__(self, idx):
        # The tokenizer automatically handles the Urdu text characters
        enc = self.tokenizer(
            self.texts[idx], 
            max_length=128, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0), 
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# --- 3. 5-Fold Cross-Validation ---
NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

fold_accs = []
best_acc = 0.0

for fold, (train_idx, val_idx) in enumerate(skf.split(main_df['review'], main_df['sentiment_label'])):
    print(f"\nTraining Fold {fold+1}...")
    
    # Initialize model with 2 labels (Positive/Negative)
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased", 
        num_labels=2
    ).to(device)
    
    train_loader = DataLoader(SentimentDataset(main_df.iloc[train_idx], tokenizer), batch_size=80, shuffle=True)
    val_loader = DataLoader(SentimentDataset(main_df.iloc[val_idx], tokenizer), batch_size=80)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training Loop (2 Epochs)
    for epoch in range(2):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    
    acc = correct / total
    fold_accs.append(acc)
    print(f"Fold {fold+1} Accuracy: {acc:.4f}")

    # Save the best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.bin")
    
    del model
    clear_gpu_memory()

# --- 4. Final Evaluation on Holdout Test Set ---
print(f"\n{'='*20} FINAL EVALUATION ON HOLDOUT SET {'='*20}")

# Reload best model architecture
model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2).to(device)
model.load_state_dict(torch.load("best_model.bin"))
model.eval()

holdout_loader = DataLoader(SentimentDataset(holdout_df, tokenizer), batch_size=32)
y_true, y_pred = [], []

with torch.no_grad():
    for batch in tqdm(holdout_loader, desc="Testing"):
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
        y_true.extend(batch['labels'].cpu().numpy())

# Calculate All Metrics
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision": precision_score(y_true, y_pred),
    "Recall": recall_score(y_true, y_pred),
    "F1-Score": f1_score(y_true, y_pred),
    "Cohen's Kappa": cohen_kappa_score(y_true, y_pred)
}

for name, val in metrics.items():
    print(f"{name}: {val:.4f}")

# --- 5. Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'], 
            yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix: Final Holdout Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
