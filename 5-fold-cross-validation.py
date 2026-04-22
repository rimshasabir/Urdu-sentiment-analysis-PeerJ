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

# --- GPU & Data Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# Load Full Dataset
df = pd.read_csv("/kaggle/input/datasets/rimshasabir/originaldataset/output.csv")
df = df.dropna(subset=['Sentiment', 'TokenizedText']).copy()
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment'] = df['Sentiment'].str.strip().str.lower().map(label_map)
df = df.dropna(subset=['sentiment'])

# 1. Create Holdout Test Set (10%)
main_df, holdout_df = train_test_split(
    df, test_size=0.10, stratify=df['sentiment'], random_state=42
)

# --- Dataset Class ---
class SentimentDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts = df['TokenizedText'].astype(str).tolist()
        self.labels = df['sentiment'].tolist()
        self.tokenizer = tokenizer
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': enc['input_ids'].squeeze(0), 'attention_mask': enc['attention_mask'].squeeze(0), 
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)}

# 2. 5-Fold Cross-Validation on Remaining 90%
NUM_FOLDS = 5
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

fold_accs = []
for fold, (train_idx, val_idx) in enumerate(skf.split(main_df['TokenizedText'], main_df['sentiment'])):
    print(f"\nTraining Fold {fold+1}...")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=3).to(device)
    
    train_loader = DataLoader(SentimentDataset(main_df.iloc[train_idx], tokenizer), batch_size=64, shuffle=True)
    val_loader = DataLoader(SentimentDataset(main_df.iloc[val_idx], tokenizer), batch_size=64)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # Simple 2-epoch training for demonstration (Standard for BERT is 3-4)
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    # Record Best Fold Accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            correct += (torch.argmax(logits, dim=1) == batch['labels']).sum().item()
            total += batch['labels'].size(0)
    fold_accs.append(correct/total)
    # Save the model from the best fold for final evaluation
    if fold == np.argmax(fold_accs):
        torch.save(model.state_dict(), "best_model.bin")
    clear_gpu_memory()

# 3. Final Evaluation on Holdout Test Set
print(f"\n{'='*20} FINAL EVALUATION ON HOLDOUT SET {'='*20}")
model.load_state_dict(torch.load("best_model.bin"))
model.eval()
holdout_loader = DataLoader(SentimentDataset(holdout_df, tokenizer), batch_size=32)

y_true, y_pred = [], []
with torch.no_grad():
    for batch in holdout_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
        y_true.extend(batch['labels'].cpu().numpy())

# Calculate All Metrics
metrics = {
    "Accuracy": accuracy_score(y_true, y_pred),
    "Precision (Weighted)": precision_score(y_true, y_pred, average='weighted'),
    "Recall (Weighted)": recall_score(y_true, y_pred, average='weighted'),
    "F1-Score (Weighted)": f1_score(y_true, y_pred, average='weighted'),
    "Cohen's Kappa": cohen_kappa_score(y_true, y_pred)
}

for name, val in metrics.items():
    print(f"{name}: {val:.4f}")

# 4. Confusion Matrix Representation
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_map.keys(), yticklabels=label_map.keys())
plt.title('Confusion Matrix: Final Holdout Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
