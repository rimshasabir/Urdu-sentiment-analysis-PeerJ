"""
train_xlmr.py

Fine-tuning XLM-RoBERTa for Urdu Sentiment Analysis

This script fine-tunes the twitter-xlm-roberta-base-sentiment model
on Urdu text data for ternary sentiment classification (positive/negative/neutral).
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
import sys

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UrduSentimentDataset(Dataset):
    """PyTorch Dataset for Urdu sentiment analysis."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Initialize dataset.

        Args:
            texts (list): List of Urdu text samples
            labels (list): List of integer labels
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Validate inputs
        assert len(self.texts) == len(self.labels), \
            "Texts and labels must have same length"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class XLMRSentimentTrainer:
    """Trainer for XLM-R sentiment classification."""

    def __init__(self, config_path="config_xlmr.yaml"):
        """
        Initialize trainer.

        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract config
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.data_config = self.config['data']

        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.training_config.get('use_gpu', True)
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.label_map_reverse = None

    def setup_label_mapping(self, df):
        """Setup label mapping from dataset."""
        if 'Sentiment' in df.columns:
            unique_labels = df['Sentiment'].str.strip().str.lower().unique()
            self.label_map = {}
            self.label_map_reverse = {}

            # Determine mapping based on label names
            for label in unique_labels:
                if 'neg' in label:
                    self.label_map[label] = 0
                elif 'neu' in label or label == '0':
                    self.label_map[label] = 1
                elif 'pos' in label:
                    self.label_map[label] = 2
                else:
                    # Fallback: assign sequentially
                    if label not in self.label_map:
                        self.label_map[label] = len(self.label_map)

            # Create reverse mapping
            self.label_map_reverse = {v: k for k, v in self.label_map.items()}

            logger.info(f"Label mapping: {self.label_map}")
            logger.info(f"Reverse mapping: {self.label_map_reverse}")

        return df

    def load_and_prepare_data(self, data_path):
        """
        Load and prepare data for training.

        Args:
            data_path (str): Path to CSV file

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logger.info(f"Loading data from {data_path}")

        try:
            # Try different encodings for Urdu text
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    logger.info(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            # Validate required columns
            required_columns = ['TokenizedText', 'Sentiment']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Clean data
            df = df.dropna(subset=['TokenizedText', 'Sentiment'])
            df['Sentiment'] = df['Sentiment'].str.strip().str.lower()

            # Setup label mapping
            df = self.setup_label_mapping(df)

            # Map labels
            df['label'] = df['Sentiment'].map(self.label_map)

            # Drop any unmapped labels
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)

            # Ensure TokenizedText is string
            df['TokenizedText'] = df['TokenizedText'].astype(str)

            logger.info(f"Loaded {len(df)} samples")
            logger.info(f"Label distribution:\n{df['Sentiment'].value_counts()}")

            # Split data
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.data_config['test_size'],
                random_state=self.training_config['random_seed'],
                stratify=df['label'] if self.data_config.get('stratify', True) else None
            )

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.data_config['val_size'],
                random_state=self.training_config['random_seed'],
                stratify=train_val_df['label'] if self.data_config.get('stratify', True) else None
            )

            logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

            return train_df, val_df, test_df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def initialize_model(self):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {self.model_config['model_name']}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model_name']
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config['model_name'],
            num_labels=len(self.label_map)
        )

        self.model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return self.model, self.tokenizer

    def create_dataloaders(self, train_df, val_df, test_df):
        """Create DataLoaders for training, validation, and testing."""
        logger.info("Creating DataLoaders...")

        # Create datasets
        train_dataset = UrduSentimentDataset(
            texts=train_df['TokenizedText'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        val_dataset = UrduSentimentDataset(
            texts=val_df['TokenizedText'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        test_dataset = UrduSentimentDataset(
            texts=test_df['TokenizedText'].tolist(),
            labels=test_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_desc = f"Epoch {epoch + 1}/{self.training_config['epochs']}"

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.training_config.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['max_grad_norm']
                )

            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()

            # Get predictions
            with torch.no_grad():
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Log progress
            if (batch_idx + 1) % self.training_config.get('log_interval', 10) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f}")

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def evaluate(self, data_loader, split_name="Validation"):
        """Evaluate model on given data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        # Generate classification report
        class_report = classification_report(
            all_labels,
            all_preds,
            target_names=list(self.label_map.keys()),
            zero_division=0
        )

        logger.info(f"\n{split_name} Results:")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"\nClassification Report:\n{class_report}")

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, train_loader, val_loader):
        """Main training loop."""
        logger.info("Starting training...")

        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )

        # Setup scheduler
        num_training_steps = len(train_loader) * self.training_config['epochs']
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.training_config.get('warmup_steps', 0),
            num_training_steps=num_training_steps
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(self.training_config['epochs']):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Epoch {epoch + 1}/{self.training_config['epochs']}")
            logger.info(f"{'=' * 50}")

            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, scheduler, epoch
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validate
            val_results = self.evaluate(val_loader, "Validation")
            history['val_loss'].append(val_results['loss'])
            history['val_acc'].append(val_results['accuracy'])
            history['val_f1'].append(val_results['f1'])

            # Save best model
            if val_results['f1'] > best_val_f1:
                best_val_f1 = val_results['f1']
                self.save_model(f"best_model_f1_{best_val_f1:.4f}")
                patience_counter = 0
                logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if self.training_config.get('patience') and \
                    patience_counter >= self.training_config['patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return history

    def save_model(self, model_name="xlmr_sentiment_model"):
        """Save model and tokenizer."""
        save_path = Path(self.training_config['output_dir']) / model_name
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save label mapping
        import json
        with open(save_path / 'label_mapping.json', 'w', encoding='utf-8') as f:
            json.dump({
                'label_map': self.label_map,
                'label_map_reverse': self.label_map_reverse
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Model saved to {save_path}")

    def predict(self, texts):
        """Make predictions on new texts."""
        self.model.eval()

        # Prepare inputs
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.model_config['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Map to labels
        labels = [self.label_map_reverse[p] for p in preds]

        return labels, preds


def main():
    """Command-line interface for XLM-R training."""
    parser = argparse.ArgumentParser(
        description='Fine-tune XLM-R for Urdu sentiment analysis'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='models/xlmr_model',
        help='Directory to save model'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config_xlmr.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Only run evaluation on test set'
    )

    args = parser.parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        # Initialize trainer
        trainer = XLMRSentimentTrainer(args.config)

        # Update output directory
        trainer.training_config['output_dir'] = args.output_dir

        # Load data
        train_df, val_df, test_df = trainer.load_and_prepare_data(args.input)

        # Initialize model
        trainer.initialize_model()

        if not args.test_only:
            # Create dataloaders
            train_loader, val_loader, test_loader = trainer.create_dataloaders(
                train_df, val_df, test_df
            )

            # Train model
            history = trainer.train(train_loader, val_loader)

            # Save final model
            trainer.save_model("final_model")

            # Test final model
            logger.info("\n" + "=" * 50)
            logger.info("Final Test Results")
            logger.info("=" * 50)
            trainer.evaluate(test_loader, "Test")

        else:
            # Test only mode
            logger.info("Running evaluation on test set...")
            _, _, test_loader = trainer.create_dataloaders(train_df, val_df, test_df)
            trainer.evaluate(test_loader, "Test")

        logger.info("Training/evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()"""
train_xlmr.py

Fine-tuning XLM-RoBERTa for Urdu Sentiment Analysis

This script fine-tunes the twitter-xlm-roberta-base-sentiment model
on Urdu text data for ternary sentiment classification (positive/negative/neutral).
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
import sys

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UrduSentimentDataset(Dataset):
    """PyTorch Dataset for Urdu sentiment analysis."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Initialize dataset.

        Args:
            texts (list): List of Urdu text samples
            labels (list): List of integer labels
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Validate inputs
        assert len(self.texts) == len(self.labels), \
            "Texts and labels must have same length"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class XLMRSentimentTrainer:
    """Trainer for XLM-R sentiment classification."""

    def __init__(self, config_path="config_xlmr.yaml"):
        """
        Initialize trainer.

        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract config
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.data_config = self.config['data']

        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.training_config.get('use_gpu', True)
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.label_map_reverse = None

    def setup_label_mapping(self, df):
        """Setup label mapping from dataset."""
        if 'Sentiment' in df.columns:
            unique_labels = df['Sentiment'].str.strip().str.lower().unique()
            self.label_map = {}
            self.label_map_reverse = {}

            # Determine mapping based on label names
            for label in unique_labels:
                if 'neg' in label:
                    self.label_map[label] = 0
                elif 'neu' in label or label == '0':
                    self.label_map[label] = 1
                elif 'pos' in label:
                    self.label_map[label] = 2
                else:
                    # Fallback: assign sequentially
                    if label not in self.label_map:
                        self.label_map[label] = len(self.label_map)

            # Create reverse mapping
            self.label_map_reverse = {v: k for k, v in self.label_map.items()}

            logger.info(f"Label mapping: {self.label_map}")
            logger.info(f"Reverse mapping: {self.label_map_reverse}")

        return df

    def load_and_prepare_data(self, data_path):
        """
        Load and prepare data for training.

        Args:
            data_path (str): Path to CSV file

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logger.info(f"Loading data from {data_path}")

        try:
            # Try different encodings for Urdu text
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    logger.info(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            # Validate required columns
            required_columns = ['TokenizedText', 'Sentiment']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Clean data
            df = df.dropna(subset=['TokenizedText', 'Sentiment'])
            df['Sentiment'] = df['Sentiment'].str.strip().str.lower()

            # Setup label mapping
            df = self.setup_label_mapping(df)

            # Map labels
            df['label'] = df['Sentiment'].map(self.label_map)

            # Drop any unmapped labels
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)

            # Ensure TokenizedText is string
            df['TokenizedText'] = df['TokenizedText'].astype(str)

            logger.info(f"Loaded {len(df)} samples")
            logger.info(f"Label distribution:\n{df['Sentiment'].value_counts()}")

            # Split data
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.data_config['test_size'],
                random_state=self.training_config['random_seed'],
                stratify=df['label'] if self.data_config.get('stratify', True) else None
            )

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.data_config['val_size'],
                random_state=self.training_config['random_seed'],
                stratify=train_val_df['label'] if self.data_config.get('stratify', True) else None
            )

            logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

            return train_df, val_df, test_df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def initialize_model(self):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {self.model_config['model_name']}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model_name']
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config['model_name'],
            num_labels=len(self.label_map)
        )

        self.model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return self.model, self.tokenizer

    def create_dataloaders(self, train_df, val_df, test_df):
        """Create DataLoaders for training, validation, and testing."""
        logger.info("Creating DataLoaders...")

        # Create datasets
        train_dataset = UrduSentimentDataset(
            texts=train_df['TokenizedText'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        val_dataset = UrduSentimentDataset(
            texts=val_df['TokenizedText'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        test_dataset = UrduSentimentDataset(
            texts=test_df['TokenizedText'].tolist(),
            labels=test_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_desc = f"Epoch {epoch + 1}/{self.training_config['epochs']}"

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.training_config.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['max_grad_norm']
                )

            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()

            # Get predictions
            with torch.no_grad():
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Log progress
            if (batch_idx + 1) % self.training_config.get('log_interval', 10) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f}")

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def evaluate(self, data_loader, split_name="Validation"):
        """Evaluate model on given data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        # Generate classification report
        class_report = classification_report(
            all_labels,
            all_preds,
            target_names=list(self.label_map.keys()),
            zero_division=0
        )

        logger.info(f"\n{split_name} Results:")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"\nClassification Report:\n{class_report}")

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, train_loader, val_loader):
        """Main training loop."""
        logger.info("Starting training...")

        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )

        # Setup scheduler
        num_training_steps = len(train_loader) * self.training_config['epochs']
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.training_config.get('warmup_steps', 0),
            num_training_steps=num_training_steps
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(self.training_config['epochs']):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.training_config['epochs']}")
            logger.info(f"{'='*50}")

            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, scheduler, epoch
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validate
            val_results = self.evaluate(val_loader, "Validation")
            history['val_loss'].append(val_results['loss'])
            history['val_acc'].append(val_results['accuracy'])
            history['val_f1'].append(val_results['f1'])

            # Save best model
            if val_results['f1'] > best_val_f1:
                best_val_f1 = val_results['f1']
                self.save_model(f"best_model_f1_{best_val_f1:.4f}")
                patience_counter = 0
                logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if self.training_config.get('patience') and \
               patience_counter >= self.training_config['patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return history

    def save_model(self, model_name="xlmr_sentiment_model"):
        """Save model and tokenizer."""
        save_path = Path(self.training_config['output_dir']) / model_name
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save label mapping
        import json
        with open(save_path / 'label_mapping.json', 'w', encoding='utf-8') as f:
            json.dump({
                'label_map': self.label_map,
                'label_map_reverse': self.label_map_reverse
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Model saved to {save_path}")

    def predict(self, texts):
        """Make predictions on new texts."""
        self.model.eval()

        # Prepare inputs
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.model_config['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Map to labels
        labels = [self.label_map_reverse[p] for p in preds]

        return labels, preds


def main():
    """Command-line interface for XLM-R training."""
    parser = argparse.ArgumentParser(
        description='Fine-tune XLM-R for Urdu sentiment analysis'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='models/xlmr_model',
        help='Directory to save model'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config_xlmr.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Only run evaluation on test set'
    )

    args = parser.parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        # Initialize trainer
        trainer = XLMRSentimentTrainer(args.config)

        # Update output directory
        trainer.training_config['output_dir'] = args.output_dir

        # Load data
        train_df, val_df, test_df = trainer.load_and_prepare_data(args.input)

        # Initialize model
        trainer.initialize_model()

        if not args.test_only:
            # Create dataloaders
            train_loader, val_loader, test_loader = trainer.create_dataloaders(
                train_df, val_df, test_df
            )

            # Train model
            history = trainer.train(train_loader, val_loader)

            # Save final model
            trainer.save_model("final_model")

            # Test final model
            logger.info("\n" + "="*50)
            logger.info("Final Test Results")
            logger.info("="*50)
            trainer.evaluate(test_loader, "Test")

        else:
            # Test only mode
            logger.info("Running evaluation on test set...")
            _, _, test_loader = trainer.create_dataloaders(train_df, val_df, test_df)
            trainer.evaluate(test_loader, "Test")

        logger.info("Training/evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()"""
train_xlmr.py

Fine-tuning XLM-RoBERTa for Urdu Sentiment Analysis

This script fine-tunes the twitter-xlm-roberta-base-sentiment model
on Urdu text data for ternary sentiment classification (positive/negative/neutral).
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
import sys

# Hugging Face imports
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_scheduler
)
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UrduSentimentDataset(Dataset):
    """PyTorch Dataset for Urdu sentiment analysis."""

    def __init__(self, texts, labels, tokenizer, max_length=256):
        """
        Initialize dataset.

        Args:
            texts (list): List of Urdu text samples
            labels (list): List of integer labels
            tokenizer: Hugging Face tokenizer
            max_length (int): Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Validate inputs
        assert len(self.texts) == len(self.labels), \
            "Texts and labels must have same length"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class XLMRSentimentTrainer:
    """Trainer for XLM-R sentiment classification."""

    def __init__(self, config_path="config_xlmr.yaml"):
        """
        Initialize trainer.

        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract config
        self.model_config = self.config['model']
        self.training_config = self.config['training']
        self.data_config = self.config['data']

        # Setup device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.training_config.get('use_gpu', True)
            else "cpu"
        )
        logger.info(f"Using device: {self.device}")

        if self.device.type == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.label_map = None
        self.label_map_reverse = None

    def setup_label_mapping(self, df):
        """Setup label mapping from dataset."""
        if 'Sentiment' in df.columns:
            unique_labels = df['Sentiment'].str.strip().str.lower().unique()
            self.label_map = {}
            self.label_map_reverse = {}

            # Determine mapping based on label names
            for label in unique_labels:
                if 'neg' in label:
                    self.label_map[label] = 0
                elif 'neu' in label or label == '0':
                    self.label_map[label] = 1
                elif 'pos' in label:
                    self.label_map[label] = 2
                else:
                    # Fallback: assign sequentially
                    if label not in self.label_map:
                        self.label_map[label] = len(self.label_map)

            # Create reverse mapping
            self.label_map_reverse = {v: k for k, v in self.label_map.items()}

            logger.info(f"Label mapping: {self.label_map}")
            logger.info(f"Reverse mapping: {self.label_map_reverse}")

        return df

    def load_and_prepare_data(self, data_path):
        """
        Load and prepare data for training.

        Args:
            data_path (str): Path to CSV file

        Returns:
            tuple: (train_df, val_df, test_df)
        """
        logger.info(f"Loading data from {data_path}")

        try:
            # Try different encodings for Urdu text
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
                try:
                    df = pd.read_csv(data_path, encoding=encoding)
                    logger.info(f"Successfully loaded with {encoding} encoding")
                    break
                except UnicodeDecodeError:
                    continue

            # Validate required columns
            required_columns = ['TokenizedText', 'Sentiment']
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Clean data
            df = df.dropna(subset=['TokenizedText', 'Sentiment'])
            df['Sentiment'] = df['Sentiment'].str.strip().str.lower()

            # Setup label mapping
            df = self.setup_label_mapping(df)

            # Map labels
            df['label'] = df['Sentiment'].map(self.label_map)

            # Drop any unmapped labels
            df = df.dropna(subset=['label'])
            df['label'] = df['label'].astype(int)

            # Ensure TokenizedText is string
            df['TokenizedText'] = df['TokenizedText'].astype(str)

            logger.info(f"Loaded {len(df)} samples")
            logger.info(f"Label distribution:\n{df['Sentiment'].value_counts()}")

            # Split data
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.data_config['test_size'],
                random_state=self.training_config['random_seed'],
                stratify=df['label'] if self.data_config.get('stratify', True) else None
            )

            train_df, val_df = train_test_split(
                train_val_df,
                test_size=self.data_config['val_size'],
                random_state=self.training_config['random_seed'],
                stratify=train_val_df['label'] if self.data_config.get('stratify', True) else None
            )

            logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

            return train_df, val_df, test_df

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def initialize_model(self):
        """Initialize model and tokenizer."""
        logger.info(f"Loading model: {self.model_config['model_name']}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['model_name']
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_config['model_name'],
            num_labels=len(self.label_map)
        )

        self.model.to(self.device)

        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")

        return self.model, self.tokenizer

    def create_dataloaders(self, train_df, val_df, test_df):
        """Create DataLoaders for training, validation, and testing."""
        logger.info("Creating DataLoaders...")

        # Create datasets
        train_dataset = UrduSentimentDataset(
            texts=train_df['TokenizedText'].tolist(),
            labels=train_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        val_dataset = UrduSentimentDataset(
            texts=val_df['TokenizedText'].tolist(),
            labels=val_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        test_dataset = UrduSentimentDataset(
            texts=test_df['TokenizedText'].tolist(),
            labels=test_df['label'].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=True,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=self.device.type == 'cuda'
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader, optimizer, scheduler, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        progress_desc = f"Epoch {epoch + 1}/{self.training_config['epochs']}"

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.training_config.get('max_grad_norm'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['max_grad_norm']
                )

            optimizer.step()
            scheduler.step()

            # Track metrics
            total_loss += loss.item()

            # Get predictions
            with torch.no_grad():
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            # Log progress
            if (batch_idx + 1) % self.training_config.get('log_interval', 10) == 0:
                avg_loss = total_loss / (batch_idx + 1)
                logger.debug(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {avg_loss:.4f}")

        # Calculate epoch metrics
        epoch_loss = total_loss / len(train_loader)
        epoch_acc = accuracy_score(all_labels, all_preds)

        return epoch_loss, epoch_acc

    def evaluate(self, data_loader, split_name="Validation"):
        """Evaluate model on given data loader."""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            for batch in data_loader:
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)

                outputs = self.model(**inputs, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()

                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='macro', zero_division=0
        )

        # Generate classification report
        class_report = classification_report(
            all_labels,
            all_preds,
            target_names=list(self.label_map.keys()),
            zero_division=0
        )

        logger.info(f"\n{split_name} Results:")
        logger.info(f"Loss: {avg_loss:.4f}")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Macro F1: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"\nClassification Report:\n{class_report}")

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': all_preds,
            'labels': all_labels
        }

    def train(self, train_loader, val_loader):
        """Main training loop."""
        logger.info("Starting training...")

        # Setup optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config.get('weight_decay', 0.01)
        )

        # Setup scheduler
        num_training_steps = len(train_loader) * self.training_config['epochs']
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=self.training_config.get('warmup_steps', 0),
            num_training_steps=num_training_steps
        )

        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }

        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(self.training_config['epochs']):
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.training_config['epochs']}")
            logger.info(f"{'='*50}")

            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, scheduler, epoch
            )
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

            # Validate
            val_results = self.evaluate(val_loader, "Validation")
            history['val_loss'].append(val_results['loss'])
            history['val_acc'].append(val_results['accuracy'])
            history['val_f1'].append(val_results['f1'])

            # Save best model
            if val_results['f1'] > best_val_f1:
                best_val_f1 = val_results['f1']
                self.save_model(f"best_model_f1_{best_val_f1:.4f}")
                patience_counter = 0
                logger.info(f"New best model saved with F1: {best_val_f1:.4f}")
            else:
                patience_counter += 1

            # Early stopping
            if self.training_config.get('patience') and \
               patience_counter >= self.training_config['patience']:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

        return history

    def save_model(self, model_name="xlmr_sentiment_model"):
        """Save model and tokenizer."""
        save_path = Path(self.training_config['output_dir']) / model_name
        save_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # Save label mapping
        import json
        with open(save_path / 'label_mapping.json', 'w', encoding='utf-8') as f:
            json.dump({
                'label_map': self.label_map,
                'label_map_reverse': self.label_map_reverse
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Model saved to {save_path}")

    def predict(self, texts):
        """Make predictions on new texts."""
        self.model.eval()

        # Prepare inputs
        encodings = self.tokenizer(
            texts,
            add_special_tokens=True,
            max_length=self.model_config['max_length'],
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in encodings.items()}

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        # Map to labels
        labels = [self.label_map_reverse[p] for p in preds]

        return labels, preds


def main():
    """Command-line interface for XLM-R training."""
    parser = argparse.ArgumentParser(
        description='Fine-tune XLM-R for Urdu sentiment analysis'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='models/xlmr_model',
        help='Directory to save model'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config_xlmr.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--test_only',
        action='store_true',
        help='Only run evaluation on test set'
    )

    args = parser.parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        # Initialize trainer
        trainer = XLMRSentimentTrainer(args.config)

        # Update output directory
        trainer.training_config['output_dir'] = args.output_dir

        # Load data
        train_df, val_df, test_df = trainer.load_and_prepare_data(args.input)

        # Initialize model
        trainer.initialize_model()

        if not args.test_only:
            # Create dataloaders
            train_loader, val_loader, test_loader = trainer.create_dataloaders(
                train_df, val_df, test_df
            )

            # Train model
            history = trainer.train(train_loader, val_loader)

            # Save final model
            trainer.save_model("final_model")

            # Test final model
            logger.info("\n" + "="*50)
            logger.info("Final Test Results")
            logger.info("="*50)
            trainer.evaluate(test_loader, "Test")

        else:
            # Test only mode
            logger.info("Running evaluation on test set...")
            _, _, test_loader = trainer.create_dataloaders(train_df, val_df, test_df)
            trainer.evaluate(test_loader, "Test")

        logger.info("Training/evaluation completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    main()