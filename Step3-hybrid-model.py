"""
step3_hybrid_model.py

Hybrid model combining XLM-R embeddings with negation features.
"""

import argparse
import yaml
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from pathlib import Path
import sys

# Set TensorFlow settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridModel:
    """Hybrid model for Urdu sentiment analysis with negation handling."""

    def __init__(self, config_path="config.yaml"):
        """
        Initialize hybrid model.

        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract config
        self.model_config = self.config['hybrid_model']
        self.max_length = self.model_config['max_length']
        self.batch_size = self.model_config['batch_size']
        self.epochs = self.model_config['epochs']
        self.learning_rate = self.model_config['learning_rate']
        self.dropout_rate = self.model_config['dropout_rate']

        # Initialize tokenizer
        from transformers import XLMRobertaTokenizer
        self.tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")

        # Model will be built later
        self.model = None

    def load_and_prepare_data(self, data_path):
        """
        Load and prepare data for training.

        Args:
            data_path (str): Path to augmented dataset

        Returns:
            tuple: Prepared data and class weights
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Clean and map labels
        df['NewSentiment'] = df['NewSentiment'].str.strip().str.title()
        sentiment_mapping = {'Positive': 0, 'Negative': 1, 'Neutral': 2}
        df['label'] = df['NewSentiment'].map(sentiment_mapping)

        # Handle missing labels
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)

        # Calculate enhanced class weights
        class_counts = df['label'].value_counts().sort_index()
        total_samples = len(df)
        class_weights = {
            i: np.sqrt(total_samples / (len(class_counts) * count))
            for i, count in enumerate(class_counts)
        }

        logger.info("Class distribution:")
        for i, count in class_counts.items():
            logger.info(f"  Class {i}: {count} samples")

        logger.info("Class weights:")
        for i, weight in class_weights.items():
            logger.info(f"  Class {i}: {weight:.4f}")

        # Prepare inputs
        X_tokens = []
        X_negation = []

        for _, row in df.iterrows():
            # Tokenize text
            if isinstance(row['TokenizedText'], str):
                import ast
                try:
                    tokens = ast.literal_eval(row['TokenizedText'])
                    text = ' '.join(tokens)
                except:
                    text = row['TokenizedText']
            else:
                text = ' '.join(row['TokenizedText'])

            # Tokenize with XLM-R
            encoded = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='tf'
            )
            X_tokens.append(encoded['input_ids'][0])

            # Negation feature
            negation_feature = float(row.get('has_negation', 0))
            X_negation.append([negation_feature])

        # Convert to arrays
        X_tokens = np.stack(X_tokens)
        X_negation = np.array(X_negation)
        y = tf.keras.utils.to_categorical(df['label'].values)

        # Split data
        from sklearn.model_selection import train_test_split
        X_train_tok, X_val_tok, X_train_neg, X_val_neg, y_train, y_val = train_test_split(
            X_tokens, X_negation, y,
            test_size=self.model_config['train_val_split'],
            random_state=42,
            stratify=df['label']
        )

        return (X_train_tok, X_val_tok, X_train_neg, X_val_neg, y_train, y_val), class_weights

    def build_model(self):
        """Build the hybrid model architecture."""
        logger.info("Building hybrid model...")

        from tensorflow.keras.layers import (
            Input, Dense, Dropout, Concatenate, LayerNormalization,
            BatchNormalization, GlobalAveragePooling1D
        )
        from tensorflow.keras.models import Model
        from transformers import TFXLMRobertaModel

        # Input layers
        token_input = Input(shape=(self.max_length,), dtype=tf.int32, name='token_input')
        negation_input = Input(shape=(1,), name='negation_input')

        # XLM-R transformer
        transformer = TFXLMRobertaModel.from_pretrained("xlm-roberta-base")

        # Gradual unfreezing
        unfreeze_layers = self.model_config.get('unfreeze_last_layers', 6)
        for i, layer in enumerate(transformer.layers):
            layer.trainable = i >= len(transformer.layers) - unfreeze_layers

        # Get transformer output
        transformer_output = transformer(token_input).last_hidden_state
        pooled_output = GlobalAveragePooling1D()(transformer_output)
        pooled_output = LayerNormalization()(pooled_output)

        # Negation pathway
        negation_path = Dense(64, activation='relu')(negation_input)
        negation_path = BatchNormalization()(negation_path)
        negation_path = Dropout(0.3)(negation_path)
        negation_path = Dense(32, activation='relu')(negation_path)

        # Combine features
        combined = Concatenate()([pooled_output, negation_path])
        combined = Dense(256, activation='relu')(combined)
        combined = Dropout(self.dropout_rate)(combined)
        combined = Dense(128, activation='relu')(combined)

        # Output layer
        output = Dense(3, activation='softmax')(combined)

        # Create model
        self.model = Model(inputs=[token_input, negation_input], outputs=output)

        # Compile model
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0
        )

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )

        logger.info("Model built successfully")
        self.model.summary(print_fn=logger.info)

        return self.model

    def train(self, train_data, val_data, class_weights, output_dir="models"):
        """
        Train the hybrid model.

        Args:
            train_data (tuple): Training data
            val_data (tuple): Validation data
            class_weights (dict): Class weights
            output_dir (str): Directory to save models

        Returns:
            History: Training history
        """
        logger.info("Starting model training...")

        # Unpack data
        X_train_tok, X_val_tok, X_train_neg, X_val_neg, y_train, y_val = train_data
        _, _, _, _, _, _ = val_data

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_path / "best_model.weights.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.CSVLogger(
                str(output_path / "training_log.csv"),
                append=False
            )
        ]

        # Train model
        history = self.model.fit(
            [X_train_tok, X_train_neg],
            y_train,
            validation_data=([X_val_tok, X_val_neg], y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )

        # Save final model
        self.model.save(str(output_path / "final_model"))
        logger.info(f"Model saved to {output_path}")

        return history

    def evaluate(self, test_data):
        """
        Evaluate the model.

        Args:
            test_data (tuple): Test data

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model...")

        X_test_tok, _, X_test_neg, _, y_test, _ = test_data

        results = self.model.evaluate(
            [X_test_tok, X_test_neg],
            y_test,
            verbose=0
        )

        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }

        # Calculate F1-score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * (metrics['precision'] * metrics['recall']) / \
                            (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0

        return metrics


def main():
    """Command-line interface for hybrid model training."""
    parser = argparse.ArgumentParser(
        description='Train hybrid model for Urdu sentiment analysis'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to augmented dataset CSV'
    )

    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='models',
        help='Directory to save model and logs'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--test_split',
        type=float,
        default=0.1,
        help='Test split ratio (default: 0.1)'
    )

    args = parser.parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        # Initialize model
        hybrid_model = HybridModel(args.config)

        # Load and prepare data
        (X_train_tok, X_val_tok, X_train_neg, X_val_neg, y_train, y_val), class_weights = \
            hybrid_model.load_and_prepare_data(args.input)

        train_data = (X_train_tok, X_val_tok, X_train_neg, X_val_neg, y_train, y_val)
        val_data = (X_val_tok, X_val_tok, X_val_neg, X_val_neg, y_val, y_val)

        # Build model
        hybrid_model.build_model()

        # Train model
        history = hybrid_model.train(train_data, val_data, class_weights, args.output_dir)

        # Evaluate model
        metrics = hybrid_model.evaluate(val_data)

        # Print results
        print("\n=== Model Evaluation ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1']:.4f}")
        print(f"Loss:      {metrics['loss']:.4f}")

        # Save metrics
        metrics_path = Path(args.output_dir) / "evaluation_metrics.txt"
        with open(metrics_path, 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("=" * 50 + "\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")

        logger.info(f"Metrics saved to {metrics_path}")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Suppress TensorFlow warnings
    tf.get_logger().setLevel('ERROR')
    main()