"""
step2_data_augmentation.py

Augments dataset with synthetic negated examples.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import re
import ast
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataAugmenter:
    """Augments dataset with negated examples."""

    def __init__(self, config_path="config.yaml"):
        """
        Initialize data augmenter.

        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract config
        self.aug_config = self.config['data_augmentation']
        self.negation_words = self.config['negation_detection']['negation_words']
        self.sentiment_words = self.config['negation_detection']['sentiment_words']

        self.augmentation_ratio = self.aug_config['augmentation_ratio']
        self.negation_word = self.aug_config['negation_word']
        self.random_state = self.aug_config['random_state']

    def clean_data(self, df):
        """
        Clean and prepare DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        logger.info("Cleaning data...")

        # Convert stringified lists to actual lists
        def safe_literal_eval(x):
            try:
                return ast.literal_eval(x)
            except:
                return x

        if 'TokenizedText' in df.columns:
            df['TokenizedText'] = df['TokenizedText'].apply(safe_literal_eval)

        # Clean embeddings if they exist
        if 'xlmr_embeddings' in df.columns:
            def clean_embedding(x):
                if isinstance(x, str):
                    # Clean brackets and extra spaces
                    x = re.sub(r'[\[\]\s]+', ' ', x)
                    x = re.sub(r'\.\.\.', '', x)
                    try:
                        return np.array([float(i) for i in x.split() if i])
                    except:
                        return np.zeros(768)  # Default empty embedding
                return x

            df['xlmr_embeddings'] = df['xlmr_embeddings'].apply(clean_embedding)
            logger.info("Embeddings cleaned and converted to arrays")

        # Normalize sentiment labels
        if 'NewSentiment' in df.columns:
            df['SentimentNormalized'] = df['NewSentiment'].str.lower().str.strip()

        return df

    def create_negated_version(self, row, tokens, sentiment_idx):
        """
        Create a negated version of a sentence.

        Args:
            row (pd.Series): Original row
            tokens (list): Tokenized text
            sentiment_idx (int): Index of sentiment word

        Returns:
            dict: New row with negated version
        """
        # Insert negation word before sentiment word
        negated_tokens = tokens[:sentiment_idx] + [self.negation_word] + tokens[sentiment_idx:]

        # Determine new sentiment (flip positive/negative)
        old_sentiment = row.get('SentimentNormalized', '')
        if old_sentiment in ['positive', 'pos']:
            new_sentiment = 'negative'
        elif old_sentiment in ['negative', 'neg']:
            new_sentiment = 'positive'
        else:
            new_sentiment = old_sentiment  # Keep neutral as neutral

        # Create new row
        new_row = row.copy()
        new_row['TokenizedText'] = negated_tokens
        new_row['OriginalTweet'] = ' '.join(negated_tokens)
        new_row['NewSentiment'] = new_sentiment
        new_row['has_negation'] = 1
        new_row['negation_scopes'] = [(sentiment_idx, min(sentiment_idx + 3, len(negated_tokens)))]

        # Clear embeddings (will be recomputed if needed)
        if 'xlmr_embeddings' in new_row:
            new_row['xlmr_embeddings'] = None

        return new_row

    def augment_row(self, row):
        """
        Augment a single row if it contains sentiment words.

        Args:
            row (pd.Series): Input row

        Returns:
            dict or None: Augmented row if applicable
        """
        tokens = row['TokenizedText']

        # Find first sentiment word
        for i, token in enumerate(tokens):
            if token in self.sentiment_words:
                return self.create_negated_version(row, tokens, i)

        return None

    def augment_dataframe(self, df):
        """
        Augment DataFrame with negated examples.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Augmented DataFrame
        """
        logger.info("Starting data augmentation...")

        # Select candidates (sentences without negation and clear sentiment)
        candidates = df[
            (df['has_negation'] == 0) &
            (df['SentimentNormalized'].isin(['positive', 'negative', 'pos', 'neg']))
            ].copy()

        logger.info(f"Found {len(candidates)} eligible candidates")

        # Generate augmented rows
        augmented_rows = []
        for _, row in tqdm(candidates.iterrows(), total=len(candidates), desc="Augmenting"):
            new_row = self.augment_row(row)
            if new_row is not None:
                augmented_rows.append(new_row)

        logger.info(f"Generated {len(augmented_rows)} synthetic examples")

        # Select subset based on augmentation ratio
        if augmented_rows:
            n_samples = min(int(len(df) * self.augmentation_ratio), len(augmented_rows))
            augmented_df = pd.DataFrame(augmented_rows).sample(
                n=n_samples,
                random_state=self.random_state
            )

            # Forward fill embeddings if needed
            if 'xlmr_embeddings' in df.columns:
                augmented_df['xlmr_embeddings'] = augmented_df['xlmr_embeddings'].fillna(method='ffill')

            # Combine with original
            final_df = pd.concat([df, augmented_df], ignore_index=True)

            logger.info(f"Original: {len(df)} | Augmented: {len(augmented_df)} | Total: {len(final_df)}")

            return final_df
        else:
            logger.warning("No augmented samples generated")
            return df

    def analyze_dataset(self, df):
        """Analyze and log dataset statistics."""
        logger.info("\n=== Dataset Analysis ===")
        logger.info(f"Total samples: {len(df)}")

        if 'has_negation' in df.columns:
            logger.info(f"Samples with negation: {df['has_negation'].sum()}")
            logger.info(f"Negation rate: {df['has_negation'].mean():.2%}")

        if 'SentimentNormalized' in df.columns:
            logger.info("\nSentiment distribution:")
            sentiment_counts = df['SentimentNormalized'].value_counts()
            for sentiment, count in sentiment_counts.items():
                logger.info(f"  {sentiment}: {count} ({count / len(df):.2%})")


def main():
    """Command-line interface for data augmentation."""
    parser = argparse.ArgumentParser(
        description='Augment Urdu dataset with negated examples'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file (from Step 1)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='augmented_dataset_with_negation.csv',
        help='Path to output CSV file'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Analyze dataset before augmentation'
    )

    args = parser.parse_args()

    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        # Load data
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df)} rows")

        # Initialize augmenter
        augmenter = DataAugmenter(args.config)

        # Clean data
        df_clean = augmenter.clean_data(df)

        # Analyze if requested
        if args.analyze:
            augmenter.analyze_dataset(df_clean)

        # Augment data
        augmented_df = augmenter.augment_dataframe(df_clean)

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        augmented_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved augmented data to {output_path}")

        # Final analysis
        print("\n=== Augmentation Summary ===")
        print(f"Original size: {len(df_clean)}")
        print(f"Augmented size: {len(augmented_df)}")
        print(f"Added samples: {len(augmented_df) - len(df_clean)}")

        if 'SentimentNormalized' in augmented_df.columns:
            print("\nFinal sentiment distribution:")
            print(augmented_df['SentimentNormalized'].value_counts())

    except Exception as e:
        logger.error(f"Augmentation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()