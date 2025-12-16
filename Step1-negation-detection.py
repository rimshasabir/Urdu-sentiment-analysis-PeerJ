"""
step1_negation_detection.py

Detects negation scopes in Urdu text and adjusts XLM-R embeddings.
"""

import argparse
import yaml
import pandas as pd
import numpy as np
import torch
import ast
import logging
from pathlib import Path
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from tqdm import tqdm
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NegationDetector:
    """Detects and handles negation in Urdu text."""

    def __init__(self, config_path="config.yaml"):
        """
        Initialize negation detector.

        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract config
        self.negation_config = self.config['negation_detection']
        self.negation_words = self.negation_config['negation_words']
        self.sentiment_words = self.negation_config['sentiment_words']
        self.max_scope_length = self.negation_config['max_scope_length']
        self.scope_terminators = self.negation_config['scope_terminators']

        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        self.tokenizer = XLMRobertaTokenizer.from_pretrained(
            self.negation_config['model_name']
        )
        self.model = XLMRobertaModel.from_pretrained(
            self.negation_config['model_name']
        ).to(self.device)
        self.model.eval()

        # Cache for negation deltas
        self.delta_cache = {}

    def clear_cache(self):
        """Clear GPU cache."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def get_negation_scopes(self, tokens):
        """
        Detect negation scopes in tokenized text.

        Args:
            tokens (list): List of tokens

        Returns:
            list: List of (start, end) scope tuples
        """
        scopes = []
        for i, token in enumerate(tokens):
            if token in self.negation_words:
                start = i + 1
                end = min(i + self.max_scope_length, len(tokens))

                # Look for scope terminators
                for j in range(start, len(tokens)):
                    if tokens[j] in self.scope_terminators:
                        end = j
                        break

                if start < end:
                    scopes.append((start, end))

        return scopes

    def get_negation_delta(self, phrase, negation_word="نہیں"):
        """
        Compute delta vector for a phrase when negated.

        Args:
            phrase (str): Input phrase
            negation_word (str): Negation word to use

        Returns:
            np.ndarray: Delta vector
        """
        # Original phrase embedding
        inputs = self.tokenizer(
            phrase,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            orig_embed = self.model(**inputs).last_hidden_state
            orig_embed = orig_embed.mean(dim=1).cpu().numpy()

        # Negated phrase embedding
        negated_phrase = f"{negation_word} {phrase}"
        inputs = self.tokenizer(
            negated_phrase,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        with torch.no_grad():
            negated_embed = self.model(**inputs).last_hidden_state
            negated_embed = negated_embed.mean(dim=1).cpu().numpy()

        # Compute delta
        delta = negated_embed - orig_embed
        return delta.flatten()

    def precompute_deltas(self):
        """Precompute delta vectors for sentiment words."""
        logger.info("Precomputing negation deltas...")
        for word in tqdm(self.sentiment_words, desc="Computing deltas"):
            self.delta_cache[word] = self.get_negation_delta(word)
        logger.info(f"Precomputed deltas for {len(self.delta_cache)} words")

    def adjust_embeddings(self, tokens, embeddings, scopes):
        """
        Adjust embeddings for tokens in negation scopes.

        Args:
            tokens (list): List of tokens
            embeddings (np.ndarray): Original embeddings
            scopes (list): Negation scopes

        Returns:
            np.ndarray: Adjusted embeddings
        """
        if not scopes:
            return embeddings

        adjusted = embeddings.copy()

        for start, end in scopes:
            for i in range(start, min(end, len(tokens))):
                if i < len(tokens) and tokens[i] in self.delta_cache:
                    # Apply delta vector
                    if i < len(adjusted):
                        adjusted[i] += self.delta_cache[tokens[i]]

        return adjusted

    def process_dataframe(self, df):
        """
        Process DataFrame with negation detection.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: Enhanced DataFrame
        """
        logger.info("Processing DataFrame for negation detection...")

        # Ensure TokenizedText is a list
        if 'TokenizedText' in df.columns:
            df['TokenizedText'] = df['TokenizedText'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )

        # Check for negation
        df["has_negation"] = df["TokenizedText"].apply(
            lambda tokens: int(any(token in self.negation_words for token in tokens))
        )

        # Get negation scopes
        df["negation_scopes"] = df["TokenizedText"].apply(self.get_negation_scopes)

        # Precompute deltas
        self.precompute_deltas()

        # Adjust embeddings if they exist
        if 'xlmr_embeddings' in df.columns:
            logger.info("Adjusting embeddings for negation...")

            def adjust_row(row):
                tokens = row['TokenizedText']
                scopes = row['negation_scopes']

                if not scopes or 'xlmr_embeddings' not in row:
                    return row['xlmr_embeddings']

                try:
                    embeddings = np.array(row['xlmr_embeddings'])
                    adjusted = self.adjust_embeddings(tokens, embeddings, scopes)
                    return adjusted.tolist()
                except Exception as e:
                    logger.warning(f"Failed to adjust embeddings: {e}")
                    return row['xlmr_embeddings']

            tqdm.pandas(desc="Adjusting embeddings")
            df["xlmr_embeddings_negadjusted"] = df.progress_apply(adjust_row, axis=1)

        logger.info(f"Found {df['has_negation'].sum()} sentences with negation")
        return df


def main():
    """Command-line interface for negation detection."""
    parser = argparse.ArgumentParser(
        description='Detect negation in Urdu text and adjust embeddings'
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to input CSV file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='dataset_with_negation.csv',
        help='Path to output CSV file'
    )

    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

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

        # Initialize detector
        detector = NegationDetector(args.config)

        # Process data
        processed_df = detector.process_dataframe(df)

        # Save output
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved processed data to {output_path}")

        # Show summary
        print("\n=== Negation Detection Summary ===")
        print(f"Total sentences: {len(processed_df)}")
        print(f"Sentences with negation: {processed_df['has_negation'].sum()}")
        print(f"Negation rate: {processed_df['has_negation'].mean():.2%}")

        if 'negation_scopes' in processed_df.columns:
            total_scopes = processed_df['negation_scopes'].apply(len).sum()
            print(f"Total negation scopes detected: {total_scopes}")

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

    finally:
        # Clear cache
        detector.clear_cache()


if __name__ == "__main__":
    main()