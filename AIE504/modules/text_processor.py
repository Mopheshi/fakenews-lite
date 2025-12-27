import re
import string
from collections import Counter
from functools import lru_cache

import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())

    return text


class TextPreprocessor:
    def __init__(self, use_pos_features=False):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # I am keeping some negation words because they are important for sentiment sometimes
        self.stop_words -= {'no', 'not', 'nor', 'neither', 'never', 'none'}
        self.use_pos_features = use_pos_features

        # POS tag categories for feature extraction
        self.pos_categories = {
            'nouns': ['NN', 'NNS', 'NNP', 'NNPS'],
            'verbs': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
            'adjectives': ['JJ', 'JJR', 'JJS'],
            'adverbs': ['RB', 'RBR', 'RBS'],
            'pronouns': ['PRP', 'PRP$', 'WP', 'WP$'],
            'modals': ['MD'],
        }

        # Cache for POS features to avoid recomputation
        self._pos_cache = {}

    @lru_cache(maxsize=10000)
    def _cached_tokenize(self, text):
        """Cached tokenization to avoid redundant processing"""
        return tuple(word_tokenize(text))

    def extract_pos_features(self, text):
        """
        Optimised POS feature extraction with caching
        """
        if not text or len(text.strip()) == 0:
            return {cat: 0.0 for cat in self.pos_categories}

        # Check cache first
        text_hash = hash(text)
        if text_hash in self._pos_cache:
            return self._pos_cache[text_hash]

        try:
            tokens = self._cached_tokenize(text)
            if len(tokens) == 0:
                return {cat: 0.0 for cat in self.pos_categories}

            # Batch POS tagging is more efficient
            pos_tags = pos_tag(tokens)
            tag_counts = Counter(tag for word, tag in pos_tags)
            total_tokens = len(tokens)

            # Calculate ratio of each POS category
            features = {}
            for category, tags in self.pos_categories.items():
                count = sum(tag_counts.get(tag, 0) for tag in tags)
                features[category] = count / total_tokens

            # Cache the result
            self._pos_cache[text_hash] = features
            return features
        except Exception:
            return {cat: 0.0 for cat in self.pos_categories}

    def extract_pos_features_batch(self, texts):
        """
        Batch processing for POS features - more efficient for large datasets
        """
        print(f"  Extracting POS features for {len(texts)} texts in batch mode...")

        results = []
        batch_size = 1000

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.extract_pos_features(text) for text in batch]
            results.extend(batch_results)

            if (i + batch_size) % 5000 == 0:
                print(f"    Processed {min(i + batch_size, len(texts))}/{len(texts)} texts...")

        return results

    def remove_stopwords(self, text):
        words = text.split()
        return ' '.join([w for w in words if w not in self.stop_words])

    def lemmatize_text(self, text):
        words = text.split()
        return ' '.join([self.lemmatizer.lemmatize(w) for w in words])

    def preprocess(self, text):
        text = clean_text(text)
        text = self.remove_stopwords(text)
        text = self.lemmatize_text(text)
        return text

    def preprocess_dataframe(self, df):
        print("\nPREPROCESSING DATA...")
        print("-" * 80)

        df = df.copy()

        # Handle missing values
        df['title'] = df['title'].fillna('')
        df['text'] = df['text'].fillna('')

        print("  Cleaning title...")
        df['title_clean'] = df['title'].apply(self.preprocess)

        print("  Cleaning text...")
        df['text_clean'] = df['text'].apply(self.preprocess)

        print("  Combining title and text...")
        # Concatenate title and text (title gets more weight by appearing first)
        df['combined'] = df['title_clean'] + ' ' + df['text_clean']

        # Extract POS features if enabled (optimised batch processing)
        if self.use_pos_features:
            print("  Extracting POS features (using optimised batch processing)...")
            combined_original = (df['title'] + ' ' + df['text']).tolist()

            # Use batch processing for efficiency
            pos_features_list = self.extract_pos_features_batch(combined_original)

            # Convert to dataframe columns
            for category in self.pos_categories:
                df[f'pos_{category}'] = [feat.get(category, 0.0) for feat in pos_features_list]

            print(f"  POS features extracted: {list(self.pos_categories.keys())}")

        # Remove empty documents
        df = df[df['combined'].str.strip() != '']

        print(f"  Preprocessing complete")
        print(f"  Final dataset size: {len(df)} articles")
        print(f"  Cache size: {len(self._pos_cache)} entries")

        return df

    def clear_cache(self):
        """Clear the POS feature cache to free memory"""
        self._pos_cache.clear()
        self._cached_tokenize.cache_clear()
