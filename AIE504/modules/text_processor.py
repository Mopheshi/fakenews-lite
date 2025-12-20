import re
import string
from collections import Counter

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove extra whitespace
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

    def extract_pos_features(self, text):
        """Extract POS-based features from text"""
        if not text or len(text.strip()) == 0:
            return {cat: 0.0 for cat in self.pos_categories}

        try:
            tokens = word_tokenize(text)
            if len(tokens) == 0:
                return {cat: 0.0 for cat in self.pos_categories}

            pos_tags = pos_tag(tokens)
            tag_counts = Counter(tag for word, tag in pos_tags)
            total_tokens = len(tokens)

            # Calculate ratio of each POS category
            features = {}
            for category, tags in self.pos_categories.items():
                count = sum(tag_counts.get(tag, 0) for tag in tags)
                features[category] = count / total_tokens

            return features
        except Exception:
            return {cat: 0.0 for cat in self.pos_categories}

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

        # Extract POS features if enabled
        if self.use_pos_features:
            print("  Extracting POS features (this may take a while)...")
            # Use original text for POS tagging (before preprocessing removes structure)
            combined_original = df['title'] + ' ' + df['text']
            pos_features = combined_original.apply(self.extract_pos_features)

            # Convert to separate columns
            for category in self.pos_categories:
                df[f'pos_{category}'] = pos_features.apply(lambda x: x.get(category, 0.0))

            print(f"  POS features extracted: {list(self.pos_categories.keys())}")

        # Remove empty documents
        df = df[df['combined'].str.strip() != '']

        print(f"  Preprocessing complete")
        print(f"  Final dataset size: {len(df)} articles")

        return df
