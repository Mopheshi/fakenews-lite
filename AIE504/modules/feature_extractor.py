import numpy as np
from keras.src.utils import pad_sequences
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer


class FeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.svd = None
        self.tokenizer = None
        self.embedding_matrix = None

    def create_tfidf_features(self, texts, max_features=10000, ngram_range=(1, 3)):
        print("\nEXTRACTING TF-IDF FEATURES...")
        print("-" * 80)

        self.tfidf_vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=2,
                                                max_df=0.9, sublinear_tf=True)

        X_tfidf = self.tfidf_vectorizer.fit_transform(texts)

        print(f"  TF-IDF matrix shape: {X_tfidf.shape}")
        print(f"  Vocabulary size: {len(self.tfidf_vectorizer.vocabulary_)}")

        return X_tfidf

    def apply_svd(self, x_tfidf, n_components=300):
        """
        Apply Truncated SVD for dimensionality reduction
        Required for XGBoost to handle sparse matrices efficiently
        """
        print(f"\n  Applying SVD (n_components={n_components})...")

        self.svd = TruncatedSVD(n_components=n_components, random_state=5)
        X_reduced = self.svd.fit_transform(x_tfidf)

        explained_var = self.svd.explained_variance_ratio_.sum()
        print(f"  Reduced shape: {X_reduced.shape}")
        print(f"  Explained variance: {explained_var:.4f}")

        return X_reduced

    def prepare_sequences(self, texts, max_words=50000, max_len=500):
        print("\nPREPARING SEQUENCES FOR TextCNN...")
        print("-" * 80)

        self.tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)

        sequences = self.tokenizer.texts_to_sequences(texts)
        X_seq = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

        vocab_size = min(len(self.tokenizer.word_index) + 1, max_words)

        print(f"  Sequences prepared")
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  Sequence shape: {X_seq.shape}")

        return X_seq, vocab_size

    def load_glove_embeddings(self, glove_path, embedding_dim=100):
        print(f"\n  Loading GloVe embeddings from {glove_path}...")

        embeddings_index = {}

        try:
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    coefficients = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefficients

            print(f"  Loaded {len(embeddings_index)} word vectors")

            # Create embedding matrix - respect the tokeniser's num_words limit
            max_words = self.tokenizer.num_words if self.tokenizer.num_words else len(self.tokenizer.word_index) + 1
            vocab_size = min(len(self.tokenizer.word_index) + 1, max_words)
            self.embedding_matrix = np.zeros((vocab_size, embedding_dim))

            hits = 0
            for word, i in self.tokenizer.word_index.items():
                if i < vocab_size:
                    embedding_vector = embeddings_index.get(word)
                    if embedding_vector is not None:
                        self.embedding_matrix[i] = embedding_vector
                        hits += 1

            coverage = hits / vocab_size
            print(f"  Embedding matrix created: {self.embedding_matrix.shape}")
            print(f"  Vocabulary coverage: {coverage:.2%}")

            return self.embedding_matrix

        except FileNotFoundError:
            print(f"  GloVe file not found at {glove_path}")
            return None

    def combine_features(self, svd_features, pos_features_df):
        """
        Combine SVD features with POS features for XGBoost

        Args:
            svd_features: numpy array of SVD-reduced TF-IDF features
            pos_features_df: DataFrame containing POS feature columns (pos_nouns, pos_verbs, etc.)

        Returns:
            Combined feature matrix
        """
        print("\n  Combining SVD and POS features...")

        # Get POS feature columns
        pos_cols = [col for col in pos_features_df.columns if col.startswith('pos_')]

        if not pos_cols:
            print("  No POS features found, returning SVD features only")
            return svd_features

        pos_array = pos_features_df[pos_cols].values
        combined = np.hstack([svd_features, pos_array])

        print(f"  SVD features: {svd_features.shape[1]}")
        print(f"  POS features: {pos_array.shape[1]}")
        print(f"  Combined shape: {combined.shape}")

        return combined
