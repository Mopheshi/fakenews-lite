"""
Train and save the model for the web demo.
Run this script before starting the web application.
"""

import os
import sys
import pickle
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
from sklearn.model_selection import train_test_split

# Download NLTK data
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource in ['stopwords', 'wordnet', 'omw-1.4'] else f'tokenizers/{resource}' if 'punkt' in resource else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from modules.data_loader import DataLoader
from modules.text_processor import TextPreprocessor
from modules.feature_extractor import FeatureExtractor
from models.xgboost_model import XGBoostModel


def main():
    print("\n" + "=" * 60)
    print("TRAINING MODEL FOR WEB DEMO")
    print("=" * 60 + "\n")

    # Paths
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'files', 'WELFake_Dataset.csv')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_model.pkl')

    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        print("Please ensure WELFake_Dataset.csv is in the 'files' directory.")
        return False

    # Load data
    loader = DataLoader()
    df = loader.load_data(data_path)

    # Preprocess (without POS features for speed)
    preprocessor = TextPreprocessor(use_pos_features=False)
    df = preprocessor.preprocess_dataframe(df)

    # Prepare data
    X = df['combined'].values
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")

    # Extract features
    feature_extractor = FeatureExtractor()
    X_train_tfidf = feature_extractor.create_tfidf_features(X_train)
    X_train_svd = feature_extractor.apply_svd(X_train_tfidf)

    # Train model
    model = XGBoostModel()
    model.train(X_train_svd, y_train, tune_hyperparameters=False)

    # Evaluate on test set
    X_test_tfidf = feature_extractor.tfidf_vectorizer.transform(X_test)
    X_test_svd = feature_extractor.svd.transform(X_test_tfidf)

    y_pred = model.predict(X_test_svd)
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Save model
    print(f"\nSaving model to {save_path}...")

    saved_data = {
        'preprocessor': preprocessor,
        'feature_extractor': feature_extractor,
        'model': model
    }

    with open(save_path, 'wb') as f:
        pickle.dump(saved_data, f)

    print("Model saved successfully!")
    print("\n" + "=" * 60)
    print("You can now run the web app with: python app.py")
    print("=" * 60 + "\n")

    return True


if __name__ == '__main__':
    main()

