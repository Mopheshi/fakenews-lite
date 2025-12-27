import os
import pickle
import sys
import warnings

import numpy as np

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data
for resource in ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'punkt_tab',
                 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.data.find(f'corpora/{resource}' if resource in ['stopwords', 'wordnet', 'omw-1.4']
                       else f'tokenizers/{resource}' if 'punkt' in resource else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

from modules.data_loader import DataLoader
from modules.text_processor import TextPreprocessor
from modules.feature_extractor import FeatureExtractor
from models.xgboost_model import XGBoostModel
from models.text_cnn_model import TextCNNModel
from models.ensemble_model import EnsembleModel
from modules.model_evaluator import ModelEvaluator


def main():
    print("\n" + "=" * 60)
    print("TRAINING OPTIMISED ENSEMBLE MODEL FOR WEB DEMO")
    print("=" * 60 + "\n")

    # Paths
    data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'files', 'WELFake_Dataset.csv')
    glove_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'files', 'glove.6B.100d.txt')
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'saved_model.pkl')

    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return False

    # Configuration
    USE_POS_FEATURES = True  # POS extraction
    TRAIN_BOTH_MODELS = True
    ENSEMBLE_METHOD = 'soft_voting'  # The available options are: 'soft_voting', 'weighted_voting', 'stacking'

    # Load data
    loader = DataLoader()
    df = loader.load_data(data_path)

    # Preprocess with optimised preprocessor
    preprocessor = TextPreprocessor(use_pos_features=USE_POS_FEATURES)
    df = preprocessor.preprocess_dataframe(df)

    # Prepare data
    X = df['combined'].values
    y = df['label'].values
    df_indices = df.index.values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=5, stratify=y
    )
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_train, y_train, idx_train, test_size=0.125, random_state=5, stratify=y_train
    )

    if USE_POS_FEATURES:
        df_train_pos = df.loc[idx_train]
        df_val_pos = df.loc[idx_val]
        df_test_pos = df.loc[idx_test]

    print(f"\nTraining: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")

    # Train XGBoost
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODEL")
    print("=" * 60)

    feature_extractor_xgb = FeatureExtractor()
    X_train_tfidf = feature_extractor_xgb.create_tfidf_features(X_train)
    X_train_svd = feature_extractor_xgb.apply_svd(X_train_tfidf)

    X_val_tfidf = feature_extractor_xgb.tfidf_vectorizer.transform(X_val)
    X_val_svd = feature_extractor_xgb.svd.transform(X_val_tfidf)

    X_test_tfidf = feature_extractor_xgb.tfidf_vectorizer.transform(X_test)
    X_test_svd = feature_extractor_xgb.svd.transform(X_test_tfidf)

    if USE_POS_FEATURES:
        X_train_final_xgb = feature_extractor_xgb.combine_features(X_train_svd, df_train_pos)
        X_val_final_xgb = feature_extractor_xgb.combine_features(X_val_svd, df_val_pos)
        X_test_final_xgb = feature_extractor_xgb.combine_features(X_test_svd, df_test_pos)
    else:
        X_train_final_xgb = X_train_svd
        X_val_final_xgb = X_val_svd
        X_test_final_xgb = X_test_svd

    xgb_model = XGBoostModel()
    xgb_model.train(X_train_final_xgb, y_train, X_val_final_xgb, y_val,
                    tune_hyperparameters=False)

    # Evaluate XGBoost
    y_pred_xgb = xgb_model.predict(X_test_final_xgb)
    y_proba_xgb = xgb_model.predict_probabilities(X_test_final_xgb)
    results_xgb = ModelEvaluator.evaluate(y_test, y_pred_xgb, y_proba_xgb, "XGBoost")

    # Train TextCNN if requested
    text_cnn = None
    feature_extractor_cnn = None
    results_cnn = None

    if TRAIN_BOTH_MODELS:
        print("\n" + "=" * 60)
        print("TRAINING TEXTCNN MODEL")
        print("=" * 60)

        feature_extractor_cnn = FeatureExtractor()
        X_train_seq, vocab_size = feature_extractor_cnn.prepare_sequences(X_train)
        X_val_seq = pad_sequences(
            feature_extractor_cnn.tokenizer.texts_to_sequences(X_val),
            maxlen=500, padding='post', truncating='post'
        )
        X_test_seq = pad_sequences(
            feature_extractor_cnn.tokenizer.texts_to_sequences(X_test),
            maxlen=500, padding='post', truncating='post'
        )

        # Try to load GloVe frozen embeddings
        try:
            embedding_matrix = feature_extractor_cnn.load_glove_embeddings(glove_path, embedding_dim=100)
        except:
            embedding_matrix = None
            print("  Continuing without GloVe embeddings")

        text_cnn = TextCNNModel(vocab_size=vocab_size, embedding_dim=100, max_len=500,
                                embedding_matrix=embedding_matrix)
        text_cnn.build_model()
        text_cnn.train(X_train_seq, y_train, X_val_seq, y_val, epochs=10)

        # Evaluate TextCNN
        y_pred_cnn = text_cnn.predict(X_test_seq)
        y_proba_cnn = text_cnn.predict_probabilities(X_test_seq)
        results_cnn = ModelEvaluator.evaluate(y_test, y_pred_cnn, y_proba_cnn, "TextCNN")

    # Create ensemble if both models trained
    ensemble_model = None
    if TRAIN_BOTH_MODELS and text_cnn is not None:
        print("\n" + "=" * 60)
        print("CREATING ENSEMBLE MODEL")
        print("=" * 60)

        if ENSEMBLE_METHOD == 'weighted_voting':
            # Use F1 scores as weights
            weights = [results_xgb['f1'], results_cnn['f1']]
            print(f"Using F1-based weights: {weights}")

            ensemble_model = EnsembleModel(
                models=[xgb_model, text_cnn],
                model_names=['XGBoost', 'TextCNN'],
                method='weighted_voting',
                weights=weights
            )

        elif ENSEMBLE_METHOD == 'stacking':
            ensemble_model = EnsembleModel(
                models=[xgb_model, text_cnn],
                model_names=['XGBoost', 'TextCNN'],
                method='stacking'
            )

            # Train meta-model on validation set
            val_probas_xgb = xgb_model.predict_probabilities(X_val_final_xgb)
            val_probas_cnn = text_cnn.predict_probabilities(X_val_seq)
            X_meta_train = np.column_stack([val_probas_xgb, val_probas_cnn])

            ensemble_model.train_meta_model(X_meta_train, y_val)

        else:  # soft_voting
            ensemble_model = EnsembleModel(
                models=[xgb_model, text_cnn],
                model_names=['XGBoost', 'TextCNN'],
                method='soft_voting'
            )

        # Evaluate ensemble
        y_pred_ensemble = ensemble_model.predict([X_test_final_xgb, X_test_seq])
        y_proba_ensemble = ensemble_model.predict_proba_ensemble([X_test_final_xgb, X_test_seq])
        results_ensemble = ModelEvaluator.evaluate(
            y_test, y_pred_ensemble, y_proba_ensemble,
            f"Ensemble ({ENSEMBLE_METHOD})"
        )

        # Compare results
        print("\n" + "=" * 60)
        print("MODEL COMPARISON")
        print("=" * 60)
        results = {
            'XGBoost': results_xgb,
            'TextCNN': results_cnn,
            f'Ensemble_{ENSEMBLE_METHOD}': results_ensemble
        }
        ModelEvaluator.compare_models(results)

    # Save model
    print(f"\nSaving model to {save_path}...")

    saved_data = {
        'xgboost_model': xgb_model,
        'textcnn_model': text_cnn,
        'preprocessor': preprocessor,
        'feature_extractor_xgb': feature_extractor_xgb,
        'feature_extractor_cnn': feature_extractor_cnn,
        'ensemble_model': ensemble_model,
        'use_pos_features': USE_POS_FEATURES,
        'ensemble_method': ENSEMBLE_METHOD if ensemble_model else None
    }

    with open(save_path, 'wb') as f:
        pickle.dump(saved_data, f)

    preprocessor.clear_cache()

    print("Model saved successfully!")
    print("\n" + "=" * 60)
    print("You can now run the web app with: python app.py")
    print("=" * 60 + "\n")

    return True


if __name__ == '__main__':
    main()
