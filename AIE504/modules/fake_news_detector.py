import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class FakeNewsDetector:
    """
    Unified detector that can use individual models or ensemble
    """

    def __init__(self, xgboost_model=None, textcnn_model=None,
                 preprocessor=None, feature_extractor_xgb=None,
                 feature_extractor_cnn=None, ensemble_model=None,
                 use_pos_features=False):
        """
        Args:
            xgboost_model: Trained XGBoost model
            textcnn_model: Trained TextCNN model
            preprocessor: Text preprocessor
            feature_extractor_xgb: Feature extractor for XGBoost
            feature_extractor_cnn: Feature extractor for TextCNN
            ensemble_model: Optional ensemble model to combine predictions
            use_pos_features: Whether to use POS features
        """
        self.xgboost_model = xgboost_model
        self.textcnn_model = textcnn_model
        self.preprocessor = preprocessor
        self.feature_extractor_xgb = feature_extractor_xgb
        self.feature_extractor_cnn = feature_extractor_cnn
        self.ensemble_model = ensemble_model
        self.use_pos_features = use_pos_features

    def _prepare_xgboost_features(self, combined_text, original_text):
        """Prepare features for XGBoost model"""
        X = self.feature_extractor_xgb.tfidf_vectorizer.transform([combined_text])
        X = self.feature_extractor_xgb.svd.transform(X)

        # Add POS features if enabled
        if self.use_pos_features and self.preprocessor.use_pos_features:
            pos_features = self.preprocessor.extract_pos_features(original_text)
            pos_array = np.array([[pos_features[cat] for cat in self.preprocessor.pos_categories]])
            X = np.hstack([X, pos_array])

        return X

    def _prepare_textcnn_features(self, combined_text):
        """Prepare features for TextCNN model"""
        sequences = self.feature_extractor_cnn.tokenizer.texts_to_sequences([combined_text])
        X = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
        return X

    def predict_article(self, title, text, verbose=False):
        """
        Predict if article is fake or legit

        Returns:
            Dictionary with prediction, confidence, and probabilities
        """
        # Preprocess
        title_clean = self.preprocessor.preprocess(title)
        text_clean = self.preprocessor.preprocess(text)
        combined = title_clean + ' ' + text_clean
        original_text = title + ' ' + text

        if verbose:
            print(f"\n[DEBUG] Preprocessed text (first 200 chars): {combined[:200]}...")

        # Prepare features for each model
        X_list = []
        model_predictions = {}

        if self.xgboost_model:
            X_xgb = self._prepare_xgboost_features(combined, original_text)
            X_list.append(X_xgb)

            xgb_pred = self.xgboost_model.predict(X_xgb)[0]
            xgb_proba = self.xgboost_model.predict_probabilities(X_xgb)[0]
            model_predictions['XGBoost'] = {
                'prediction': xgb_pred,
                'probability': xgb_proba
            }

        if self.textcnn_model:
            X_cnn = self._prepare_textcnn_features(combined)
            X_list.append(X_cnn)

            cnn_pred = self.textcnn_model.predict(X_cnn)[0]
            cnn_proba = self.textcnn_model.predict_probabilities(X_cnn)[0]
            model_predictions['TextCNN'] = {
                'prediction': cnn_pred,
                'probability': cnn_proba
            }

        # Use ensemble if available
        if self.ensemble_model and len(X_list) > 1:
            prediction = self.ensemble_model.predict(X_list)[0]
            probability = self.ensemble_model.predict_proba_ensemble(X_list)[0]

            if verbose:
                print(f"[DEBUG] Using ensemble method: {self.ensemble_model.method}")
                print(f"[DEBUG] Individual model predictions:")
                for name, pred in model_predictions.items():
                    print(f"  {name}: pred={pred['prediction']}, proba={pred['probability']:.4f}")
                print(f"[DEBUG] Ensemble: pred={prediction}, proba={probability:.4f}")
        else:
            # Use single model
            if self.xgboost_model:
                prediction = model_predictions['XGBoost']['prediction']
                probability = model_predictions['XGBoost']['probability']
            elif self.textcnn_model:
                prediction = model_predictions['TextCNN']['prediction']
                probability = model_predictions['TextCNN']['probability']
            else:
                raise ValueError("No model available for prediction")

        # Calculate final results
        # Label 0 = Legit, Label 1 = Fake
        fake_prob = float(probability)
        legit_prob = float(1 - probability)

        result = {
            'prediction': 'FAKE' if prediction == 1 else 'LEGIT',
            'confidence': float(fake_prob if prediction == 1 else legit_prob),
            'fake_probability': fake_prob,
            'legit_probability': legit_prob,
            'model_predictions': model_predictions if verbose else None
        }

        return result
