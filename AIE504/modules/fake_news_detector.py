import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class FakeNewsDetector:
    """Demo class for making predictions on new articles"""

    def __init__(self, model, preprocessor, feature_extractor, model_type='xgboost', use_pos_features=False):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.model_type = model_type
        self.use_pos_features = use_pos_features

    def predict_article(self, title, text, verbose=False):
        # Preprocess
        title_clean = self.preprocessor.preprocess(title)
        text_clean = self.preprocessor.preprocess(text)
        combined = title_clean + ' ' + text_clean

        if verbose:
            print(f"\n[DEBUG] Preprocessed text (first 200 chars): {combined[:200]}...")
            print(f"[DEBUG] Total tokens after preprocessing: {len(combined.split())}")

        # Extract features based on model type
        if self.model_type == 'xgboost':
            X = self.feature_extractor.tfidf_vectorizer.transform([combined])
            X = self.feature_extractor.svd.transform(X)

            # Add POS features if enabled
            if self.use_pos_features and self.preprocessor.use_pos_features:
                original_text = title + ' ' + text
                pos_features = self.preprocessor.extract_pos_features(original_text)
                pos_array = np.array([[pos_features[cat] for cat in self.preprocessor.pos_categories]])
                X = np.hstack([X, pos_array])
                if verbose:
                    print(f"[DEBUG] POS features: {pos_features}")

            if verbose:
                print(f"[DEBUG] Feature vector shape: {X.shape}")
        else:  # text_cnn
            sequences = self.feature_extractor.tokenizer.texts_to_sequences([combined])
            X = pad_sequences(sequences, maxlen=500, padding='post', truncating='post')
            if verbose:
                # Count how many tokens were recognized vs OOV
                original_tokens = combined.split()
                recognized = sum(1 for t in original_tokens if t in self.feature_extractor.tokenizer.word_index)
                print(
                    f"[DEBUG] Tokens recognized by tokenizer: {recognized}/{len(original_tokens)} ({recognized / len(original_tokens) * 100:.1f}%)")

        prediction = self.model.predict(X)[0]
        probability = self.model.predict_probabilities(X)[0]

        if verbose:
            print(f"[DEBUG] Raw prediction: {prediction}")
            print(f"[DEBUG] Raw probability (P(class=1)=P(FAKE)): {probability:.4f}")

        # IMPORTANT: Dataset documentation is INCORRECT!
        # Actual data inspection shows:
        #   Label 0 = REAL news (Reuters, professional sources)
        #   Label 1 = FAKE news (sensationalist, clickbait, political attacks)
        # probability = P(class 1) = P(FAKE)
        fake_prob = probability
        legit_prob = 1 - probability

        result = {'prediction': 'FAKE' if prediction == 1 else 'LEGIT',
            'confidence': float(fake_prob if prediction == 1 else legit_prob), 'fake_probability': float(fake_prob),
            'legit_probability': float(legit_prob)}

        return result

    def demo_prediction(self):
        print("\n" + "=" * 80)
        print("FAKE NEWS DETECTION - INTERACTIVE DEMO")
        print("=" * 80 + "\n")

        # TODO: Test articles
        examples = [{'title': 'Scientists Discover Cure for Common Cold',
                     'text': 'Researchers at MIT have announced a breakthrough in treating the common cold...',
                     'actual': 'This is an example - would need verification'},
                    {'title': 'President Announces New Policy',
                     'text': 'In a press conference today, officials outlined new measures...',
                     'actual': 'This is an example - would need verification'}]

        for i, example in enumerate(examples, 1):
            print(f"Example {i}:")
            print(f"Title: {example['title']}")
            print(f"Text: {example['text'][:100]}...")

            result = self.predict_article(example['title'], example['text'])

            print(f"\nPrediction: {result['prediction']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"  - Fake probability: {result['fake_probability']:.2%}")
            print(f"  - Legit probability: {result['legit_probability']:.2%}")
            print("-" * 80 + "\n")
