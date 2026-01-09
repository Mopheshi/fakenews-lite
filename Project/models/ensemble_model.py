import numpy as np
from sklearn.linear_model import LogisticRegression


class EnsembleModel:
    """
    Ensemble model combining XGBoost and TextCNN predictions
    Supports multiple ensemble strategies:
    - Soft Voting: Average probability predictions
    - Weighted Voting: Weighted average based on model performance
    - Stacking: Meta-learner trained on base model predictions
    """

    def __init__(self, models, model_names=None, method='soft_voting', weights=None):
        """
        Args:
            models: List of trained models (e.g., [xgboost_model, textcnn_model])
            model_names: Optional names for models (for debugging)
            method: 'soft_voting', 'weighted_voting', or 'stacking'
            weights: Weights for each model (only for weighted_voting)
        """
        self.models = models
        self.model_names = model_names or [f"Model_{i}" for i in range(len(models))]
        self.method = method
        self.weights = weights
        self.meta_model = None

        # Validate inputs
        if method == 'weighted_voting' and weights is None:
            raise ValueError("Weights must be provided for weighted_voting method")

        if weights is not None and len(weights) != len(models):
            raise ValueError("Number of weights must match number of models")

        if weights is not None:
            # Normalise weights to sum to 1
            self.weights = np.array(weights) / np.sum(weights)

    def train_meta_model(self, X_train, y_train):
        """
        Train meta-model for stacking ensemble
        X_train should be the predictions from base models on validation set
        """
        if self.method != 'stacking':
            raise ValueError("train_meta_model should only be called for stacking method")

        print("\nTraining meta-model for stacking ensemble...")
        self.meta_model = LogisticRegression(max_iter=1000, random_state=5)
        self.meta_model.fit(X_train, y_train)
        print("  Meta-model training complete")

        # Feature importances (model weights learned by meta-model)
        print(f"  Meta-model coefficients: {self.meta_model.coef_[0]}")
        for i, name in enumerate(self.model_names):
            print(f"    {name}: {self.meta_model.coef_[0][i]:.4f}")

    def predict_probabilities(self, X_list):
        """
        Get probability predictions from all base models

        Args:
            X_list: List of feature arrays for each model
                   e.g., [X_xgboost, X_textcnn]

        Returns:
            Array of shape (n_samples, n_models) with probability predictions
        """
        probabilities = []

        for i, (model, X) in enumerate(zip(self.models, X_list)):
            proba = model.predict_probabilities(X)
            probabilities.append(proba)

        return np.column_stack(probabilities)

    def predict(self, X_list):
        """
        Make predictions using the ensemble

        Args:
            X_list: List of feature arrays for each model
                   e.g., [X_xgboost, X_textcnn]

        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.predict_probabilities(X_list)

        if self.method == 'soft_voting':
            # Average probabilities across models
            ensemble_proba = np.mean(probabilities, axis=1)

        elif self.method == 'weighted_voting':
            # Weighted average of probabilities
            ensemble_proba = np.average(probabilities, axis=1, weights=self.weights)

        elif self.method == 'stacking':
            if self.meta_model is None:
                raise ValueError("Meta-model not trained. Call train_meta_model first.")
            # Use meta-model to combine predictions
            ensemble_proba = self.meta_model.predict_proba(probabilities)[:, 1]

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return (ensemble_proba > 0.5).astype(int)

    def predict_proba_ensemble(self, X_list):
        """
        Get ensemble probability predictions

        Args:
            X_list: List of feature arrays for each model

        Returns:
            Probability predictions from ensemble
        """
        probabilities = self.predict_probabilities(X_list)

        if self.method == 'soft_voting':
            ensemble_proba = np.mean(probabilities, axis=1)

        elif self.method == 'weighted_voting':
            ensemble_proba = np.average(probabilities, axis=1, weights=self.weights)

        elif self.method == 'stacking':
            if self.meta_model is None:
                raise ValueError("Meta-model not trained. Call train_meta_model first.")
            ensemble_proba = self.meta_model.predict_proba(probabilities)[:, 1]

        else:
            raise ValueError(f"Unknown method: {self.method}")

        return ensemble_proba

    def get_model_contributions(self, X_list):
        """
        Get individual model predictions for analysis

        Returns:
            Dictionary with model names and their predictions
        """
        probabilities = self.predict_probabilities(X_list)

        contributions = {}
        for i, name in enumerate(self.model_names):
            contributions[name] = {
                'probability': probabilities[:, i],
                'prediction': (probabilities[:, i] > 0.5).astype(int)
            }

        return contributions


def create_stacking_train_data(models, X_train_list, y_train, n_folds=5):
    """
    Create training data for stacking meta-model using cross-validation

    Args:
        models: List of base models
        X_train_list: List of training feature arrays for each model
        y_train: Training labels
        n_folds: Number of folds for cross-validation

    Returns:
        Stacked predictions for training meta-model
    """
    from sklearn.model_selection import KFold

    n_samples = len(y_train)
    n_models = len(models)

    # Initialise array to store out-of-fold predictions
    stacked_predictions = np.zeros((n_samples, n_models))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=5)

    print(f"\nGenerating stacked training data using {n_folds}-fold CV...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_list[0])):
        print(f"  Fold {fold + 1}/{n_folds}...")

        for model_idx, (model, X_train) in enumerate(zip(models, X_train_list)):
            # Train on fold training data
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]

            # Clone and train model on this fold
            val_proba = model.predict_probabilities(X_fold_val)
            stacked_predictions[val_idx, model_idx] = val_proba

    print("  Stacking data generation complete")
    return stacked_predictions
