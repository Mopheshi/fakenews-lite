import xgboost as xgb
from sklearn.model_selection import GridSearchCV


class XGBoostModel:
    """XGBoost classifier for fake news detection"""

    def __init__(self):
        self.model = None
        self.best_params = None

    def train(self, x_train, y_train, x_val=None, y_val=None, tune_hyperparameters=True):
        print("\nTRAINING XGBoost MODEL...")
        print("-" * 80)

        if tune_hyperparameters:
            print("  Performing hyperparameter tuning...")

            param_grid = {'max_depth': [5, 7, 9], 'learning_rate': [0.01, 0.1, 0.2], 'n_estimators': [100, 200, 300],
                          'min_child_weight': [1, 3, 5], 'gamma': [0, 0.1, 0.2], 'subsample': [0.8, 0.9, 1.0],
                          'colsample_bytree': [0.8, 0.9, 1.0]}

            xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=5, n_jobs=-1,
                                          eval_metric='logloss')

            grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)

            grid_search.fit(x_train, y_train)

            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_

            print(f"\n  Best parameters: {self.best_params}")
            print(f"  Best CV score: {grid_search.best_score_:.4f}")

        else:
            # Otherwise, use default parameters
            self.model = xgb.XGBClassifier(max_depth=7, learning_rate=0.1, n_estimators=200, min_child_weight=3,
                                           gamma=0.1, subsample=0.9, colsample_bytree=0.9, objective='binary:logistic',
                                           random_state=5, n_jobs=-1, eval_metric='logloss')

            if x_val is not None and y_val is not None:
                eval_set = [(x_train, y_train), (x_val, y_val)]
                self.model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
            else:
                self.model.fit(x_train, y_train)

            print("  Model trained with default parameters")

        return self.model

    def predict(self, x):
        return self.model.predict(x)

    def predict_probabilities(self, x):
        return self.model.predict_proba(x)[:, 1]
