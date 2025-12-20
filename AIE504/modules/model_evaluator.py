import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
                             classification_report, roc_auc_score, roc_curve)


class ModelEvaluator:
    """Model evaluation"""

    @staticmethod
    def evaluate(y_true, y_pred, y_pred_proba=None, model_name="Model"):
        print(f"\n{'=' * 80}")
        print(f"EVALUATION: {model_name}")
        print("=" * 80)

        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        print(f"\n  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        # ROC-AUC if probabilities provided
        roc_auc = None

        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            print(f"  ROC-AUC:   {roc_auc:.4f}")

        print(f"\n  Classification Report:")
        # Actual data: Label 0 = Legit, Label 1 = Fake (opposite of documentation)
        print(classification_report(y_true, y_pred, target_names=['Legit', 'Fake']))

        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
                'roc_auc': roc_auc if y_pred_proba is not None else None}

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, model_name="Model"):
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        # Actual data: Label 0 = Legit, Label 1 = Fake (opposite of documentation)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Fake'], yticklabels=['Legit', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()

        return cm

    @staticmethod
    def plot_roc_curve(y_true, y_pred_proba, model_name="Model"):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def compare_models(results_dict):
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80 + "\n")

        # Create comparison dataframe
        df_results = pd.DataFrame(results_dict).T
        df_results = df_results[['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]

        print(df_results.to_string())

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Bar plot
        df_results[['accuracy', 'precision', 'recall', 'f1']].plot(kind='bar', ax=axes[0], rot=0)
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_ylabel('Score')
        axes[0].set_ylim([0, 1])
        axes[0].legend(loc='lower right')
        axes[0].grid(axis='y', alpha=0.3)

        # ROC-AUC comparison
        if 'roc_auc' in df_results.columns and not df_results['roc_auc'].isna().all():
            df_results['roc_auc'].plot(kind='bar', ax=axes[1], color='green', rot=0)
            axes[1].set_title('ROC-AUC Score Comparison')
            axes[1].set_ylabel('ROC-AUC')
            axes[1].set_ylim([0, 1])
            axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.show()

        return df_results
