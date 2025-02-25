"""
Baseline model for direct emotion classification from audio features.

This model directly categorizes emotions from audio features without
using the intermediate DAV (dominance, arousal, valence) representation.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def train_emotion_classifier(train_data, val_data=None, model_type='random_forest', output_dir=None):
    """
    Train a classifier to predict emotion categories directly from audio features.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with features and emotion labels
    val_data : pandas.DataFrame, optional
        Validation data with features and emotion labels
    model_type : str, optional
        Type of classifier to use ('random_forest' or 'svm')
    output_dir : str or Path, optional
        Directory to save model and results
    
    Returns:
    --------
    dict
        Dictionary containing trained model and evaluation results
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract features and target
    feature_cols = [col for col in train_data.columns if col.startswith('feature_')]
    target_col = 'emotion'
    
    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    
    if val_data is not None:
        X_val = val_data[feature_cols].values
        y_val = val_data[target_col].values
    else:
        # Split training data if validation data not provided
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize model
    print(f"\nTraining {model_type} classifier for emotion prediction...")
    
    if model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42
        )
    elif model_type == 'svm':
        model = SVC(
            kernel='rbf', 
            C=1.0, 
            probability=True, 
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, train_pred)
    val_accuracy = accuracy_score(y_val, val_pred)
    
    print(f"  Training accuracy: {train_accuracy:.4f}")
    print(f"  Validation accuracy: {val_accuracy:.4f}")
    
    # Generate classification report
    val_report = classification_report(y_val, val_pred)
    print("\nClassification Report (Validation):")
    print(val_report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_val, val_pred)
    
    # Save model and results if output directory provided
    if output_dir is not None:
        # Save model and scaler
        joblib.dump(model, output_dir / f"emotion_classifier_{model_type}.pkl")
        joblib.dump(scaler, output_dir / "feature_scaler.pkl")
        
        # Save metrics
        with open(output_dir / "classification_report.txt", "w") as f:
            f.write("Classification Report (Validation):\n")
            f.write(val_report)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_val),
                    yticklabels=np.unique(y_val))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix (Validation)\nAccuracy: {val_accuracy:.4f}')
        plt.tight_layout()
        plt.savefig(output_dir / "confusion_matrix.png")
        
        # Plot feature importances if random forest
        if model_type == 'random_forest':
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(indices[:20])), importances[indices[:20]])
            plt.xticks(range(len(indices[:20])), [f'feature_{i}' for i in indices[:20]], rotation=90)
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title('Top 20 Feature Importances')
            plt.tight_layout()
            plt.savefig(output_dir / "feature_importances.png")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy,
        'val_report': val_report,
        'confusion_matrix': cm
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a baseline emotion classifier')
    parser.add_argument('--train_data', type=str, default='data/synthetic/small_train.csv',
                        help='Path to the training data CSV file')
    parser.add_argument('--val_data', type=str, default='data/synthetic/small_val.csv',
                        help='Path to the validation data CSV file')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'svm'],
                        help='Type of classifier to use')
    parser.add_argument('--output_dir', type=str, default='models/baseline/emotion_classifier',
                        help='Directory to save model and results')
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_csv(args.train_data)
    val_data = pd.read_csv(args.val_data)
    
    # Train model
    train_emotion_classifier(
        train_data, 
        val_data, 
        model_type=args.model_type, 
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
