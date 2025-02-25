"""
Script to evaluate the emotion recognition pipeline.

This script evaluates both the DAV regression and emotion mapping components
of the emotion recognition pipeline.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.dav_regression import DAVRegressor
from src.models.emotion_mapper import EmotionMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_dav_regression(test_data, model_dir, output_dir=None):
    """
    Evaluate the DAV regression model.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data with features and DAV values
    model_dir : str or Path
        Directory containing the trained DAV regression model
    output_dir : str or Path, optional
        Directory to save evaluation results
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Create output directory if provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load DAV regression model
    logger.info(f"Loading DAV regression model from {model_dir}")
    regressor = DAVRegressor.load(model_dir)
    
    # Extract feature columns
    feature_cols = [col for col in test_data.columns if col.startswith('feature_')]
    target_cols = ['dominance', 'arousal', 'valence']
    
    # Evaluate model
    logger.info("Evaluating DAV regression model...")
    metrics = regressor.evaluate(test_data, feature_cols=feature_cols)
    
    # Plot predictions
    if output_dir is not None:
        regressor.plot_predictions(test_data, feature_cols=feature_cols, output_dir=output_dir)
    
    return metrics

def evaluate_emotion_mapping(test_data, model_dir, output_dir=None):
    """
    Evaluate the emotion mapping model.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data with DAV values and emotion labels
    model_dir : str or Path
        Directory containing the trained emotion mapper
    output_dir : str or Path, optional
        Directory to save evaluation results
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Create output directory if provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load emotion mapper
    logger.info(f"Loading emotion mapper from {model_dir}")
    mapper = EmotionMapper.load(model_dir)
    
    # Extract DAV values and emotion labels
    dav_cols = ['dominance', 'arousal', 'valence']
    emotion_col = 'emotion'
    
    # Evaluate model
    logger.info("Evaluating emotion mapping model...")
    metrics = mapper.evaluate(test_data[dav_cols], test_data[emotion_col])
    
    # Plot confusion matrix
    if output_dir is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=mapper.emotion_categories,
                    yticklabels=mapper.emotion_categories)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / "emotion_confusion_matrix.png")
        plt.close()
    
    return metrics

def evaluate_full_pipeline(test_data, dav_model_dir, emotion_model_dir, output_dir=None):
    """
    Evaluate the full emotion recognition pipeline.
    
    Parameters:
    -----------
    test_data : pandas.DataFrame
        Test data with features, DAV values, and emotion labels
    dav_model_dir : str or Path
        Directory containing the trained DAV regression model
    emotion_model_dir : str or Path
        Directory containing the trained emotion mapper
    output_dir : str or Path, optional
        Directory to save evaluation results
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Create output directory if provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load models
    logger.info(f"Loading DAV regression model from {dav_model_dir}")
    regressor = DAVRegressor.load(dav_model_dir)
    
    logger.info(f"Loading emotion mapper from {emotion_model_dir}")
    mapper = EmotionMapper.load(emotion_model_dir)
    
    # Extract columns
    feature_cols = [col for col in test_data.columns if col.startswith('feature_')]
    dav_cols = ['dominance', 'arousal', 'valence']
    emotion_col = 'emotion'
    
    # Predict DAV values
    logger.info("Predicting DAV values...")
    dav_pred = regressor.predict(test_data[feature_cols])
    
    # Predict emotions from predicted DAV values
    logger.info("Predicting emotions from predicted DAV values...")
    emotion_pred = mapper.predict(dav_pred)
    
    # Calculate metrics
    # DAV regression metrics
    dav_metrics = {}
    for i, col in enumerate(dav_cols):
        mse = mean_squared_error(test_data[col], dav_pred[:, i])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data[col], dav_pred[:, i])
        r2 = r2_score(test_data[col], dav_pred[:, i])
        
        dav_metrics[col] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"DAV regression metrics for {col}:")
        logger.info(f"  MSE: {mse:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
    
    # Emotion classification metrics
    accuracy = accuracy_score(test_data[emotion_col], emotion_pred)
    report = classification_report(test_data[emotion_col], emotion_pred, output_dict=True)
    cm = confusion_matrix(test_data[emotion_col], emotion_pred)
    
    logger.info(f"Emotion classification accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    if output_dir is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(test_data[emotion_col]),
                    yticklabels=np.unique(test_data[emotion_col]))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Full Pipeline Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / "full_pipeline_confusion_matrix.png")
        plt.close()
    
    # Return metrics
    return {
        'dav_metrics': dav_metrics,
        'emotion_accuracy': accuracy,
        'emotion_report': report,
        'emotion_confusion_matrix': cm
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate the emotion recognition pipeline')
    parser.add_argument('--test_data', type=str, default='data/synthetic/small_val.csv',
                        help='Path to the test data CSV file')
    parser.add_argument('--dav_model_dir', type=str, default='models/dav_regression',
                        help='Directory containing the trained DAV regression model')
    parser.add_argument('--emotion_model_dir', type=str, default='models/emotion_mapper',
                        help='Directory containing the trained emotion mapper')
    parser.add_argument('--output_dir', type=str, default='evaluation/results',
                        help='Directory to save evaluation results')
    parser.add_argument('--evaluate_components', action='store_true',
                        help='Evaluate individual components separately')
    
    args = parser.parse_args()
    
    # Load test data
    test_data = pd.read_csv(args.test_data)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Evaluate individual components if requested
    if args.evaluate_components:
        logger.info("Evaluating DAV regression component...")
        dav_metrics = evaluate_dav_regression(
            test_data, 
            args.dav_model_dir, 
            output_dir / 'dav_regression'
        )
        
        logger.info("Evaluating emotion mapping component...")
        emotion_metrics = evaluate_emotion_mapping(
            test_data, 
            args.emotion_model_dir, 
            output_dir / 'emotion_mapping'
        )
    
    # Evaluate full pipeline
    logger.info("Evaluating full pipeline...")
    pipeline_metrics = evaluate_full_pipeline(
        test_data, 
        args.dav_model_dir, 
        args.emotion_model_dir, 
        output_dir / 'full_pipeline'
    )
    
    # Save metrics to file
    with open(output_dir / 'evaluation_summary.txt', 'w') as f:
        f.write("Emotion Recognition Pipeline Evaluation\n")
        f.write("=====================================\n\n")
        
        f.write("DAV Regression Metrics:\n")
        f.write("---------------------\n")
        for col, metrics in pipeline_metrics['dav_metrics'].items():
            f.write(f"{col}:\n")
            f.write(f"  MSE: {metrics['mse']:.4f}\n")
            f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
            f.write(f"  MAE: {metrics['mae']:.4f}\n")
            f.write(f"  R²: {metrics['r2']:.4f}\n\n")
        
        f.write("Emotion Classification Metrics:\n")
        f.write("-----------------------------\n")
        f.write(f"Accuracy: {pipeline_metrics['emotion_accuracy']:.4f}\n\n")
        
        f.write("Classification Report:\n")
        report = pipeline_metrics['emotion_report']
        for label, metrics in report.items():
            if isinstance(metrics, dict):
                f.write(f"{label}:\n")
                f.write(f"  Precision: {metrics['precision']:.4f}\n")
                f.write(f"  Recall: {metrics['recall']:.4f}\n")
                f.write(f"  F1-score: {metrics['f1-score']:.4f}\n")
                f.write(f"  Support: {metrics['support']}\n\n")
    
    logger.info(f"Evaluation results saved to {output_dir}")

if __name__ == '__main__':
    main()
