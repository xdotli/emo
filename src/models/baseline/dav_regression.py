"""
Baseline model for predicting dominance, arousal, and valence (DAV) values from audio features.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def train_dav_regression_model(train_data, val_data=None, model_type='random_forest', output_dir=None):
    """
    Train a regression model to predict dominance, arousal, and valence (DAV) values.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with features and DAV values
    val_data : pandas.DataFrame, optional
        Validation data with features and DAV values
    model_type : str, optional
        Type of regression model to use ('random_forest', 'svr', or 'ridge')
    output_dir : str or Path, optional
        Directory to save model and results
    
    Returns:
    --------
    dict
        Dictionary containing trained models and evaluation results
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract features and targets
    feature_cols = [col for col in train_data.columns if col.startswith('feature_')]
    target_cols = ['dominance', 'arousal', 'valence']
    
    X_train = train_data[feature_cols].values
    y_train = train_data[target_cols].values
    
    if val_data is not None:
        X_val = val_data[feature_cols].values
        y_val = val_data[target_cols].values
    else:
        # Split training data if validation data not provided
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Initialize models for each target
    models = {}
    val_predictions = {}
    metrics = {}
    
    for i, target in enumerate(target_cols):
        print(f"\nTraining model for {target}...")
        
        # Select model type
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100, 
                max_depth=10, 
                random_state=42
            )
        elif model_type == 'svr':
            model = SVR(
                kernel='rbf', 
                C=1.0, 
                epsilon=0.1
            )
        elif model_type == 'ridge':
            model = Ridge(
                alpha=1.0, 
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train_scaled, y_train[:, i])
        
        # Make predictions
        train_pred = model.predict(X_train_scaled)
        val_pred = model.predict(X_val_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train[:, i], train_pred)
        val_mse = mean_squared_error(y_val[:, i], val_pred)
        train_r2 = r2_score(y_train[:, i], train_pred)
        val_r2 = r2_score(y_val[:, i], val_pred)
        
        print(f"  Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}")
        print(f"  Validation MSE: {val_mse:.4f}, R²: {val_r2:.4f}")
        
        # Store model and metrics
        models[target] = model
        val_predictions[target] = val_pred
        metrics[target] = {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_r2': train_r2,
            'val_r2': val_r2
        }
        
        # Save model if output directory provided
        if output_dir is not None:
            joblib.dump(model, output_dir / f"{target}_{model_type}_model.pkl")
    
    # Save scaler if output directory provided
    if output_dir is not None:
        joblib.dump(scaler, output_dir / "feature_scaler.pkl")
        
        # Plot actual vs predicted values
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, target in enumerate(target_cols):
            axes[i].scatter(y_val[:, i], val_predictions[target], alpha=0.5)
            axes[i].plot([-1, 1], [-1, 1], 'r--')
            axes[i].set_xlabel(f'Actual {target}')
            axes[i].set_ylabel(f'Predicted {target}')
            axes[i].set_title(f'{target.capitalize()} Prediction\nMSE: {metrics[target]["val_mse"]:.4f}, R²: {metrics[target]["val_r2"]:.4f}')
            axes[i].grid(True)
            axes[i].set_xlim([-1.1, 1.1])
            axes[i].set_ylim([-1.1, 1.1])
        
        plt.tight_layout()
        plt.savefig(output_dir / "dav_prediction_results.png")
        
        # Save metrics
        metrics_df = pd.DataFrame({
            'Target': target_cols,
            'Train MSE': [metrics[t]['train_mse'] for t in target_cols],
            'Val MSE': [metrics[t]['val_mse'] for t in target_cols],
            'Train R²': [metrics[t]['train_r2'] for t in target_cols],
            'Val R²': [metrics[t]['val_r2'] for t in target_cols]
        })
        
        metrics_df.to_csv(output_dir / "dav_regression_metrics.csv", index=False)
        
        # Print summary
        print("\nModel training complete. Summary:")
        print(metrics_df)
    
    return {
        'models': models,
        'scaler': scaler,
        'metrics': metrics,
        'val_predictions': val_predictions
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a baseline DAV regression model')
    parser.add_argument('--train_data', type=str, default='data/synthetic/small_train.csv',
                        help='Path to the training data CSV file')
    parser.add_argument('--val_data', type=str, default='data/synthetic/small_val.csv',
                        help='Path to the validation data CSV file')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'svr', 'ridge'],
                        help='Type of regression model to use')
    parser.add_argument('--output_dir', type=str, default='models/baseline/dav_regression',
                        help='Directory to save model and results')
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_csv(args.train_data)
    val_data = pd.read_csv(args.val_data)
    
    # Train model
    train_dav_regression_model(
        train_data, 
        val_data, 
        model_type=args.model_type, 
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
