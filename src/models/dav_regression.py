"""
Model for predicting dominance, arousal, and valence (DAV) values from audio features.

This module implements regression models to convert audio features to
DAV values, which can then be mapped to emotion categories.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DAVRegressor:
    """
    A class for predicting dominance, arousal, and valence (DAV) values from audio features.
    
    This class implements regression models to convert audio features to DAV values,
    which can then be mapped to emotion categories.
    """
    
    def __init__(self, model_type='random_forest', model_params=None):
        """
        Initialize the DAV regressor.
        
        Parameters:
        -----------
        model_type : str, optional
            Type of regression model to use ('random_forest', 'svr', or 'ridge')
        model_params : dict, optional
            Parameters for the regression model
        """
        self.model_type = model_type
        self.model_params = model_params or {}
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.target_cols = ['dominance', 'arousal', 'valence']
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize regression models for each DAV dimension."""
        for target in self.target_cols:
            if self.model_type == 'random_forest':
                params = {
                    'n_estimators': self.model_params.get('n_estimators', 100),
                    'max_depth': self.model_params.get('max_depth', 10),
                    'random_state': self.model_params.get('random_state', 42)
                }
                self.models[target] = RandomForestRegressor(**params)
            
            elif self.model_type == 'svr':
                params = {
                    'kernel': self.model_params.get('kernel', 'rbf'),
                    'C': self.model_params.get('C', 1.0),
                    'epsilon': self.model_params.get('epsilon', 0.1),
                    'gamma': self.model_params.get('gamma', 'scale')
                }
                self.models[target] = SVR(**params)
            
            elif self.model_type == 'ridge':
                params = {
                    'alpha': self.model_params.get('alpha', 1.0),
                    'random_state': self.model_params.get('random_state', 42)
                }
                self.models[target] = Ridge(**params)
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y=None, feature_cols=None, **kwargs):
        """
        Fit the DAV regressor to the training data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Training data features
        y : pandas.DataFrame or numpy.ndarray, optional
            Training data targets (DAV values)
            If None, assumes X contains both features and targets
        feature_cols : list, optional
            List of feature column names if X is a DataFrame
        **kwargs : dict
            Additional keyword arguments for the model's fit method
        
        Returns:
        --------
        self : DAVRegressor
            The fitted regressor
        """
        # Handle input data
        if y is None and isinstance(X, pd.DataFrame):
            # Assume X contains both features and targets
            if feature_cols is None:
                # Try to infer feature columns
                self.feature_cols = [col for col in X.columns if col.startswith('feature_')]
                if not self.feature_cols:
                    raise ValueError("No feature columns found in X. Please specify feature_cols.")
            else:
                self.feature_cols = feature_cols
            
            # Extract features and targets
            features = X[self.feature_cols].values
            targets = X[self.target_cols].values
        
        elif y is not None:
            # Features and targets provided separately
            if isinstance(X, pd.DataFrame):
                if feature_cols is None:
                    # Try to infer feature columns
                    self.feature_cols = [col for col in X.columns if col.startswith('feature_')]
                    if not self.feature_cols:
                        raise ValueError("No feature columns found in X. Please specify feature_cols.")
                else:
                    self.feature_cols = feature_cols
                
                features = X[self.feature_cols].values
            else:
                features = X
            
            if isinstance(y, pd.DataFrame):
                targets = y[self.target_cols].values
            else:
                targets = y
        
        else:
            raise ValueError("Invalid input data. Please provide features and targets.")
        
        # Fit scaler
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit models for each target
        for i, target in enumerate(self.target_cols):
            logger.info(f"Training {self.model_type} regressor for {target}...")
            self.models[target].fit(features_scaled, targets[:, i], **kwargs)
        
        return self
    
    def predict(self, X, feature_cols=None):
        """
        Predict DAV values for new data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Data to predict
        feature_cols : list, optional
            List of feature column names if X is a DataFrame and different from training
        
        Returns:
        --------
        numpy.ndarray
            Predicted DAV values
        """
        # Handle input data
        if isinstance(X, pd.DataFrame):
            if feature_cols is None:
                if self.feature_cols is None:
                    raise ValueError("No feature columns specified. Please fit the model first or provide feature_cols.")
                feature_cols = self.feature_cols
            
            features = X[feature_cols].values
        else:
            features = X
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions for each target
        predictions = np.zeros((features.shape[0], len(self.target_cols)))
        
        for i, target in enumerate(self.target_cols):
            predictions[:, i] = self.models[target].predict(features_scaled)
        
        return predictions
    
    def evaluate(self, X, y=None, feature_cols=None):
        """
        Evaluate the DAV regressor on test data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Test data features
        y : pandas.DataFrame or numpy.ndarray, optional
            Test data targets (DAV values)
            If None, assumes X contains both features and targets
        feature_cols : list, optional
            List of feature column names if X is a DataFrame
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Handle input data
        if y is None and isinstance(X, pd.DataFrame):
            # Assume X contains both features and targets
            if feature_cols is None:
                if self.feature_cols is None:
                    raise ValueError("No feature columns specified. Please fit the model first or provide feature_cols.")
                feature_cols = self.feature_cols
            
            # Extract features and targets
            features = X[feature_cols].values
            targets = X[self.target_cols].values
        
        elif y is not None:
            # Features and targets provided separately
            if isinstance(X, pd.DataFrame):
                if feature_cols is None:
                    if self.feature_cols is None:
                        raise ValueError("No feature columns specified. Please fit the model first or provide feature_cols.")
                    feature_cols = self.feature_cols
                
                features = X[feature_cols].values
            else:
                features = X
            
            if isinstance(y, pd.DataFrame):
                targets = y[self.target_cols].values
            else:
                targets = y
        
        else:
            raise ValueError("Invalid input data. Please provide features and targets.")
        
        # Make predictions
        predictions = self.predict(features)
        
        # Calculate metrics
        metrics = {}
        
        for i, target in enumerate(self.target_cols):
            mse = mean_squared_error(targets[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            
            metrics[target] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"Evaluation metrics for {target}:")
            logger.info(f"  MSE: {mse:.4f}")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  MAE: {mae:.4f}")
            logger.info(f"  R²: {r2:.4f}")
        
        # Calculate overall metrics
        overall_mse = mean_squared_error(targets.flatten(), predictions.flatten())
        overall_rmse = np.sqrt(overall_mse)
        overall_mae = mean_absolute_error(targets.flatten(), predictions.flatten())
        overall_r2 = r2_score(targets.flatten(), predictions.flatten())
        
        metrics['overall'] = {
            'mse': overall_mse,
            'rmse': overall_rmse,
            'mae': overall_mae,
            'r2': overall_r2
        }
        
        logger.info(f"Overall evaluation metrics:")
        logger.info(f"  MSE: {overall_mse:.4f}")
        logger.info(f"  RMSE: {overall_rmse:.4f}")
        logger.info(f"  MAE: {overall_mae:.4f}")
        logger.info(f"  R²: {overall_r2:.4f}")
        
        return metrics
    
    def save(self, output_dir):
        """
        Save the DAV regressor to disk.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save the model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save models
        for target in self.target_cols:
            joblib.dump(self.models[target], output_dir / f"{target}_{self.model_type}_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, output_dir / "feature_scaler.pkl")
        
        # Save feature columns
        if self.feature_cols is not None:
            with open(output_dir / "feature_cols.txt", "w") as f:
                for col in self.feature_cols:
                    f.write(f"{col}\n")
        
        # Save model info
        model_info = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'target_cols': self.target_cols
        }
        
        joblib.dump(model_info, output_dir / "model_info.pkl")
        
        logger.info(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir):
        """
        Load a DAV regressor from disk.
        
        Parameters:
        -----------
        model_dir : str or Path
            Directory containing the saved model
        
        Returns:
        --------
        DAVRegressor
            The loaded regressor
        """
        model_dir = Path(model_dir)
        
        # Load model info
        model_info = joblib.load(model_dir / "model_info.pkl")
        
        # Create instance
        instance = cls(
            model_type=model_info['model_type'],
            model_params=model_info['model_params']
        )
        
        # Load models
        for target in model_info['target_cols']:
            instance.models[target] = joblib.load(model_dir / f"{target}_{instance.model_type}_model.pkl")
        
        # Load scaler
        instance.scaler = joblib.load(model_dir / "feature_scaler.pkl")
        
        # Load feature columns
        feature_cols_file = model_dir / "feature_cols.txt"
        if feature_cols_file.exists():
            with open(feature_cols_file, "r") as f:
                instance.feature_cols = [line.strip() for line in f.readlines()]
        
        logger.info(f"Model loaded from {model_dir}")
        
        return instance
    
    def plot_feature_importances(self, output_dir=None, top_n=20):
        """
        Plot feature importances for random forest models.
        
        Parameters:
        -----------
        output_dir : str or Path, optional
            Directory to save the plots
        top_n : int, optional
            Number of top features to plot
        
        Returns:
        --------
        dict
            Dictionary containing feature importances for each target
        """
        if self.model_type != 'random_forest':
            logger.warning(f"Feature importances are only available for random forest models, not {self.model_type}")
            return None
        
        if self.feature_cols is None:
            logger.warning("No feature columns specified. Please fit the model first.")
            return None
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
        
        importances = {}
        
        for target in self.target_cols:
            model = self.models[target]
            
            # Get feature importances
            feature_importances = model.feature_importances_
            importances[target] = dict(zip(self.feature_cols, feature_importances))
            
            # Sort features by importance
            sorted_idx = np.argsort(feature_importances)[::-1]
            
            # Plot top N features
            plt.figure(figsize=(12, 6))
            plt.bar(range(min(top_n, len(sorted_idx))), feature_importances[sorted_idx[:top_n]])
            plt.xticks(range(min(top_n, len(sorted_idx))), [self.feature_cols[i] for i in sorted_idx[:top_n]], rotation=90)
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.title(f'Top {top_n} Feature Importances for {target}')
            plt.tight_layout()
            
            if output_dir is not None:
                plt.savefig(output_dir / f"{target}_feature_importances.png")
                plt.close()
            else:
                plt.show()
        
        return importances
    
    def plot_predictions(self, X, y=None, feature_cols=None, output_dir=None):
        """
        Plot actual vs predicted values.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            Test data features
        y : pandas.DataFrame or numpy.ndarray, optional
            Test data targets (DAV values)
            If None, assumes X contains both features and targets
        feature_cols : list, optional
            List of feature column names if X is a DataFrame
        output_dir : str or Path, optional
            Directory to save the plots
        
        Returns:
        --------
        tuple
            Tuple containing actual and predicted values
        """
        # Handle input data
        if y is None and isinstance(X, pd.DataFrame):
            # Assume X contains both features and targets
            if feature_cols is None:
                if self.feature_cols is None:
                    raise ValueError("No feature columns specified. Please fit the model first or provide feature_cols.")
                feature_cols = self.feature_cols
            
            # Extract features and targets
            features = X[feature_cols].values
            targets = X[self.target_cols].values
        
        elif y is not None:
            # Features and targets provided separately
            if isinstance(X, pd.DataFrame):
                if feature_cols is None:
                    if self.feature_cols is None:
                        raise ValueError("No feature columns specified. Please fit the model first or provide feature_cols.")
                    feature_cols = self.feature_cols
                
                features = X[feature_cols].values
            else:
                features = X
            
            if isinstance(y, pd.DataFrame):
                targets = y[self.target_cols].values
            else:
                targets = y
        
        else:
            raise ValueError("Invalid input data. Please provide features and targets.")
        
        # Make predictions
        predictions = self.predict(features)
        
        # Calculate metrics
        metrics = {}
        
        for i, target in enumerate(self.target_cols):
            mse = mean_squared_error(targets[:, i], predictions[:, i])
            r2 = r2_score(targets[:, i], predictions[:, i])
            
            metrics[target] = {
                'mse': mse,
                'r2': r2
            }
        
        # Plot actual vs predicted values
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, target in enumerate(self.target_cols):
            axes[i].scatter(targets[:, i], predictions[:, i], alpha=0.5)
            axes[i].plot([-1, 1], [-1, 1], 'r--')
            axes[i].set_xlabel(f'Actual {target}')
            axes[i].set_ylabel(f'Predicted {target}')
            axes[i].set_title(f'{target.capitalize()} Prediction\nMSE: {metrics[target]["mse"]:.4f}, R²: {metrics[target]["r2"]:.4f}')
            axes[i].grid(True)
            axes[i].set_xlim([-1.1, 1.1])
            axes[i].set_ylim([-1.1, 1.1])
        
        plt.tight_layout()
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True, parents=True)
            plt.savefig(output_dir / "dav_prediction_results.png")
            plt.close()
        else:
            plt.show()
        
        return targets, predictions

def train_dav_regressor(train_data, val_data=None, model_type='random_forest', model_params=None, output_dir=None, tune_params=False):
    """
    Train a DAV regressor on the given data.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with features and DAV values
    val_data : pandas.DataFrame, optional
        Validation data with features and DAV values
    model_type : str, optional
        Type of regression model to use ('random_forest', 'svr', or 'ridge')
    model_params : dict, optional
        Parameters for the regression model
    output_dir : str or Path, optional
        Directory to save model and results
    tune_params : bool, optional
        Whether to tune hyperparameters using grid search
    
    Returns:
    --------
    DAVRegressor
        Trained DAV regressor
    """
    # Create output directory if provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Extract feature columns
    feature_cols = [col for col in train_data.columns if col.startswith('feature_')]
    
    # Create and train regressor
    regressor = DAVRegressor(model_type=model_type, model_params=model_params)
    
    # Fit regressor
    logger.info("Fitting regressor...")
    regressor.fit(train_data, feature_cols=feature_cols)
    
    # Evaluate on validation data if provided
    if val_data is not None:
        logger.info("Evaluating on validation data...")
        metrics = regressor.evaluate(val_data, feature_cols=feature_cols)
        
        # Plot predictions
        if output_dir is not None:
            regressor.plot_predictions(val_data, feature_cols=feature_cols, output_dir=output_dir)
    
    # Plot feature importances for random forest
    if model_type == 'random_forest' and output_dir is not None:
        regressor.plot_feature_importances(output_dir=output_dir)
    
    # Save model if output directory provided
    if output_dir is not None:
        regressor.save(output_dir)
    
    return regressor

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a DAV regression model')
    parser.add_argument('--train_data', type=str, default='data/synthetic/small_train.csv',
                        help='Path to the training data CSV file')
    parser.add_argument('--val_data', type=str, default='data/synthetic/small_val.csv',
                        help='Path to the validation data CSV file')
    parser.add_argument('--model_type', type=str, default='random_forest',
                        choices=['random_forest', 'svr', 'ridge'],
                        help='Type of regression model to use')
    parser.add_argument('--output_dir', type=str, default='models/dav_regression',
                        help='Directory to save model and results')
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_csv(args.train_data)
    val_data = pd.read_csv(args.val_data)
    
    # Train model
    train_dav_regressor(
        train_data, 
        val_data, 
        model_type=args.model_type, 
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
