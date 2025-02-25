"""
Module for mapping dominance, arousal, and valence (DAV) values to emotion categories.

This module implements methods to convert DAV values to discrete emotion categories
using various mapping techniques.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
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

class EmotionMapper:
    """
    A class for mapping dominance, arousal, and valence (DAV) values to emotion categories.
    
    This class implements methods to convert DAV values to discrete emotion categories
    using various mapping techniques.
    """
    
    def __init__(self, mapping_method='rule_based', model_params=None):
        """
        Initialize the emotion mapper.
        
        Parameters:
        -----------
        mapping_method : str, optional
            Method to use for mapping DAV values to emotions
            ('rule_based', 'knn', 'random_forest', or 'svm')
        model_params : dict, optional
            Parameters for the mapping model (if using a machine learning method)
        """
        self.mapping_method = mapping_method
        self.model_params = model_params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.emotion_categories = None
        
        # Initialize model if using a machine learning method
        if mapping_method != 'rule_based':
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the mapping model if using a machine learning method."""
        if self.mapping_method == 'knn':
            params = {
                'n_neighbors': self.model_params.get('n_neighbors', 5),
                'weights': self.model_params.get('weights', 'distance'),
                'metric': self.model_params.get('metric', 'euclidean')
            }
            self.model = KNeighborsClassifier(**params)
        
        elif self.mapping_method == 'random_forest':
            params = {
                'n_estimators': self.model_params.get('n_estimators', 100),
                'max_depth': self.model_params.get('max_depth', 10),
                'random_state': self.model_params.get('random_state', 42)
            }
            self.model = RandomForestClassifier(**params)
        
        elif self.mapping_method == 'svm':
            params = {
                'kernel': self.model_params.get('kernel', 'rbf'),
                'C': self.model_params.get('C', 1.0),
                'gamma': self.model_params.get('gamma', 'scale'),
                'probability': True,
                'random_state': self.model_params.get('random_state', 42)
            }
            self.model = SVC(**params)
    
    def _rule_based_mapping(self, dav_values):
        """
        Map DAV values to emotions using rule-based approach.
        
        Parameters:
        -----------
        dav_values : numpy.ndarray
            Array of shape (n_samples, 3) containing DAV values
            (dominance, arousal, valence)
        
        Returns:
        --------
        list
            List of emotion categories
        """
        emotions = []
        
        for i in range(dav_values.shape[0]):
            dominance, arousal, valence = dav_values[i]
            
            # Define emotion regions in DAV space
            # These are approximate regions based on literature
            if valence > 0.3:
                if arousal > 0.3:
                    if dominance > 0.3:
                        emotion = 'happy'
                    else:
                        emotion = 'surprised'
                else:
                    if dominance > 0.3:
                        emotion = 'calm'
                    else:
                        emotion = 'relaxed'
            else:
                if arousal > 0.3:
                    if dominance > 0.3:
                        emotion = 'angry'
                    else:
                        emotion = 'fearful'
                else:
                    if dominance > 0.3:
                        emotion = 'disgusted'
                    else:
                        emotion = 'sad'
            
            # Special case for neutral
            if abs(valence) < 0.2 and abs(arousal) < 0.2 and abs(dominance) < 0.2:
                emotion = 'neutral'
            
            emotions.append(emotion)
        
        return emotions
    
    def fit(self, dav_values, emotion_labels):
        """
        Fit the emotion mapper to the training data.
        
        Parameters:
        -----------
        dav_values : numpy.ndarray or pandas.DataFrame
            Array of shape (n_samples, 3) containing DAV values
            (dominance, arousal, valence)
        emotion_labels : numpy.ndarray or pandas.Series
            Array of shape (n_samples,) containing emotion labels
        
        Returns:
        --------
        self : EmotionMapper
            The fitted mapper
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(dav_values, pd.DataFrame):
            dav_values = dav_values[['dominance', 'arousal', 'valence']].values
        
        if isinstance(emotion_labels, pd.Series):
            emotion_labels = emotion_labels.values
        
        # Store emotion categories
        self.emotion_categories = np.unique(emotion_labels)
        
        # If using rule-based mapping, no fitting needed
        if self.mapping_method == 'rule_based':
            logger.info("Using rule-based mapping, no fitting needed")
            return self
        
        # Scale DAV values
        dav_values_scaled = self.scaler.fit_transform(dav_values)
        
        # Fit model
        logger.info(f"Fitting {self.mapping_method} model for emotion mapping...")
        self.model.fit(dav_values_scaled, emotion_labels)
        
        return self
    
    def predict(self, dav_values):
        """
        Predict emotion categories from DAV values.
        
        Parameters:
        -----------
        dav_values : numpy.ndarray or pandas.DataFrame
            Array of shape (n_samples, 3) containing DAV values
            (dominance, arousal, valence)
        
        Returns:
        --------
        numpy.ndarray
            Array of shape (n_samples,) containing predicted emotion categories
        """
        # Convert input to numpy array if needed
        if isinstance(dav_values, pd.DataFrame):
            dav_values = dav_values[['dominance', 'arousal', 'valence']].values
        
        # If using rule-based mapping
        if self.mapping_method == 'rule_based':
            return np.array(self._rule_based_mapping(dav_values))
        
        # Scale DAV values
        dav_values_scaled = self.scaler.transform(dav_values)
        
        # Predict emotions
        return self.model.predict(dav_values_scaled)
    
    def evaluate(self, dav_values, emotion_labels):
        """
        Evaluate the emotion mapper on test data.
        
        Parameters:
        -----------
        dav_values : numpy.ndarray or pandas.DataFrame
            Array of shape (n_samples, 3) containing DAV values
            (dominance, arousal, valence)
        emotion_labels : numpy.ndarray or pandas.Series
            Array of shape (n_samples,) containing emotion labels
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        # Convert inputs to numpy arrays if needed
        if isinstance(dav_values, pd.DataFrame):
            dav_values = dav_values[['dominance', 'arousal', 'valence']].values
        
        if isinstance(emotion_labels, pd.Series):
            emotion_labels = emotion_labels.values
        
        # Predict emotions
        predictions = self.predict(dav_values)
        
        # Calculate metrics
        accuracy = accuracy_score(emotion_labels, predictions)
        report = classification_report(emotion_labels, predictions, output_dict=True)
        cm = confusion_matrix(emotion_labels, predictions)
        
        logger.info(f"Evaluation accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'report': report,
            'confusion_matrix': cm
        }
    
    def save(self, output_dir):
        """
        Save the emotion mapper to disk.
        
        Parameters:
        -----------
        output_dir : str or Path
            Directory to save the model
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save model if not rule-based
        if self.mapping_method != 'rule_based':
            joblib.dump(self.model, output_dir / f"emotion_mapper_{self.mapping_method}.pkl")
            joblib.dump(self.scaler, output_dir / "dav_scaler.pkl")
        
        # Save emotion categories
        if self.emotion_categories is not None:
            with open(output_dir / "emotion_categories.txt", "w") as f:
                for category in self.emotion_categories:
                    f.write(f"{category}\n")
        
        # Save model info
        model_info = {
            'mapping_method': self.mapping_method,
            'model_params': self.model_params
        }
        
        joblib.dump(model_info, output_dir / "model_info.pkl")
        
        logger.info(f"Emotion mapper saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir):
        """
        Load an emotion mapper from disk.
        
        Parameters:
        -----------
        model_dir : str or Path
            Directory containing the saved model
        
        Returns:
        --------
        EmotionMapper
            The loaded mapper
        """
        model_dir = Path(model_dir)
        
        # Load model info
        model_info = joblib.load(model_dir / "model_info.pkl")
        
        # Create instance
        instance = cls(
            mapping_method=model_info['mapping_method'],
            model_params=model_info['model_params']
        )
        
        # Load model if not rule-based
        if instance.mapping_method != 'rule_based':
            instance.model = joblib.load(model_dir / f"emotion_mapper_{instance.mapping_method}.pkl")
            instance.scaler = joblib.load(model_dir / "dav_scaler.pkl")
        
        # Load emotion categories
        categories_file = model_dir / "emotion_categories.txt"
        if categories_file.exists():
            with open(categories_file, "r") as f:
                instance.emotion_categories = np.array([line.strip() for line in f.readlines()])
        
        logger.info(f"Emotion mapper loaded from {model_dir}")
        
        return instance

def train_emotion_mapper(train_data, val_data=None, mapping_method='random_forest', model_params=None, output_dir=None):
    """
    Train an emotion mapper on the given data.
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Training data with DAV values and emotion labels
    val_data : pandas.DataFrame, optional
        Validation data with DAV values and emotion labels
    mapping_method : str, optional
        Method to use for mapping DAV values to emotions
        ('rule_based', 'knn', 'random_forest', or 'svm')
    model_params : dict, optional
        Parameters for the mapping model (if using a machine learning method)
    output_dir : str or Path, optional
        Directory to save model and results
    
    Returns:
    --------
    EmotionMapper
        Trained emotion mapper
    """
    # Create output directory if provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    # Create and train mapper
    mapper = EmotionMapper(mapping_method=mapping_method, model_params=model_params)
    
    # Extract DAV values and emotion labels
    dav_cols = ['dominance', 'arousal', 'valence']
    emotion_col = 'emotion'
    
    # Fit mapper
    logger.info("Fitting emotion mapper...")
    mapper.fit(train_data[dav_cols], train_data[emotion_col])
    
    # Evaluate on validation data if provided
    if val_data is not None:
        logger.info("Evaluating on validation data...")
        metrics = mapper.evaluate(val_data[dav_cols], val_data[emotion_col])
    
    # Save model if output directory provided
    if output_dir is not None:
        mapper.save(output_dir)
    
    return mapper

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train an emotion mapper')
    parser.add_argument('--train_data', type=str, default='data/synthetic/small_train.csv',
                        help='Path to the training data CSV file')
    parser.add_argument('--val_data', type=str, default='data/synthetic/small_val.csv',
                        help='Path to the validation data CSV file')
    parser.add_argument('--mapping_method', type=str, default='random_forest',
                        choices=['rule_based', 'knn', 'random_forest', 'svm'],
                        help='Method to use for mapping DAV values to emotions')
    parser.add_argument('--output_dir', type=str, default='models/emotion_mapper',
                        help='Directory to save model and results')
    
    args = parser.parse_args()
    
    # Load data
    train_data = pd.read_csv(args.train_data)
    val_data = pd.read_csv(args.val_data)
    
    # Train model
    train_emotion_mapper(
        train_data, 
        val_data, 
        mapping_method=args.mapping_method, 
        output_dir=args.output_dir
    )

if __name__ == '__main__':
    main()
