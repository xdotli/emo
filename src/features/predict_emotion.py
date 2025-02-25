"""
Script to predict emotions from DAV values.

This script uses a trained emotion mapper to predict emotion categories
from dominance, arousal, and valence (DAV) values.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
import logging

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.emotion_mapper import EmotionMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def predict_emotion_from_dav(dav_values, model_dir, output_file=None):
    """
    Predict emotion categories from DAV values using a trained model.
    
    Parameters:
    -----------
    dav_values : pandas.DataFrame
        DataFrame containing DAV values (dominance, arousal, valence)
    model_dir : str or Path
        Directory containing the trained emotion mapper
    output_file : str or Path, optional
        Path to save the predicted emotions as CSV
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing DAV values and predicted emotions
    """
    # Load emotion mapper
    logger.info(f"Loading emotion mapper from {model_dir}")
    mapper = EmotionMapper.load(model_dir)
    
    # Predict emotions
    logger.info("Predicting emotions from DAV values")
    emotions = mapper.predict(dav_values[['dominance', 'arousal', 'valence']])
    
    # Add emotions to DataFrame
    result_df = dav_values.copy()
    result_df['emotion'] = emotions
    
    # Save to CSV if output file provided
    if output_file is not None:
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Predict emotions from DAV values')
    parser.add_argument('--dav_file', type=str, required=True,
                        help='Path to CSV file containing DAV values')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained emotion mapper')
    parser.add_argument('--output_file', type=str,
                        help='Path to save the predicted emotions as CSV')
    
    args = parser.parse_args()
    
    # Load DAV values
    dav_values = pd.read_csv(args.dav_file)
    
    # Predict emotions
    predict_emotion_from_dav(dav_values, args.model_dir, args.output_file)

if __name__ == '__main__':
    main()
