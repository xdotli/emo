"""
Script to extract audio features and predict DAV values.

This script extracts features from audio files and uses a trained
DAV regression model to predict dominance, arousal, and valence values.
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import sys
import logging

# Add parent directory to path to import from src
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.features.audio_features import extract_features_from_file
from src.models.dav_regression import DAVRegressor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_dav_from_audio(audio_file, model_dir, output_file=None):
    """
    Extract DAV values from an audio file using a trained model.
    
    Parameters:
    -----------
    audio_file : str or Path
        Path to the audio file
    model_dir : str or Path
        Directory containing the trained DAV regression model
    output_file : str or Path, optional
        Path to save the extracted DAV values as CSV
    
    Returns:
    --------
    dict
        Dictionary containing extracted DAV values
    """
    # Extract audio features
    logger.info(f"Extracting features from {audio_file}")
    features = extract_features_from_file(audio_file)
    
    if features is None:
        logger.error(f"Failed to extract features from {audio_file}")
        return None
    
    # Create DataFrame from features
    features_df = pd.DataFrame([features])
    
    # Load DAV regression model
    logger.info(f"Loading DAV regression model from {model_dir}")
    regressor = DAVRegressor.load(model_dir)
    
    # Predict DAV values
    logger.info("Predicting DAV values")
    dav_values = regressor.predict(features_df)
    
    # Create result dictionary
    result = {
        'file': str(audio_file),
        'dominance': dav_values[0, 0],
        'arousal': dav_values[0, 1],
        'valence': dav_values[0, 2]
    }
    
    # Save to CSV if output file provided
    if output_file is not None:
        pd.DataFrame([result]).to_csv(output_file, index=False)
        logger.info(f"Saved DAV values to {output_file}")
    
    return result

def batch_extract_dav(audio_dir, model_dir, output_file=None):
    """
    Extract DAV values from multiple audio files.
    
    Parameters:
    -----------
    audio_dir : str or Path
        Directory containing audio files
    model_dir : str or Path
        Directory containing the trained DAV regression model
    output_file : str or Path, optional
        Path to save the extracted DAV values as CSV
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing extracted DAV values for all files
    """
    audio_dir = Path(audio_dir)
    
    # Get audio files
    audio_files = list(audio_dir.glob('*.wav'))
    
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return None
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Extract features from each file
    all_features = []
    
    for audio_file in audio_files:
        features = extract_features_from_file(audio_file)
        
        if features is not None:
            features['file'] = str(audio_file)
            all_features.append(features)
    
    if not all_features:
        logger.error("Failed to extract features from any audio file")
        return None
    
    # Create DataFrame from features
    features_df = pd.DataFrame(all_features)
    
    # Load DAV regression model
    logger.info(f"Loading DAV regression model from {model_dir}")
    regressor = DAVRegressor.load(model_dir)
    
    # Predict DAV values
    logger.info("Predicting DAV values")
    feature_cols = [col for col in features_df.columns if col not in ['file']]
    dav_values = regressor.predict(features_df[feature_cols])
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'file': features_df['file'],
        'dominance': dav_values[:, 0],
        'arousal': dav_values[:, 1],
        'valence': dav_values[:, 2]
    })
    
    # Save to CSV if output file provided
    if output_file is not None:
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved DAV values to {output_file}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Extract DAV values from audio files')
    parser.add_argument('--audio_file', type=str, help='Path to a single audio file')
    parser.add_argument('--audio_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Directory containing the trained DAV regression model')
    parser.add_argument('--output_file', type=str, help='Path to save the extracted DAV values as CSV')
    
    args = parser.parse_args()
    
    if args.audio_file:
        # Extract DAV values from a single audio file
        extract_dav_from_audio(args.audio_file, args.model_dir, args.output_file)
    elif args.audio_dir:
        # Extract DAV values from multiple audio files
        batch_extract_dav(args.audio_dir, args.model_dir, args.output_file)
    else:
        logger.error("Either --audio_file or --audio_dir must be provided")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
