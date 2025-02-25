"""
Script to process audio files and predict emotions.

This script extracts features from audio files, predicts DAV values,
and then maps these values to emotion categories.
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
from src.models.emotion_mapper import EmotionMapper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def process_audio_file(audio_file, dav_model_dir, emotion_model_dir, output_file=None):
    """
    Process an audio file and predict emotion.
    
    Parameters:
    -----------
    audio_file : str or Path
        Path to the audio file
    dav_model_dir : str or Path
        Directory containing the trained DAV regression model
    emotion_model_dir : str or Path
        Directory containing the trained emotion mapper
    output_file : str or Path, optional
        Path to save the results as CSV
    
    Returns:
    --------
    dict
        Dictionary containing extracted features, DAV values, and predicted emotion
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
    logger.info(f"Loading DAV regression model from {dav_model_dir}")
    dav_regressor = DAVRegressor.load(dav_model_dir)
    
    # Predict DAV values
    logger.info("Predicting DAV values")
    dav_values = dav_regressor.predict(features_df)
    
    # Create DAV DataFrame
    dav_df = pd.DataFrame({
        'file': str(audio_file),
        'dominance': [dav_values[0, 0]],
        'arousal': [dav_values[0, 1]],
        'valence': [dav_values[0, 2]]
    })
    
    # Load emotion mapper
    logger.info(f"Loading emotion mapper from {emotion_model_dir}")
    emotion_mapper = EmotionMapper.load(emotion_model_dir)
    
    # Predict emotion
    logger.info("Predicting emotion")
    emotion = emotion_mapper.predict(dav_values)[0]
    
    # Create result dictionary
    result = {
        'file': str(audio_file),
        'dominance': dav_values[0, 0],
        'arousal': dav_values[0, 1],
        'valence': dav_values[0, 2],
        'emotion': emotion
    }
    
    # Save to CSV if output file provided
    if output_file is not None:
        pd.DataFrame([result]).to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
    
    return result

def batch_process_audio_files(audio_dir, dav_model_dir, emotion_model_dir, output_file=None):
    """
    Process multiple audio files and predict emotions.
    
    Parameters:
    -----------
    audio_dir : str or Path
        Directory containing audio files
    dav_model_dir : str or Path
        Directory containing the trained DAV regression model
    emotion_model_dir : str or Path
        Directory containing the trained emotion mapper
    output_file : str or Path, optional
        Path to save the results as CSV
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing results for all files
    """
    audio_dir = Path(audio_dir)
    
    # Get audio files
    audio_files = list(audio_dir.glob('*.wav'))
    
    if not audio_files:
        logger.error(f"No audio files found in {audio_dir}")
        return None
    
    logger.info(f"Found {len(audio_files)} audio files")
    
    # Process each file
    results = []
    
    for audio_file in audio_files:
        result = process_audio_file(audio_file, dav_model_dir, emotion_model_dir)
        
        if result is not None:
            results.append(result)
    
    if not results:
        logger.error("Failed to process any audio file")
        return None
    
    # Create DataFrame
    result_df = pd.DataFrame(results)
    
    # Save to CSV if output file provided
    if output_file is not None:
        result_df.to_csv(output_file, index=False)
        logger.info(f"Saved results to {output_file}")
    
    return result_df

def main():
    parser = argparse.ArgumentParser(description='Process audio files and predict emotions')
    parser.add_argument('--audio_file', type=str, help='Path to a single audio file')
    parser.add_argument('--audio_dir', type=str, help='Directory containing audio files')
    parser.add_argument('--dav_model_dir', type=str, required=True,
                        help='Directory containing the trained DAV regression model')
    parser.add_argument('--emotion_model_dir', type=str, required=True,
                        help='Directory containing the trained emotion mapper')
    parser.add_argument('--output_file', type=str,
                        help='Path to save the results as CSV')
    
    args = parser.parse_args()
    
    if args.audio_file:
        # Process a single audio file
        process_audio_file(args.audio_file, args.dav_model_dir, args.emotion_model_dir, args.output_file)
    elif args.audio_dir:
        # Process multiple audio files
        batch_process_audio_files(args.audio_dir, args.dav_model_dir, args.emotion_model_dir, args.output_file)
    else:
        logger.error("Either --audio_file or --audio_dir must be provided")
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
