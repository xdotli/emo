"""
Script to download and prepare the CREMA-D dataset.

The CREMA-D dataset is publicly available and can be downloaded automatically.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import requests
import glob
import tqdm
import csv

def download_file(url, filename):
    """Download a file from a URL."""
    print(f"Downloading {url} to {filename}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    with open(filename, 'wb') as file:
        for data in response.iter_content(block_size):
            file.write(data)
    
    print(f"Download complete: {filename}")

def download_crema_d(data_dir):
    """Download the CREMA-D dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    crema_dir = data_dir / "CREMA-D"
    crema_dir.mkdir(exist_ok=True)
    
    # URLs for the dataset files
    audio_url = "https://media.githubusercontent.com/media/CheyneyComputerScience/CREMA-D/master/AudioWAV.zip"
    summary_url = "https://raw.githubusercontent.com/CheyneyComputerScience/CREMA-D/master/VideoDemographics.csv"
    
    # Download the audio dataset
    audio_zip = crema_dir / "AudioWAV.zip"
    if not audio_zip.exists():
        print(f"Downloading CREMA-D audio dataset...")
        download_file(audio_url, audio_zip)
    else:
        print(f"CREMA-D audio dataset already downloaded at {audio_zip}")
    
    # Download the summary file
    summary_file = crema_dir / "VideoDemographics.csv"
    if not summary_file.exists():
        print(f"Downloading CREMA-D demographics data...")
        download_file(summary_url, summary_file)
    else:
        print(f"CREMA-D demographics data already downloaded at {summary_file}")
    
    # Extract the audio dataset
    audio_dir = crema_dir / "AudioWAV"
    if not audio_dir.exists():
        print(f"Extracting CREMA-D audio dataset...")
        with zipfile.ZipFile(audio_zip, 'r') as zip_ref:
            zip_ref.extractall(crema_dir)
    else:
        print(f"CREMA-D audio dataset already extracted at {audio_dir}")
    
    return crema_dir

def process_crema_d(data_dir, output_dir):
    """Process the CREMA-D dataset and create CSV files with metadata."""
    crema_dir = Path(data_dir) / "CREMA-D"
    audio_dir = crema_dir / "AudioWAV"
    summary_file = crema_dir / "VideoDemographics.csv"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define emotion mapping
    emotion_map = {
        'A': 'angry',
        'D': 'disgust',
        'F': 'fear',
        'H': 'happy',
        'N': 'neutral',
        'S': 'sad'
    }
    
    # Define level mapping
    level_map = {
        'LO': 'low',
        'MD': 'medium',
        'HI': 'high',
        'XX': 'unspecified'
    }
    
    # Load demographics data
    demographics = {}
    with open(summary_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            demographics[row['ActorID']] = {
                'Sex': row['Sex'],
                'Race': row['Race'],
                'Age': row['Age']
            }
    
    # Initialize lists to store data
    data = []
    
    # Process each audio file
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"Processing {len(audio_files)} audio files...")
    
    for wav_file in audio_files:
        # Parse filename
        # Format: ActorID_SentenceID_EmotionType_EmotionalIntensity.wav
        # Example: 1012_IEO_ANG_XX.wav
        filename = wav_file.stem
        parts = filename.split('_')
        
        if len(parts) == 4:
            actor_id = parts[0]
            sentence_id = parts[1]
            emotion_code = parts[2]
            level_code = parts[3]
            
            # Get emotion and level
            emotion = emotion_map.get(emotion_code[0], 'unknown')
            level = level_map.get(level_code, 'unknown')
            
            # Get demographics
            sex = demographics.get(actor_id, {}).get('Sex', 'unknown')
            race = demographics.get(actor_id, {}).get('Race', 'unknown')
            age = demographics.get(actor_id, {}).get('Age', 'unknown')
            
            # Map categorical emotions to approximate DAV values
            # These are approximate values based on literature
            dav_map = {
                'angry': (-0.5, 0.7, 0.7),
                'disgust': (-0.6, 0.2, 0.3),
                'fear': (-0.7, 0.6, -0.6),
                'happy': (0.8, 0.5, 0.4),
                'neutral': (0.0, 0.0, 0.0),
                'sad': (-0.8, -0.5, -0.4)
            }
            
            # Get DAV values
            valence, arousal, dominance = dav_map.get(emotion, (None, None, None))
            
            # Adjust for intensity level
            if level == 'low' and valence is not None:
                # Decrease arousal and dominance for low intensity
                arousal *= 0.7
                dominance *= 0.7
                # Reduce valence (less positive or less negative)
                valence *= 0.7
            elif level == 'high' and valence is not None:
                # Increase arousal and dominance for high intensity
                arousal = min(1.0, arousal * 1.3) if arousal > 0 else max(-1.0, arousal * 1.3)
                dominance = min(1.0, dominance * 1.3) if dominance > 0 else max(-1.0, dominance * 1.3)
                # Amplify valence (more positive or more negative)
                valence = min(1.0, valence * 1.3) if valence > 0 else max(-1.0, valence * 1.3)
            
            # Add to data list
            data.append({
                'actor_id': actor_id,
                'sentence_id': sentence_id,
                'sex': sex,
                'race': race,
                'age': age,
                'emotion': emotion,
                'emotion_code': emotion_code,
                'level': level,
                'valence': valence,
                'arousal': arousal,
                'dominance': dominance,
                'wav_file': str(wav_file.relative_to(crema_dir))
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save full dataset
    df.to_csv(output_dir / "crema_d_full.csv", index=False)
    
    # Print statistics
    print(f"Total utterances: {len(df)}")
    
    # Print emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Download and prepare the CREMA-D dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory to store raw data")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to store processed data")
    args = parser.parse_args()
    
    # Download CREMA-D dataset
    crema_dir = download_crema_d(args.data_dir)
    
    # Process CREMA-D dataset
    process_crema_d(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
