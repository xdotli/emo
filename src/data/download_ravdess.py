"""
Script to download and prepare the RAVDESS dataset.

The RAVDESS dataset is publicly available and can be downloaded automatically.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file from a URL with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    filename_str = str(filename)  # Convert Path to string for tqdm
    
    with open(filename, 'wb') as file, tqdm(
            desc=filename_str,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_ravdess(data_dir):
    """Download the RAVDESS dataset."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    ravdess_dir = data_dir / "RAVDESS"
    ravdess_dir.mkdir(exist_ok=True)
    
    # URL for the audio-only dataset
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    
    # Download the dataset
    zip_file = ravdess_dir / "Audio_Speech_Actors_01-24.zip"
    
    if not zip_file.exists():
        print(f"Downloading RAVDESS dataset to {zip_file}...")
        download_file(url, zip_file)
    else:
        print(f"RAVDESS dataset already downloaded at {zip_file}")
    
    # Extract the dataset
    extract_dir = ravdess_dir / "Audio_Speech_Actors_01-24"
    
    if not extract_dir.exists():
        print(f"Extracting RAVDESS dataset to {extract_dir}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(ravdess_dir)
    else:
        print(f"RAVDESS dataset already extracted at {extract_dir}")
    
    return ravdess_dir

def process_ravdess(data_dir, output_dir):
    """Process the RAVDESS dataset and create CSV files with metadata."""
    ravdess_dir = Path(data_dir) / "RAVDESS" / "Audio_Speech_Actors_01-24"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define emotion mapping
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    # Define intensity mapping
    intensity_map = {
        '01': 'normal',
        '02': 'strong'
    }
    
    # Define statement mapping
    statement_map = {
        '01': 'Kids are talking by the door',
        '02': 'Dogs are sitting by the door'
    }
    
    # Define gender mapping
    gender_map = {
        'odd': 'male',
        'even': 'female'
    }
    
    # Initialize lists to store data
    data = []
    
    # Process each actor directory
    for actor_dir in ravdess_dir.glob("Actor_*"):
        # Process each audio file
        for wav_file in actor_dir.glob("*.wav"):
            # Parse filename
            # Format: modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav
            # Example: 03-01-01-01-01-01-01.wav
            parts = wav_file.stem.split('-')
            
            if len(parts) == 7:
                modality = parts[0]  # 01=full-AV, 02=video-only, 03=audio-only
                vocal_channel = parts[1]  # 01=speech, 02=song
                emotion_code = parts[2]
                intensity_code = parts[3]
                statement_code = parts[4]
                repetition = parts[5]
                actor_id = parts[6]
                
                # Skip if not speech
                if vocal_channel != '01':
                    continue
                
                # Get emotion, intensity, and statement
                emotion = emotion_map.get(emotion_code, 'unknown')
                intensity = intensity_map.get(intensity_code, 'unknown')
                statement = statement_map.get(statement_code, 'unknown')
                
                # Get gender
                gender = gender_map.get('even' if int(actor_id) % 2 == 0 else 'odd', 'unknown')
                
                # Map categorical emotions to approximate DAV values
                # These are approximate values based on literature
                dav_map = {
                    'neutral': (0.0, 0.0, 0.0),
                    'calm': (0.3, -0.3, 0.0),
                    'happy': (0.8, 0.5, 0.4),
                    'sad': (-0.8, -0.5, -0.4),
                    'angry': (-0.5, 0.7, 0.7),
                    'fearful': (-0.7, 0.6, -0.6),
                    'disgust': (-0.6, 0.2, 0.3),
                    'surprised': (0.4, 0.7, -0.1)
                }
                
                # Get DAV values
                valence, arousal, dominance = dav_map.get(emotion, (None, None, None))
                
                # Adjust for intensity
                if intensity == 'strong' and valence is not None:
                    # Increase arousal and dominance for strong intensity
                    arousal = min(1.0, arousal * 1.5) if arousal > 0 else max(-1.0, arousal * 1.5)
                    dominance = min(1.0, dominance * 1.2) if dominance > 0 else max(-1.0, dominance * 1.2)
                    # Amplify valence (more positive or more negative)
                    valence = min(1.0, valence * 1.2) if valence > 0 else max(-1.0, valence * 1.2)
                
                # Add to data list
                data.append({
                    'actor_id': actor_id,
                    'gender': gender,
                    'emotion': emotion,
                    'emotion_code': emotion_code,
                    'intensity': intensity,
                    'statement': statement,
                    'repetition': repetition,
                    'valence': valence,
                    'arousal': arousal,
                    'dominance': dominance,
                    'wav_file': str(wav_file.relative_to(ravdess_dir.parent))
                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save full dataset
    df.to_csv(output_dir / "ravdess_full.csv", index=False)
    
    # Print statistics
    print(f"Total utterances: {len(df)}")
    
    # Print emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Download and prepare the RAVDESS dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory to store raw data")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to store processed data")
    args = parser.parse_args()
    
    # Download RAVDESS dataset
    ravdess_dir = download_ravdess(args.data_dir)
    
    # Process RAVDESS dataset
    process_ravdess(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
