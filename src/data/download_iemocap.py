"""
Script to download and prepare the IEMOCAP dataset.

Note: The IEMOCAP dataset requires permission to access.
This script provides instructions for obtaining the dataset
and then processes it once downloaded.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def check_iemocap_exists(data_dir):
    """Check if IEMOCAP dataset exists in the specified directory."""
    iemocap_dir = Path(data_dir) / "IEMOCAP_full_release"
    
    if not iemocap_dir.exists():
        print(f"IEMOCAP dataset not found at {iemocap_dir}")
        print("\nTo obtain the IEMOCAP dataset:")
        print("1. Visit: https://sail.usc.edu/iemocap/")
        print("2. Fill out the request form")
        print("3. Once approved, download the dataset")
        print("4. Extract the dataset to the specified directory")
        print(f"5. Run this script again with --data_dir={data_dir}")
        return False
    
    return True

def process_iemocap(data_dir, output_dir):
    """Process the IEMOCAP dataset and create CSV files with metadata."""
    iemocap_dir = Path(data_dir) / "IEMOCAP_full_release"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define emotion mapping
    emotion_map = {
        'ang': 'angry',
        'hap': 'happy',
        'exc': 'excited',
        'sad': 'sad',
        'neu': 'neutral',
        'fru': 'frustrated',
        'fea': 'fear',
        'sur': 'surprised',
        'dis': 'disgust',
        'oth': 'other',
        'xxx': 'unknown'
    }
    
    # Initialize lists to store data
    data = []
    
    # Process each session
    for session in range(1, 6):
        session_dir = iemocap_dir / f"Session{session}"
        labels_dir = session_dir / "dialog" / "EmoEvaluation"
        
        # Process each evaluation file
        for eval_file in labels_dir.glob("*.txt"):
            if eval_file.name.startswith("Ses"):
                dialog_id = eval_file.stem
                
                # Read evaluation file
                with open(eval_file, 'r') as f:
                    lines = f.readlines()
                
                # Extract utterance information
                for line in lines:
                    if line.startswith("["):
                        parts = line.strip().split("\t")
                        if len(parts) >= 4:
                            utterance_id = parts[0].strip("[]")
                            emotion = parts[1].split(",")[0]
                            
                            # Extract valence, arousal, dominance if available
                            val, aro, dom = None, None, None
                            if len(parts) >= 5 and "val" in parts[2]:
                                val_str = parts[2].split("=")[1].strip()
                                if val_str.replace(".", "", 1).isdigit():
                                    val = float(val_str)
                            
                            if len(parts) >= 5 and "act" in parts[3]:
                                aro_str = parts[3].split("=")[1].strip()
                                if aro_str.replace(".", "", 1).isdigit():
                                    aro = float(aro_str)
                            
                            if len(parts) >= 5 and "dom" in parts[4]:
                                dom_str = parts[4].split("=")[1].strip()
                                if dom_str.replace(".", "", 1).isdigit():
                                    dom = float(dom_str)
                            
                            # Find audio file
                            wav_dir = session_dir / "sentences" / "wav" / dialog_id
                            wav_file = wav_dir / f"{utterance_id}.wav"
                            
                            if wav_file.exists():
                                # Add to data list
                                data.append({
                                    'session': session,
                                    'dialog_id': dialog_id,
                                    'utterance_id': utterance_id,
                                    'emotion': emotion_map.get(emotion, 'unknown'),
                                    'emotion_code': emotion,
                                    'valence': val,
                                    'arousal': aro,
                                    'dominance': dom,
                                    'wav_file': str(wav_file.relative_to(iemocap_dir))
                                })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save full dataset
    df.to_csv(output_dir / "iemocap_full.csv", index=False)
    
    # Save dataset with complete DAV values
    df_dav = df.dropna(subset=['valence', 'arousal', 'dominance'])
    df_dav.to_csv(output_dir / "iemocap_dav.csv", index=False)
    
    # Save dataset with categorical emotions (excluding 'unknown' and 'other')
    df_cat = df[~df['emotion'].isin(['unknown', 'other'])]
    df_cat.to_csv(output_dir / "iemocap_categorical.csv", index=False)
    
    # Save dataset with both DAV and categorical emotions
    df_both = df_dav[~df_dav['emotion'].isin(['unknown', 'other'])]
    df_both.to_csv(output_dir / "iemocap_dav_categorical.csv", index=False)
    
    # Print statistics
    print(f"Total utterances: {len(df)}")
    print(f"Utterances with DAV values: {len(df_dav)}")
    print(f"Utterances with categorical emotions: {len(df_cat)}")
    print(f"Utterances with both DAV and categorical emotions: {len(df_both)}")
    
    # Print emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = df_cat['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} ({count/len(df_cat)*100:.1f}%)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Download and prepare the IEMOCAP dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory to store raw data")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to store processed data")
    args = parser.parse_args()
    
    # Check if IEMOCAP dataset exists
    if not check_iemocap_exists(args.data_dir):
        sys.exit(1)
    
    # Process IEMOCAP dataset
    process_iemocap(args.data_dir, args.output_dir)

if __name__ == "__main__":
    main()
