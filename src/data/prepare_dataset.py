"""
Script to prepare a combined dataset for emotion recognition.

This script combines data from multiple datasets and creates train/val/test splits.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_combined_dataset(data_dir, output_dir, use_iemocap=True, use_ravdess=True):
    """Prepare a combined dataset from multiple sources."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dfs = []
    
    # Load IEMOCAP dataset if available
    if use_iemocap:
        iemocap_file = data_dir / "iemocap_dav_categorical.csv"
        if iemocap_file.exists():
            print(f"Loading IEMOCAP dataset from {iemocap_file}")
            df_iemocap = pd.read_csv(iemocap_file)
            df_iemocap['dataset'] = 'iemocap'
            dfs.append(df_iemocap)
        else:
            print(f"IEMOCAP dataset not found at {iemocap_file}")
    
    # Load RAVDESS dataset if available
    if use_ravdess:
        ravdess_file = data_dir / "ravdess_full.csv"
        if ravdess_file.exists():
            print(f"Loading RAVDESS dataset from {ravdess_file}")
            df_ravdess = pd.read_csv(ravdess_file)
            df_ravdess['dataset'] = 'ravdess'
            dfs.append(df_ravdess)
        else:
            print(f"RAVDESS dataset not found at {ravdess_file}")
    
    # Combine datasets
    if not dfs:
        print("No datasets found. Please download and process at least one dataset.")
        return None
    
    # Combine all dataframes
    df_combined = pd.concat(dfs, ignore_index=True)
    
    # Save combined dataset
    df_combined.to_csv(output_dir / "combined_full.csv", index=False)
    
    # Create train/val/test splits
    train_df, test_df = train_test_split(df_combined, test_size=0.2, random_state=42, stratify=df_combined['emotion'])
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['emotion'])
    
    # Save splits
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
    # Print statistics
    print(f"\nCombined dataset statistics:")
    print(f"Total samples: {len(df_combined)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Print emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = df_combined['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} ({count/len(df_combined)*100:.1f}%)")
    
    return df_combined

def main():
    parser = argparse.ArgumentParser(description="Prepare a combined dataset for emotion recognition")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Directory with processed datasets")
    parser.add_argument("--output_dir", type=str, default="data/final", help="Directory to store final dataset")
    parser.add_argument("--use_iemocap", action="store_true", help="Use IEMOCAP dataset")
    parser.add_argument("--use_ravdess", action="store_true", help="Use RAVDESS dataset")
    args = parser.parse_args()
    
    # Prepare combined dataset
    prepare_combined_dataset(args.data_dir, args.output_dir, args.use_iemocap, args.use_ravdess)

if __name__ == "__main__":
    main()
