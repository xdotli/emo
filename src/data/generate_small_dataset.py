"""
Script to generate a small synthetic dataset for quick testing.

This script creates a small synthetic dataset with pre-computed features
and corresponding dominance, arousal, valence (DAV) values and emotion categories.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

def emotion_to_dav(emotion):
    """Map emotion to dominance, arousal, valence (DAV) values."""
    # Define DAV values for each emotion
    # Values are in the range [-1, 1]
    dav_map = {
        'happy': (0.4, 0.6, 0.8),     # High valence, moderate-high arousal, moderate dominance
        'sad': (-0.4, -0.6, -0.8),    # Low valence, low arousal, low dominance
        'angry': (0.7, 0.8, -0.6),    # High dominance, high arousal, negative valence
        'fearful': (-0.7, 0.7, -0.8), # Low dominance, high arousal, negative valence
        'disgust': (0.3, 0.4, -0.7),  # Moderate dominance, moderate arousal, negative valence
        'neutral': (0.0, 0.0, 0.0),   # Neutral on all dimensions
        'surprised': (-0.2, 0.9, 0.3),# Low-moderate dominance, high arousal, moderate positive valence
        'calm': (0.2, -0.7, 0.5)      # Moderate dominance, low arousal, moderate positive valence
    }
    
    return dav_map.get(emotion, (0.0, 0.0, 0.0))

def generate_features(emotion):
    """Generate synthetic features for the given emotion."""
    # Define base features for each emotion
    feature_map = {
        'happy': np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]),
        'sad': np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1]),
        'angry': np.array([0.7, 0.8, 0.9, -0.6, -0.5, -0.4, 0.3, 0.2]),
        'fearful': np.array([-0.7, 0.8, 0.9, -0.6, -0.5, -0.4, -0.3, -0.2]),
        'disgust': np.array([0.3, 0.4, 0.5, -0.7, -0.6, -0.5, -0.4, -0.3]),
        'neutral': np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1]),
        'surprised': np.array([-0.2, 0.9, 0.8, 0.3, 0.2, 0.1, 0.0, -0.1]),
        'calm': np.array([0.2, -0.7, -0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    }
    
    # Get base features for the emotion
    base_features = feature_map.get(emotion, np.zeros(8))
    
    # Add random noise to make each sample unique
    noise = np.random.normal(0, 0.1, len(base_features))
    features = base_features + noise
    
    # Extend to more features by adding random values
    extended_features = np.random.normal(0, 0.2, 32)
    
    # Combine base features and extended features
    all_features = np.concatenate([features, extended_features])
    
    return all_features

def generate_small_dataset(output_dir, num_samples=100):
    """Generate a small synthetic dataset for emotion recognition."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define emotions
    emotions = ['happy', 'sad', 'angry', 'fearful', 'disgust', 'neutral', 'surprised', 'calm']
    
    # Initialize lists to store data
    data = []
    
    # Generate samples
    for i in range(num_samples):
        # Select a random emotion
        emotion = np.random.choice(emotions)
        
        # Generate features
        features = generate_features(emotion)
        
        # Map emotion to DAV values
        dominance, arousal, valence = emotion_to_dav(emotion)
        
        # Add some noise to DAV values to simulate real-world variation
        dominance += np.random.normal(0, 0.1)
        arousal += np.random.normal(0, 0.1)
        valence += np.random.normal(0, 0.1)
        
        # Clip DAV values to [-1, 1]
        dominance = np.clip(dominance, -1.0, 1.0)
        arousal = np.clip(arousal, -1.0, 1.0)
        valence = np.clip(valence, -1.0, 1.0)
        
        # Add to data list
        data.append({
            'id': i,
            'emotion': emotion,
            'dominance': dominance,
            'arousal': arousal,
            'valence': valence,
            **{f'feature_{j}': value for j, value in enumerate(features)}
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save full dataset
    df.to_csv(output_dir / "small_synthetic_full.csv", index=False)
    
    # Create train/val/test splits
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['emotion'])
    
    # Save splits
    train_df.to_csv(output_dir / "small_train.csv", index=False)
    val_df.to_csv(output_dir / "small_val.csv", index=False)
    test_df.to_csv(output_dir / "small_test.csv", index=False)
    
    # Print statistics
    print(f"Generated {len(df)} synthetic samples")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Print emotion distribution
    print("\nEmotion distribution:")
    emotion_counts = df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count} ({count/len(df)*100:.1f}%)")
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Generate a small synthetic dataset for emotion recognition")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Directory to store the synthetic dataset")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to generate")
    args = parser.parse_args()
    
    # Generate dataset
    generate_small_dataset(args.output_dir, args.num_samples)

if __name__ == "__main__":
    main()
