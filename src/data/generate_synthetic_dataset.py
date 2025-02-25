"""
Script to generate a synthetic dataset for emotion recognition development.

This script creates a synthetic dataset with audio features and corresponding
dominance, arousal, valence (DAV) values and emotion categories.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

def generate_tone(frequency, duration, sample_rate=22050, amplitude=0.5):
    """Generate a pure tone with the given frequency and duration."""
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = amplitude * np.sin(2 * np.pi * frequency * t)
    return tone

def add_vibrato(tone, rate=5, depth=0.1, sample_rate=22050):
    """Add vibrato to a tone."""
    t = np.linspace(0, len(tone) / sample_rate, len(tone), False)
    vibrato = depth * np.sin(2 * np.pi * rate * t)
    indices = np.clip(np.arange(len(tone)) + (vibrato * sample_rate).astype(int), 0, len(tone) - 1)
    return tone[indices.astype(int)]

def add_tremolo(tone, rate=5, depth=0.3):
    """Add tremolo to a tone."""
    t = np.linspace(0, len(tone) / 22050, len(tone), False)
    tremolo = 1 + depth * np.sin(2 * np.pi * rate * t)
    return tone * tremolo

def add_noise(tone, noise_level=0.01):
    """Add white noise to a tone."""
    noise = np.random.normal(0, noise_level, len(tone))
    return tone + noise

def apply_envelope(tone, attack=0.1, decay=0.1, sustain=0.7, release=0.1):
    """Apply ADSR envelope to a tone."""
    total_length = len(tone)
    attack_samples = int(attack * total_length)
    decay_samples = int(decay * total_length)
    sustain_samples = int(sustain * total_length)
    release_samples = int(release * total_length)
    
    envelope = np.ones(total_length)
    
    # Attack
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay
    if decay_samples > 0:
        envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, 0.7, decay_samples)
    
    # Sustain is already set to 1
    
    # Release
    if release_samples > 0:
        envelope[-release_samples:] = np.linspace(0.7, 0, release_samples)
    
    return tone * envelope

def generate_emotional_tone(emotion, duration=3.0, sample_rate=22050):
    """Generate a tone with characteristics matching the given emotion."""
    # Base frequencies for different emotions
    emotion_params = {
        'happy': {
            'frequency': 440,  # A4
            'vibrato_rate': 6,
            'vibrato_depth': 0.05,
            'tremolo_rate': 6,
            'tremolo_depth': 0.2,
            'noise_level': 0.005,
            'attack': 0.05,
            'decay': 0.1,
            'sustain': 0.7,
            'release': 0.15,
            'amplitude': 0.7
        },
        'sad': {
            'frequency': 220,  # A3
            'vibrato_rate': 3,
            'vibrato_depth': 0.1,
            'tremolo_rate': 2,
            'tremolo_depth': 0.1,
            'noise_level': 0.01,
            'attack': 0.1,
            'decay': 0.2,
            'sustain': 0.6,
            'release': 0.3,
            'amplitude': 0.5
        },
        'angry': {
            'frequency': 330,  # E4
            'vibrato_rate': 8,
            'vibrato_depth': 0.15,
            'tremolo_rate': 8,
            'tremolo_depth': 0.4,
            'noise_level': 0.03,
            'attack': 0.02,
            'decay': 0.05,
            'sustain': 0.8,
            'release': 0.1,
            'amplitude': 0.9
        },
        'fearful': {
            'frequency': 293.66,  # D4
            'vibrato_rate': 7,
            'vibrato_depth': 0.2,
            'tremolo_rate': 7,
            'tremolo_depth': 0.3,
            'noise_level': 0.02,
            'attack': 0.03,
            'decay': 0.1,
            'sustain': 0.7,
            'release': 0.2,
            'amplitude': 0.6
        },
        'disgust': {
            'frequency': 277.18,  # C#4
            'vibrato_rate': 4,
            'vibrato_depth': 0.1,
            'tremolo_rate': 5,
            'tremolo_depth': 0.25,
            'noise_level': 0.025,
            'attack': 0.07,
            'decay': 0.15,
            'sustain': 0.65,
            'release': 0.2,
            'amplitude': 0.65
        },
        'neutral': {
            'frequency': 261.63,  # C4
            'vibrato_rate': 0,
            'vibrato_depth': 0,
            'tremolo_rate': 0,
            'tremolo_depth': 0,
            'noise_level': 0.001,
            'attack': 0.1,
            'decay': 0.1,
            'sustain': 0.7,
            'release': 0.1,
            'amplitude': 0.6
        },
        'surprised': {
            'frequency': 392,  # G4
            'vibrato_rate': 10,
            'vibrato_depth': 0.1,
            'tremolo_rate': 10,
            'tremolo_depth': 0.2,
            'noise_level': 0.01,
            'attack': 0.01,
            'decay': 0.05,
            'sustain': 0.8,
            'release': 0.1,
            'amplitude': 0.8
        },
        'calm': {
            'frequency': 196,  # G3
            'vibrato_rate': 2,
            'vibrato_depth': 0.03,
            'tremolo_rate': 2,
            'tremolo_depth': 0.05,
            'noise_level': 0.002,
            'attack': 0.15,
            'decay': 0.2,
            'sustain': 0.6,
            'release': 0.25,
            'amplitude': 0.4
        }
    }
    
    params = emotion_params.get(emotion, emotion_params['neutral'])
    
    # Generate base tone
    tone = generate_tone(
        params['frequency'], 
        duration, 
        sample_rate, 
        params['amplitude']
    )
    
    # Add effects
    if params['vibrato_depth'] > 0:
        tone = add_vibrato(
            tone, 
            params['vibrato_rate'], 
            params['vibrato_depth'], 
            sample_rate
        )
    
    if params['tremolo_depth'] > 0:
        tone = add_tremolo(
            tone, 
            params['tremolo_rate'], 
            params['tremolo_depth']
        )
    
    tone = add_noise(tone, params['noise_level'])
    
    # Apply envelope
    tone = apply_envelope(
        tone, 
        params['attack'], 
        params['decay'], 
        params['sustain'], 
        params['release']
    )
    
    return tone

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

def extract_features(audio, sample_rate):
    """Extract audio features from the audio signal."""
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    # Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_std = np.std(spectral_centroid)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_std = np.std(spectral_bandwidth)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    spectral_rolloff_std = np.std(spectral_rolloff)
    
    # Extract zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    
    # Extract RMS energy
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # Combine all features
    features = np.concatenate([
        mfccs_mean, mfccs_std,
        [spectral_centroid_mean, spectral_centroid_std],
        [spectral_bandwidth_mean, spectral_bandwidth_std],
        [spectral_rolloff_mean, spectral_rolloff_std],
        [zcr_mean, zcr_std],
        [rms_mean, rms_std]
    ])
    
    return features

def generate_dataset(output_dir, num_samples=1000, sample_rate=22050):
    """Generate a synthetic dataset for emotion recognition."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create directories for audio files
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    # Define emotions
    emotions = ['happy', 'sad', 'angry', 'fearful', 'disgust', 'neutral', 'surprised', 'calm']
    
    # Initialize lists to store data
    data = []
    
    # Generate samples
    for i in range(num_samples):
        # Select a random emotion
        emotion = np.random.choice(emotions)
        
        # Generate audio with the selected emotion
        duration = np.random.uniform(1.0, 5.0)  # Random duration between 1 and 5 seconds
        audio = generate_emotional_tone(emotion, duration, sample_rate)
        
        # Add some random variation to make each sample unique
        variation = np.random.uniform(0.9, 1.1)
        audio = audio * variation
        
        # Extract features
        features = extract_features(audio, sample_rate)
        
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
        
        # Save audio file
        audio_file = f"sample_{i:04d}_{emotion}.wav"
        audio_path = audio_dir / audio_file
        sf.write(audio_path, audio, sample_rate)
        
        # Add to data list
        data.append({
            'id': i,
            'emotion': emotion,
            'dominance': dominance,
            'arousal': arousal,
            'valence': valence,
            'duration': duration,
            'wav_file': str(audio_path.relative_to(output_dir)),
            **{f'feature_{j}': value for j, value in enumerate(features)}
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save full dataset
    df.to_csv(output_dir / "synthetic_full.csv", index=False)
    
    # Create train/val/test splits
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
    train_df, val_df = train_test_split(train_df, test_size=0.25, random_state=42, stratify=train_df['emotion'])
    
    # Save splits
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    
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
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset for emotion recognition")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Directory to store the synthetic dataset")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--sample_rate", type=int, default=22050, help="Sample rate for audio files")
    args = parser.parse_args()
    
    # Generate dataset
    generate_dataset(args.output_dir, args.num_samples, args.sample_rate)

if __name__ == "__main__":
    main()
