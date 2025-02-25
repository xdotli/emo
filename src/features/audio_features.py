"""
Module for extracting audio features from audio files.

This module provides functions to extract various audio features
that can be used for emotion recognition.
"""

import numpy as np
import librosa
import pandas as pd
from pathlib import Path
import soundfile as sf

def extract_features_from_file(audio_file, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract audio features from an audio file.
    
    Parameters:
    -----------
    audio_file : str or Path
        Path to the audio file
    sr : int, optional
        Sample rate for audio processing
    n_mfcc : int, optional
        Number of MFCCs to extract
    n_fft : int, optional
        FFT window size
    hop_length : int, optional
        Hop length for feature extraction
    
    Returns:
    --------
    dict
        Dictionary containing extracted features
    """
    # Load audio file
    try:
        y, sr = librosa.load(audio_file, sr=sr)
    except Exception as e:
        print(f"Error loading audio file {audio_file}: {e}")
        return None
    
    # Extract features
    return extract_features_from_signal(y, sr, n_mfcc, n_fft, hop_length)

def extract_features_from_signal(y, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract audio features from an audio signal.
    
    Parameters:
    -----------
    y : numpy.ndarray
        Audio signal
    sr : int, optional
        Sample rate for audio processing
    n_mfcc : int, optional
        Number of MFCCs to extract
    n_fft : int, optional
        FFT window size
    hop_length : int, optional
        Hop length for feature extraction
    
    Returns:
    --------
    dict
        Dictionary containing extracted features
    """
    features = {}
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    for i in range(n_mfcc):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
    
    # Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Extract zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Extract RMS energy
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # Extract tempo and beat features
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    
    # Extract chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    
    return features

def batch_extract_features(audio_files, output_file=None, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Extract features from multiple audio files.
    
    Parameters:
    -----------
    audio_files : list
        List of paths to audio files
    output_file : str or Path, optional
        Path to save the extracted features as CSV
    sr : int, optional
        Sample rate for audio processing
    n_mfcc : int, optional
        Number of MFCCs to extract
    n_fft : int, optional
        FFT window size
    hop_length : int, optional
        Hop length for feature extraction
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing extracted features for all files
    """
    all_features = []
    
    for audio_file in audio_files:
        features = extract_features_from_file(audio_file, sr, n_mfcc, n_fft, hop_length)
        
        if features is not None:
            features['file'] = str(audio_file)
            all_features.append(features)
    
    # Create DataFrame
    df = pd.DataFrame(all_features)
    
    # Save to CSV if output file provided
    if output_file is not None:
        df.to_csv(output_file, index=False)
    
    return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract audio features from audio files')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing audio files')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save the extracted features as CSV')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Sample rate for audio processing')
    parser.add_argument('--n_mfcc', type=int, default=13,
                        help='Number of MFCCs to extract')
    
    args = parser.parse_args()
    
    # Get audio files
    input_dir = Path(args.input_dir)
    audio_files = list(input_dir.glob('*.wav'))
    
    if not audio_files:
        print(f"No audio files found in {input_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files")
    
    # Extract features
    df = batch_extract_features(
        audio_files, 
        args.output_file, 
        args.sr, 
        args.n_mfcc
    )
    
    print(f"Extracted features from {len(df)} audio files")
    print(f"Saved features to {args.output_file}")

if __name__ == '__main__':
    main()
