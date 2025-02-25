"""
Script to analyze the structure and properties of the synthetic dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_dataset(dataset_path, output_dir):
    """Analyze the dataset and generate visualizations."""
    # Set up directories for saving plots
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Print basic information
    print('Dataset shape:', df.shape)
    print('\nDataset columns:')
    print(df.columns.tolist())
    print('\nDataset info:')
    print(df.info())
    print('\nDataset statistics:')
    print(df.describe())
    
    # Analyze emotion distribution
    print('\nEmotion distribution:')
    emotion_counts = df['emotion'].value_counts()
    print(emotion_counts)
    
    # Plot emotion distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='emotion')
    plt.title('Emotion Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'emotion_distribution.png')
    
    # Analyze DAV values
    print('\nDAV values statistics:')
    print(df[['dominance', 'arousal', 'valence']].describe())
    
    # Plot DAV distributions
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.histplot(df['dominance'], kde=True)
    plt.title('Dominance Distribution')
    
    plt.subplot(1, 3, 2)
    sns.histplot(df['arousal'], kde=True)
    plt.title('Arousal Distribution')
    
    plt.subplot(1, 3, 3)
    sns.histplot(df['valence'], kde=True)
    plt.title('Valence Distribution')
    plt.tight_layout()
    plt.savefig(plots_dir / 'dav_distributions.png')
    
    # Plot DAV values by emotion
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='emotion', y='dominance')
    plt.title('Dominance by Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'dominance_by_emotion.png')
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='emotion', y='arousal')
    plt.title('Arousal by Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'arousal_by_emotion.png')
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df, x='emotion', y='valence')
    plt.title('Valence by Emotion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / 'valence_by_emotion.png')
    
    # Plot 3D scatter plot of DAV values colored by emotion
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    emotions = df['emotion'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(emotions)))
    
    for i, emotion in enumerate(emotions):
        subset = df[df['emotion'] == emotion]
        ax.scatter(subset['valence'], subset['arousal'], subset['dominance'], 
                   label=emotion, color=colors[i], alpha=0.7)
    
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.set_zlabel('Dominance')
    ax.set_title('3D Scatter Plot of DAV Values by Emotion')
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / 'dav_3d_scatter.png')
    
    # Analyze feature correlations with DAV values
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    dav_cols = ['dominance', 'arousal', 'valence']
    
    # Calculate correlations
    corr_matrix = df[feature_cols + dav_cols].corr()
    dav_feature_corr = corr_matrix.loc[dav_cols, feature_cols]
    
    # Plot heatmap of feature correlations with DAV values
    plt.figure(figsize=(20, 8))
    sns.heatmap(dav_feature_corr, cmap='coolwarm', center=0, annot=False)
    plt.title('Feature Correlations with DAV Values')
    plt.tight_layout()
    plt.savefig(plots_dir / 'feature_dav_correlations.png')
    
    # Find top correlated features for each DAV dimension
    top_n = 5
    for dav in dav_cols:
        print(f'\nTop {top_n} features correlated with {dav}:')
        top_corr = dav_feature_corr.loc[dav].abs().sort_values(ascending=False).head(top_n)
        print(top_corr)
    
    # Create a summary report
    with open(plots_dir / 'dataset_analysis_summary.txt', 'w') as f:
        f.write(f'Dataset Analysis Summary\n')
        f.write(f'======================\n\n')
        f.write(f'Dataset shape: {df.shape}\n\n')
        f.write(f'Emotion distribution:\n{emotion_counts}\n\n')
        f.write(f'DAV values statistics:\n{df[["dominance", "arousal", "valence"]].describe()}\n\n')
        
        for dav in dav_cols:
            f.write(f'Top {top_n} features correlated with {dav}:\n')
            top_corr = dav_feature_corr.loc[dav].abs().sort_values(ascending=False).head(top_n)
            f.write(f'{top_corr}\n\n')
    
    return df

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze the structure and properties of the dataset')
    parser.add_argument('--dataset_path', type=str, default='data/synthetic/small_synthetic_full.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--output_dir', type=str, default='data/analysis',
                        help='Directory to save analysis results')
    
    args = parser.parse_args()
    
    # Analyze the dataset
    analyze_dataset(args.dataset_path, args.output_dir)

if __name__ == '__main__':
    main()
