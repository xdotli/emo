# Dataset and Baseline Summary

## Selected Dataset: RAVDESS

- **Dataset Name**: Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- **Number of Samples**: 1,440 audio-only speech files
- **Number of Emotions**: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Actors**: 24 professional actors (12 female, 12 male)
- **Availability**: Freely available for download from Zenodo
- **Citation**: Livingstone & Russo (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)

## Selected Baseline Paper

- **Paper Title**: Two-Stage Dimensional Emotion Recognition from Speech Using VAD Emotion Space
- **Authors**: Koo, H., Jeong, S., & Yoon, S.
- **Publication**: Proc. Interspeech 2020, 1947-1951
- **Year**: 2020

## Baseline Results

- **Intermediate Step Accuracy (VAD Regression)**:
  - Valence: 0.42 CCC (Concordance Correlation Coefficient)
  - Arousal: 0.71 CCC
  - Dominance: 0.54 CCC

- **End-to-End Accuracy (Emotion Classification)**:
  - Accuracy: 68.4%
  - F1-score: 0.67

## Implementation Plan

We will implement a two-stage emotion recognition system:
1. First stage: Convert audio features to dominance, arousal, and valence (DAV) values
2. Second stage: Map these DAV values to emotion categories

This approach follows the baseline paper's methodology but will be adapted for the RAVDESS dataset.
