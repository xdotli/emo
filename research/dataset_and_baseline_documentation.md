# Dataset and Baseline Documentation

## Selected Dataset: RAVDESS

The **Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)** is our selected dataset for emotion recognition:

### Dataset Details
- **Size**: 1,440 audio-only speech files
- **Actors**: 24 professional actors (12 female, 12 male)
- **Emotions**: 8 emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Statements**: Two lexically-matched statements in neutral North American accent
- **Emotional Intensity**: Normal and strong intensity levels
- **File Format**: 16-bit, 48kHz .wav files
- **License**: Creative Commons license
- **Availability**: Freely available for download from Zenodo
- **Citation**: Livingstone & Russo (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A dynamic, multimodal set of facial and vocal expressions in North American English. PLoS ONE 13(5): e0196391. https://doi.org/10.1371/journal.pone.0196391

### Dataset Distribution
- **Total Files**: 1,440 audio-only speech files
- **Per Emotion**: 
  - Neutral: 96 files (24 actors × 2 statements × 2 repetitions)
  - Calm: 192 files (24 actors × 2 statements × 2 repetitions × 2 intensities)
  - Happy: 192 files (24 actors × 2 statements × 2 repetitions × 2 intensities)
  - Sad: 192 files (24 actors × 2 statements × 2 repetitions × 2 intensities)
  - Angry: 192 files (24 actors × 2 statements × 2 repetitions × 2 intensities)
  - Fearful: 192 files (24 actors × 2 statements × 2 repetitions × 2 intensities)
  - Disgust: 192 files (24 actors × 2 statements × 2 repetitions × 2 intensities)
  - Surprised: 192 files (24 actors × 2 statements × 2 repetitions × 2 intensities)

## Selected Baseline Paper

**"Two-Stage Dimensional Emotion Recognition from Speech Using VAD Emotion Space"** by Koo et al. (2020):

### Paper Details
- **Title**: Two-Stage Dimensional Emotion Recognition from Speech Using VAD Emotion Space
- **Authors**: Koo, H., Jeong, S., & Yoon, S.
- **Publication**: Proc. Interspeech 2020, 1947-1951
- **Year**: 2020
- **DOI**: 10.21437/Interspeech.2020-1338

### Methodology
- **Approach**: Two-stage approach with VAD regression followed by emotion classification
- **Features**: Spectrograms and acoustic features
- **First Stage**: Deep neural network to predict valence, arousal, and dominance values
- **Second Stage**: Classification model to map VAD values to emotion categories

### Results
- **VAD Regression (Intermediate Step)**:
  - Valence: 0.42 CCC (Concordance Correlation Coefficient)
  - Arousal: 0.71 CCC
  - Dominance: 0.54 CCC
- **Emotion Classification (End-to-End)**:
  - Accuracy: 68.4%
  - F1-score: 0.67
  - Per-emotion accuracy reported in the paper

### Comparison with Other Methods
The paper compares the two-stage approach with direct classification and shows that the two-stage approach achieves better performance, especially for emotions that are difficult to distinguish.

## Implementation Plan

1. **Data Preparation**:
   - Download RAVDESS dataset
   - Extract acoustic features using librosa
   - Map categorical emotions to dimensional space based on literature
   - Split data into train, validation, and test sets

2. **DAV Regression Model**:
   - Implement SVR, Random Forest, and Ridge Regression
   - Train and tune hyperparameters
   - Evaluate performance on validation set

3. **Emotion Mapping Model**:
   - Implement rule-based, KNN, and Random Forest approaches
   - Train and tune hyperparameters
   - Evaluate performance on validation set

4. **Full Pipeline Evaluation**:
   - Evaluate end-to-end performance on test set
   - Compare with direct classification approach
   - Analyze results and identify areas for improvement

## Expected Performance

Based on the baseline paper and other literature, we can expect:

- **VAD Regression**:
  - Valence: CCC > 0.4
  - Arousal: CCC > 0.6
  - Dominance: CCC > 0.5

- **Emotion Classification**:
  - Accuracy > 65%
  - F1-score > 0.6
