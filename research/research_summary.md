# Emotion Recognition Research Summary

## Key Papers

1. **Leveraging Semantic Information for Efficient Self-Supervised Emotion Recognition with Audio-Textual Distilled Models**
   - Authors: Danilo de Oliveira, Navin Raj Prabhu, Timo Gerkmann
   - Link: http://arxiv.org/abs/2305.19184v1
   - Focus: Self-supervised emotion recognition using audio-textual models

2. **Detecting Emotion Primitives from Speech and their use in discerning Categorical Emotions**
   - Authors: Vasudha Kowtha, Vikramjit Mitra, Chris Bartels, Erik Marchi, Sue Booker, William Caruso, Sachin Kajarekar, Devang Naik
   - Link: http://arxiv.org/abs/2002.01323v1
   - Focus: Converting speech to emotion primitives (valence, arousal, dominance) and then mapping to categorical emotions
   - **Potential baseline for our approach**

3. **Testing Correctness, Fairness, and Robustness of Speech Emotion Recognition Models**
   - Authors: Anna Derington, Hagen Wierstorf, Ali Özkil, Florian Eyben, Felix Burkhardt, Björn W. Schuller
   - Link: http://arxiv.org/abs/2312.06270v4
   - Focus: Evaluation methodologies for speech emotion recognition models

4. **Emotional Voice Messages (EMOVOME) database: emotion recognition in spontaneous voice messages**
   - Authors: Lucía Gómez Zaragozá, Rocío del Amor, Elena Parra Vargas, Valery Naranjo, Mariano Alcañiz Raya, Javier Marín-Morales
   - Link: http://arxiv.org/abs/2402.17496v2
   - Focus: New dataset for emotion recognition in voice messages
   - **Potential dataset for our approach**

5. **Sound-Based Recognition of Touch Gestures and Emotions for Enhanced Human-Robot Interaction**
   - Authors: Yuanbo Hou, Qiaoqiao Ren, Wenwu Wang, Dick Botteldooren
   - Link: http://arxiv.org/abs/2501.00038v1
   - Focus: Emotion recognition for human-robot interaction

## Relevant Datasets

Based on the literature review, the following datasets are commonly used for speech emotion recognition with valence, arousal, and dominance annotations:

1. **IEMOCAP** (Interactive Emotional Dyadic Motion Capture Database)
   - Contains approximately 12 hours of audiovisual data
   - Includes categorical emotion labels and dimensional ratings (valence, arousal, dominance)
   - Widely used in the research community as a benchmark dataset

2. **MSP-IMPROV** (MSP-IMPROV Corpus)
   - Contains audiovisual recordings of dyadic interactions
   - Includes both acted and spontaneous emotional expressions
   - Provides categorical and dimensional emotion annotations

3. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - Contains 24 professional actors (12 female, 12 male)
   - Includes speech and song recordings with 8 emotional expressions
   - Provides categorical emotion labels that can be mapped to dimensional space

4. **EMOVO** (Emotional Voice Database)
   - Italian emotional speech database
   - Contains recordings from 6 actors expressing 7 emotional states
   - Suitable for cross-lingual emotion recognition studies

5. **EMOVOME** (Emotional Voice Messages)
   - New dataset focused on spontaneous voice messages
   - Contains real-world emotional expressions
   - Includes both categorical and dimensional annotations

## Methodologies for Audio to DAV Conversion

Several approaches have been identified for converting audio features to dominance, arousal, and valence (DAV) values:

1. **Feature Extraction + Regression**
   - Extract acoustic features (MFCC, pitch, energy, etc.)
   - Train regression models (SVR, Random Forest, Neural Networks) to predict DAV values
   - Common baseline approach with good interpretability

2. **End-to-End Deep Learning**
   - Use raw audio or spectrograms as input
   - Train deep neural networks (CNN, RNN, Transformer) to directly predict DAV values
   - Higher performance but requires more data and computational resources

3. **Transfer Learning**
   - Leverage pre-trained audio models (wav2vec, HuBERT, etc.)
   - Fine-tune on emotion recognition tasks
   - State-of-the-art performance with less training data

4. **Self-Supervised Learning**
   - Pre-train models on large unlabeled audio datasets
   - Fine-tune on emotion recognition tasks
   - Emerging approach with promising results

## Mapping DAV to Emotion Categories

Several methods have been proposed for mapping dominance, arousal, and valence values to discrete emotion categories:

1. **Rule-Based Mapping**
   - Define regions in the DAV space corresponding to specific emotions
   - Simple and interpretable but may not capture complex emotional states

2. **Clustering**
   - Cluster DAV values in the training data
   - Assign emotion labels to clusters based on majority voting
   - Data-driven approach but may not generalize well

3. **Classification**
   - Train a classifier to map DAV values to emotion categories
   - Can capture complex relationships between DAV and emotions
   - Requires labeled data for both DAV and emotion categories

4. **Fuzzy Logic**
   - Define membership functions for each emotion category in the DAV space
   - Calculate membership degrees for each emotion
   - Handles uncertainty and ambiguity in emotion recognition

## Baseline Approach

Based on the literature review, a suitable baseline approach for our task would be:

1. **Audio to DAV Conversion**
   - Extract acoustic features using librosa (MFCC, spectral features, prosodic features)
   - Train a regression model (SVR or Random Forest) to predict DAV values
   - Evaluate using mean squared error (MSE) and concordance correlation coefficient (CCC)

2. **DAV to Emotion Mapping**
   - Train a classifier (SVM or Random Forest) to map DAV values to emotion categories
   - Evaluate using accuracy, F1-score, and confusion matrix

This baseline can be implemented using the IEMOCAP dataset, which provides both DAV annotations and emotion category labels.
