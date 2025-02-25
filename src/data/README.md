# Data Processing Module

This module contains scripts for downloading, processing, and preparing datasets for emotion recognition.

## Datasets

### IEMOCAP

The Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is a multimodal database of acted dyadic interactions. It contains approximately 12 hours of audiovisual data from 10 actors. The data is labeled with categorical emotion labels and dimensional ratings (valence, arousal, dominance).

**Note:** IEMOCAP requires permission to access. You need to request access from the USC SAIL lab.

### RAVDESS

The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) is a validated multimodal database of emotional speech and song. It contains recordings from 24 professional actors (12 female, 12 male) vocalizing two lexically-matched statements in a neutral North American accent.

The RAVDESS is publicly available and can be downloaded automatically using the provided script.

### CREMA-D

The Crowd-sourced Emotional Multimodal Actors Dataset (CREMA-D) is a dataset of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities. Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) and four different emotion levels (Low, Medium, High, and Unspecified).

The CREMA-D dataset is publicly available and can be downloaded automatically using the provided script.

## Usage

### Download and Process IEMOCAP

```bash
python src/data/download_iemocap.py --data_dir=data/raw --output_dir=data/processed
```

### Download and Process RAVDESS

```bash
python src/data/download_ravdess.py --data_dir=data/raw --output_dir=data/processed
```

### Download and Process CREMA-D

```bash
python src/data/download_crema_d.py --data_dir=data/raw --output_dir=data/processed
```

### Prepare Combined Dataset

```bash
python src/data/prepare_dataset.py --data_dir=data/processed --output_dir=data/final --use_crema_d
```

## Data Format

The processed datasets are saved as CSV files with the following columns:

- `emotion`: Categorical emotion label (e.g., happy, sad, angry)
- `valence`: Valence value (-1 to 1, negative to positive)
- `arousal`: Arousal value (-1 to 1, calm to excited)
- `dominance`: Dominance value (-1 to 1, submissive to dominant)
- `wav_file`: Path to the audio file relative to the dataset directory
- Other metadata columns specific to each dataset
