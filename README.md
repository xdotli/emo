# Emotion Recognition from Multimodal Data

This project aims to detect emotions using multimodal data (audio and text). The approach involves:

1. Converting audio/text to numeric data (dominance, arousal, valence - DAV)
2. Mapping the DAV tuple to emotion categories (happy, sad, etc.)

## Project Structure

```
emo/
├── data/               # Dataset storage
├── models/             # Trained models
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code
│   ├── __init__.py
│   ├── data/           # Data processing utilities
│   ├── features/       # Feature extraction
│   ├── models/         # Model implementations
│   └── utils/          # Utility functions
└── tests/              # Unit tests
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```
