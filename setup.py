from setuptools import setup, find_packages

setup(
    name="emo",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "librosa>=0.9.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "torch>=1.10.0",
        "torchaudio>=0.10.0",
        "transformers>=4.15.0",
        "soundfile>=0.10.0",
    ],
)
