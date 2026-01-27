<h1>Music Genre Classification Using Machine Learning</h1>

This project implements a machine learning–based system to automatically classify music audio files into different genres. The system analyzes audio signals, extracts meaningful features, and predicts the genre using a trained classification model. A web-based interface is provided for interactive usage.

<h2>Project Overview</h2>

With the rapid growth of digital music, automatic music genre classification plays an important role in music organization, recommendation systems, and content analysis. This project focuses on building an end-to-end pipeline that takes an audio file as input and outputs the predicted music genre along with confidence scores.

<h2>Objectives</h2>

Extract meaningful audio features from music files

Train a machine learning model for genre classification

Predict genres for unseen audio samples

Provide a user-friendly interface for interaction

<h2>Technologies Used</h2>

Python

Librosa

NumPy

Pandas

Scikit-learn

Matplotlib

Seaborn

Joblib

Streamlit

<h2>Project Structure</h2>
Music-Genre-Classifier/<br>
│<br>
├── app.py                   # Streamlit web application  <br>
├── predict.py               # Command-line prediction script <br>
├── train_model.py           # Model training script <br>
├── feature_extraction.py    # Audio feature extraction logic <br>
├── requirements.txt         # Project dependencies <br>
├── music_genre_model.pkl    # Trained machine learning model <br>
├── label_encoder.pkl        # Label encoder for genre labels <br>
└── README.md                # Project documentation <br>

<h2>Audio Feature Extraction</h2>

The system uses the Librosa library to extract numerical features from audio signals, including:

Mel Frequency Cepstral Coefficients (MFCCs)

Chroma features

Spectral centroid

Zero crossing rate

Tempo

These features represent timbre, pitch distribution, frequency characteristics, and rhythm of the music, which are essential for genre classification.

<h2>Model Training</h2>

Audio features are extracted from the dataset and stored as feature vectors

Genre labels are encoded using a label encoder

A machine learning classifier (Random Forest) is trained on the extracted features

The trained model and label encoder are saved using Joblib for reuse

<h2>Application Features</h2>

Upload WAV audio files

Audio playback for preview

Genre prediction with confidence score

Detailed probability analysis for all genres

<h2>Usage Guidelines</h2>

Use WAV format audio files

Audio duration should be at least 10 seconds

High-quality audio improves classification accuracy

Full-length music tracks provide better results

<h2>Conclusion</h2>

This project demonstrates a complete machine learning workflow for music genre classification, covering feature extraction, model training, prediction, and deployment through a web interface.
