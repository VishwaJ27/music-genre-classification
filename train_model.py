import os
import numpy as np
import joblib
from feature_extraction import extract_features
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import time
import sys

# Force output to show immediately
sys.stdout.reconfigure(line_buffering=True)

print("=" * 60)
print("MUSIC GENRE CLASSIFICATION TRAINING")
print("=" * 60)
print("Processing full dataset (this may take 10-30 minutes)...")
print("Please be patient - processing 1000 audio files...")

# Dataset path
DATASET_PATH = "genres_original"

features = []
labels = []

print("\nExtracting features from audio files...")

start_time = time.time()
total_files = 0

# Loop through genres
genres = sorted([d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])

for genre_idx, genre in enumerate(genres):
    genre_path = os.path.join(DATASET_PATH, genre)
    
    print(f"\n[{genre_idx+1}/{len(genres)}] Processing genre: {genre.upper()}")
    
    genre_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
    file_count = 0
    
    for file in genre_files:
        file_path = os.path.join(genre_path, file)
        
        # Extract features (without verbose output)
        data = extract_features(file_path)
        
        if data is not None:
            features.append(data)
            labels.append(genre)
            file_count += 1
            total_files += 1
            
            # Show progress every 20 files
            if file_count % 20 == 0:
                elapsed = time.time() - start_time
                progress = (total_files / 1000) * 100
                print(f"  Progress: {file_count}/{len(genre_files)} files | Total: {total_files}/1000 ({progress:.1f}%) | Time: {elapsed:.1f}s")
    
    print(f"✓ Completed {genre}: {file_count} files processed")

elapsed_total = time.time() - start_time
print(f"\n✓ Feature extraction complete! ({elapsed_total:.1f}s total)")

X = np.array(features)
y = np.array(labels)

print(f"\nDataset Summary:")
print(f"Total samples: {X.shape[0]}")
print(f"Feature dimensions: {X.shape[1]}")
print(f"Genres found: {len(set(labels))}")

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"Genre labels: {list(label_encoder.classes_)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nDataset split:")
print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1  # Use all CPU cores
)

print(f"\nTraining Random Forest model (200 trees)...")
print("Using all CPU cores for faster training...")

train_start = time.time()
model.fit(X_train, y_train)
train_time = time.time() - train_start

print(f"✓ Model training complete! ({train_time:.1f}s)")

# Evaluate model
print("\nEvaluating model performance...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n" + "=" * 40)
print(f"MODEL PERFORMANCE RESULTS")
print(f"=" * 40)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print(f"\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Save model and label encoder
print("Saving model and label encoder...")

joblib.dump(model, "music_genre_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

# Check file sizes
model_size = os.path.getsize("music_genre_model.pkl")
encoder_size = os.path.getsize("label_encoder.pkl")

print(f"✓ Model saved as 'music_genre_model.pkl' ({model_size:,} bytes)")
print(f"✓ Label encoder saved as 'label_encoder.pkl' ({encoder_size:,} bytes)")

total_time = time.time() - start_time
print(f"\n" + "=" * 60)
print(f"TRAINING COMPLETE! Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"=" * 60)

# Feature importance
print(f"\nTop 10 Most Important Features:")
feature_names = [f"MFCC_{i}" for i in range(20)] + [f"Chroma_{i}" for i in range(12)] + ["Spectral_Centroid", "ZCR", "Tempo"]
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for i in range(min(10, len(feature_names))):
    idx = indices[i]
    print(f"{i+1:2d}. {feature_names[idx]:15s}: {importances[idx]:.4f}")

print(f"\n Your music genre classifier is ready!")
print(f"Test it with: python predict.py genres_original/rock/rock.00000.wav")
