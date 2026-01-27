import librosa
import numpy as np
import os

def extract_features(file_path):
    """Extract audio features from a single audio file"""
    try:
        print(f"Loading audio file: {file_path}")
        
        # Load audio (30 seconds max)
        y, sr = librosa.load(file_path, duration=30)
        print(f"Audio loaded: {len(y)} samples at {sr} Hz")

        # MFCCs
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfcc, axis=1)
        print(f"MFCC features: {len(mfcc_mean)}")

        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        print(f"Chroma features: {len(chroma_mean)}")

        # Spectral centroid
        spec_centroid = np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr)
        )
        print(f"Spectral centroid: {spec_centroid}")

        # Zero Crossing Rate
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        print(f"Zero crossing rate: {zcr}")

        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        print(f"Tempo: {tempo}")

        # Combine all features
        features = np.hstack([
            mfcc_mean,
            chroma_mean,
            spec_centroid,
            zcr,
            tempo
        ])

        return features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    """Test the feature extraction function"""
    print("=== Audio Feature Extraction Test ===")
    
    test_file = "genres_original/rock/rock.00000.wav"
    print(f"Testing with file: {test_file}")
    
    if os.path.exists(test_file):
        print("✓ File found")
        print("Extracting features...")
        
        features = extract_features(test_file)
        
        if features is not None:
            print(f"✓ Success! Feature vector length: {len(features)}")
            print(f"✓ Sample features: {features[:5]}")
            print("✓ Feature extraction working correctly!")
        else:
            print("✗ Failed to extract features")
    else:
        print(f"✗ File not found: {test_file}")
    
    print("=== Test Complete ===")

if __name__ == "__main__":
    main()
