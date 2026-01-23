import joblib
import numpy as np
import os
import sys

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    
    print("=== MUSIC GENRE PREDICTION ===")
    print("Loading model...")
    
    try:
        model = joblib.load("music_genre_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        print("✓ Model loaded successfully")
        
        if len(sys.argv) > 1:
            audio_file = sys.argv[1]
        else:
            audio_file = "test_song.wav"
        
        print(f"Testing file: {audio_file}")
        
        if not os.path.exists(audio_file):
            print(f"ERROR: File not found - {audio_file}")
            return
        
        print("Extracting features...")
        
        from feature_extraction import extract_features
        
        features = extract_features(audio_file)
        
        if features is None:
            print("ERROR: Could not extract features from audio file")
            return
        
        print(f"✓ Extracted {len(features)} features")
        
        features = features.reshape(1, -1)
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        
        predicted_genre = label_encoder.inverse_transform(prediction)[0]
        confidence = max(probabilities[0]) * 100
        
        print("\n" + "="*40)
        print(f" PREDICTED GENRE: {predicted_genre.upper()}")
        print(f" CONFIDENCE: {confidence:.1f}%")
        print("="*40)
        
        print("\nAll genre probabilities:")
        for i, genre in enumerate(label_encoder.classes_):
            prob = probabilities[0][i] * 100
            bar = "BAR" * int(prob / 5)  # Simple bar chart
            print(f"{genre:10s}: {prob:5.1f}% {bar}")
        
        print("\n✓ Prediction completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

