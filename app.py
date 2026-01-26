import streamlit as st

# Main app title
st.title("🎵 Music Genre Classification")
st.write("Upload a WAV audio file to predict its genre using machine learning!")

try:
    import joblib
    import numpy as np
    import os
    st.success("✓ System ready")
    
    # Load model and label encoder
    if os.path.exists("music_genre_model.pkl") and os.path.exists("label_encoder.pkl"):
        model = joblib.load("music_genre_model.pkl")
        label_encoder = joblib.load("label_encoder.pkl")
        st.success("✓ AI model loaded successfully")
        st.info(f"**Supported Genres:** {', '.join(label_encoder.classes_)}")
    else:
        st.error("❌ Model files not found. Please train the model first.")
        st.stop()
    
    # Import feature extraction
    from feature_extraction import extract_features
    st.success("✓ Audio processing ready")
    
    # File uploader section
    st.markdown("---")
    st.subheader("📁 Upload Your Music File")
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file", 
        type=["wav"],
        help="Upload a WAV format audio file to classify its genre"
    )
    
    if uploaded_file is not None:
        # Display file information
        st.write(f"**📄 File:** {uploaded_file.name}")
        st.write(f"**📊 Size:** {uploaded_file.size:,} bytes")
        
        # Audio player
        st.subheader("🎧 Audio Preview")
        st.audio(uploaded_file, format="audio/wav")
        
        # Prediction section
        st.subheader("🎯 Genre Prediction")
        
        if st.button("🚀 Analyze Music Genre", type="primary"):
            with st.spinner("🎵 Analyzing audio features..."):
                try:
                    # Save uploaded file temporarily
                    temp_filename = f"temp_{uploaded_file.name}"
                    with open(temp_filename, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Extract audio features
                    features = extract_features(temp_filename)
                    
                    # Clean up temporary file
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
                    
                    if features is not None:
                        # Make prediction
                        features = features.reshape(1, -1)
                        prediction = model.predict(features)
                        probabilities = model.predict_proba(features)
                        
                        predicted_genre = label_encoder.inverse_transform(prediction)[0]
                        confidence = max(probabilities[0]) * 100
                        
                        # Display main result
                        st.success(f"🎶 **Predicted Genre: {predicted_genre.upper()}**")
                        st.metric("🎯 Confidence Level", f"{confidence:.1f}%")
                        
                        # Show detailed predictions
                        st.subheader("📊 Detailed Analysis")
                        
                        # Create probability data
                        prob_data = []
                        for i, genre in enumerate(label_encoder.classes_):
                            prob = probabilities[0][i] * 100
                            prob_data.append((genre, prob))
                        
                        # Sort by probability
                        prob_data.sort(key=lambda x: x[1], reverse=True)
                        
                        # Show top 5 predictions
                        st.write("**Top 5 Genre Predictions:**")
                        for i, (genre, prob) in enumerate(prob_data[:5]):
                            if i == 0:
                                st.write(f"🥇 **{genre.capitalize()}**: {prob:.1f}%")
                            elif i == 1:
                                st.write(f"🥈 **{genre.capitalize()}**: {prob:.1f}%")
                            elif i == 2:
                                st.write(f"🥉 **{genre.capitalize()}**: {prob:.1f}%")
                            else:
                                st.write(f"   {i+1}. {genre.capitalize()}: {prob:.1f}%")
                        
                        # Show all probabilities as progress bars
                        st.write("**All Genre Probabilities:**")
                        for genre, prob in prob_data:
                            st.progress(prob/100, text=f"{genre.capitalize()}: {prob:.1f}%")
                        
                    else:
                        st.error("❌ Could not analyze the audio file. Please make sure it's a valid WAV file.")
                        
                except Exception as e:
                    st.error(f"❌ Error processing audio: {e}")
                    # Clean up on error
                    if 'temp_filename' in locals() and os.path.exists(temp_filename):
                        os.remove(temp_filename)
    
    else:
        # Instructions when no file is uploaded
        st.markdown("---")
        st.subheader("📋 How to Use")
        st.write("""
        1. **Upload** a WAV audio file using the file uploader above
        2. **Preview** your music using the audio player
        3. **Click** the "Analyze Music Genre" button
        4. **View** the predicted genre with confidence scores
        """)
        
        st.subheader("🎵 Supported Music Genres")
        if 'label_encoder' in locals():
            genres_list = [genre.capitalize() for genre in label_encoder.classes_]
            st.write(", ".join(genres_list))
        
        st.subheader("💡 Tips for Best Results")
        st.write("""
        - Use **WAV format** audio files
        - Audio should be at least **10 seconds long**
        - **High-quality** recordings work better
        - **Full songs** give more accurate results than short clips
        """)
    
except Exception as e:
    st.error(f"❌ Application Error: {e}")
    st.write("Please make sure all required files are present and dependencies are installed.")

# Footer
st.markdown("---")
st.markdown("**🎵 Built with Machine Learning and Streamlit**")
st.markdown("*Upload a WAV file above to get started!*")	