import streamlit as st
import keras
import numpy as np
import librosa


def convert_class_to_emotion(pred):
    """
    Method to convert the predictions (int) into human-readable strings.
    """

    label_conversion = {
        0: 'neutral',
        1: 'calm',
        2: 'happy',
        3: 'sad',
        4: 'angry',
        5: 'fearful',
        6: 'disgust',
        7: 'surprised'
    }

    return label_conversion.get(int(pred), 'Unknown')

def main():
    st.title("Speech Emotion Recognition App")

    st.sidebar.header("Settings")
    model_path = st.sidebar.text_input("Enter Model Path", "SER_model.h5")
    
    uploaded_file = st.sidebar.file_uploader("Choose an audio file", type=["wav"])

    if st.sidebar.button("Make Prediction") and uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        try:
            prediction = make_prediction(model_path, uploaded_file)
            st.write(f"Predicted Emotion: {prediction}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def make_prediction(model_path, audio_file):
    loaded_model = keras.models.load_model(model_path)
    summ = loaded_model.summary()
    data, sampling_rate = librosa.load(audio_file)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=1)
    x = np.expand_dims(x, axis=0)
    predict_x = loaded_model.predict(x)
    prediction = np.argmax(predict_x, axis=1)
    prediction = convert_class_to_emotion(prediction)
    return prediction

if __name__ == "__main__":
    main()
