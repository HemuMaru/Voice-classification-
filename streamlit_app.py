import streamlit as st
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import librosa

def convert_class_to_emotion(pred):
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

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))  # 8 classes for emotions

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

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
    data, sampling_rate = librosa.load(audio_file, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
    x = np.expand_dims(mfccs, axis=0)
    x = np.expand_dims(x, axis=3)  # Add a channel dimension for CNN input
    predict_x = loaded_model.predict(x)
    prediction = np.argmax(predict_x, axis=1)
    prediction = convert_class_to_emotion(prediction)
    return prediction

if __name__ == "__main__":
    main()
