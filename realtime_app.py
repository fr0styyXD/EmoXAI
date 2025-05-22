import streamlit as st
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import io
import cv2
import soundfile as sf
import joblib
import time
import tempfile
from lime import lime_tabular, lime_image
from skimage.segmentation import mark_boundaries
import sounddevice as sd
import threading

# ---------- Load LIME training data ----------
X_train = np.load("X_train_ser.npy")
label_encoder = joblib.load("label_encoder_ser.pkl")
max_features = 284
time_steps = 124

# ------------------ Caching ------------------
@st.cache_resource
def load_model_ser():
    return tf.keras.models.load_model("models/serXAI.h5")

@st.cache_resource
def load_model_fer(masked=False):
    model_path = "models/maskedferVGG19.h5" if masked else "models/ferVGG19.h5"
    return tf.keras.models.load_model(model_path)

# ------------------ SER Feature Extraction ------------------
def extract_cqt(audio, sr):
    cqt = librosa.cqt(audio, sr=sr)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return cqt_db

def rasta_filtering(signal):
    pole = 0.94
    filtered_signal = np.zeros_like(signal)
    for i in range(signal.shape[0]):
        for j in range(1, signal.shape[1]):
            filtered_signal[i, j] = signal[i, j] - pole * signal[i, j - 1]
    return filtered_signal

def extract_features_from_wave(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    rasta = rasta_filtering(mfcc)
    cqt = extract_cqt(y, sr)
    return np.vstack([mfcc, cqt, rasta])

def predict_ser(model, y, sr):
    emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'surprise', 6: 'sad'}
    features = extract_features_from_wave(y, sr)
    padded = np.zeros((time_steps, max_features), dtype=np.float32)
    h = min(time_steps, features.shape[0])
    w = min(max_features, features.shape[1])
    padded[:h, :w] = features[:h, :w]
    input_tensor = np.expand_dims(padded, axis=0)
    prediction = model.predict(input_tensor)[0]
    return emotions[np.argmax(prediction)], prediction

def get_lime_plot_ser(y, sr):
    model = load_model_ser()
    features = extract_features_from_wave(y, sr)
    padded = np.zeros((time_steps, max_features))
    h = min(time_steps, features.shape[0])
    w = min(max_features, features.shape[1])
    padded[:h, :w] = features[:h, :w]
    instance = padded.reshape(1, -1)
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.reshape(X_train.shape[0], -1),
        feature_names=[f'feature_{i}' for i in range(X_train.shape[1] * X_train.shape[2])],
        class_names=label_encoder.classes_,
        mode="classification"
    )
    explanation = explainer.explain_instance(
        instance[0],
        lambda x: model.predict(x.reshape(-1, time_steps, max_features)),
        top_labels=1,
        num_features=10
    )
    top_label = explanation.top_labels[0]
    return explanation.as_pyplot_figure(label=top_label)

# ------------------ Audio Recording ------------------
def record_audio(duration=2, samplerate=22050):
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return recording.squeeze(), samplerate

# ------------------ FER ------------------
def preprocess_image_array(img_array):
    img_array = cv2.resize(img_array, (48, 48))
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    normalized = img_array / 255.0
    reshaped = normalized.reshape(1, 48, 48, 3)
    return reshaped, Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

def predict_fer(model, img_input, masked=False):
    if masked:
        emotions = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad', 4: 'surprise'}
    else:
        emotions = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
    prediction = model.predict(img_input)[0]
    return emotions[np.argmax(prediction)], prediction

def get_lime_plot_fer(image_pil, model):
    img_array = np.array(image_pil.resize((48, 48)).convert("RGB")).astype(np.uint8)
    explainer = lime_image.LimeImageExplainer()
    def predict_fn(images):
        return model.predict(np.array(images) / 255.0)
    explanation = explainer.explain_instance(
        image=img_array,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=5, hide_rest=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(img_array)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    ax[1].imshow(mark_boundaries(temp, mask))
    ax[1].set_title("Contributing Features")
    ax[1].axis("off")
    return fig

# ------------------ Webcam Capture ------------------
def capture_webcam_image():
    cap = cv2.VideoCapture(0)
    time.sleep(2)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

# ------------------ UI ------------------
def main():
    st.title("Real-Time Emotion Recognition with XAI")

    tab1, tab2 = st.tabs(["🎙️ Real-Time", "📁 File Upload"])

    with tab1:
        st.header("Live Emotion Recognition")

        mode = st.selectbox("Choose Mode", ["Speech (SER)", "Face (FER)", "Face (Masked FER)"])

        if mode == "Speech (SER)":
            st.write("Click below to record:")
            if st.button("🎤 Record Audio (2s)"):
                y, sr = record_audio()
                st.audio(np.array(y), sample_rate=sr)
                model = load_model_ser()
                emotion, _ = predict_ser(model, y, sr)
                st.success(f"Predicted Emotion: **{emotion}**")
                st.pyplot(get_lime_plot_ser(y, sr))

        else:
            st.write("Click below to capture image from webcam:")
            if st.button("📷 Capture Image"):
                frame = capture_webcam_image()
                if frame is not None:
                    img_input, image_pil = preprocess_image_array(frame)
                    use_masked = (mode == "Face (Masked FER)")
                    model = load_model_fer(masked=use_masked)
                    emotion, _ = predict_fer(model, img_input, masked=use_masked)
                    st.image(image_pil, caption=f"Predicted Emotion: {emotion}", width=250)
                    st.pyplot(get_lime_plot_fer(image_pil, model))
                else:
                    st.error("Failed to capture image from webcam.")

    with tab2:
        st.header("Upload File for Emotion Recognition")
        mode = st.selectbox("Choose Mode", ["Speech (SER)", "Face (FER)", "Face (Masked FER)"], key="upload_mode")

        if mode == "Speech (SER)":
            uploaded_audio = st.file_uploader("Upload a .wav file", type="wav")
            if uploaded_audio:
                y, sr = sf.read(io.BytesIO(uploaded_audio.read()))
                st.audio(uploaded_audio, format="audio/wav")
                model = load_model_ser()
                emotion, _ = predict_ser(model, y, sr)
                st.success(f"Predicted Emotion: **{emotion}**")
                st.pyplot(get_lime_plot_ser(y, sr))

        else:
            uploaded_image = st.file_uploader("Upload a face image", type=["png", "jpg", "jpeg"])
            if uploaded_image:
                file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
                img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                img_input, image_pil = preprocess_image_array(img_array)
                use_masked = (mode == "Face (Masked FER)")
                model = load_model_fer(masked=use_masked)
                emotion, _ = predict_fer(model, img_input, masked=use_masked)
                st.image(image_pil, caption=f"Predicted Emotion: {emotion}", width=250)
                st.pyplot(get_lime_plot_fer(image_pil, model))

if __name__ == "__main__":
    main()