# 🎭 Emotion Detection with Explainable AI (XAI)

This project implements a **multimodal emotion recognition system** combining **Speech Emotion Recognition (SER)**, **Facial Emotion Recognition (FER)**, and **Masked Facial Emotion Recognition** using **Deep Learning**. To ensure transparency, it integrates **LIME-based Explainable AI (XAI)**, showing *which regions of audio and face contribute most* to the predicted emotion.

The system runs locally with a **Streamlit dashboard** for user-friendly interaction and visualization.

---

## 🚀 Features

- **Speech Emotion Recognition (SER)** using an LSTM model
- **Facial Emotion Recognition (FER)** using VGG19
- **Masked Facial Emotion Recognition** using VGG19 (robust to masks)
- **XAI with LIME** – explains predictions for both images and audio
- **Multimodal Fusion** with rule-based logic
- **Streamlit Frontend** for file upload & real-time webcam/mic detection

---

## 📊 System Architecture

```
Video / Audio Input
        │
        ├──▶ Audio ──▶ SER Model (LSTM) ──▶ LIME
        ├──▶ Frames ──▶ FER Model (VGG19) ──▶ LIME
        └──▶ Frames ──▶ Masked FER Model (VGG19) ──▶ LIME

                ▼
        Fusion & Rule-Based Logic
                ▼
        Final Emotion Prediction + Explanations
```

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python |
| **Deep Learning** | TensorFlow, Keras |
| **Feature Extraction** | Librosa, OpenCV |
| **Explainable AI** | LIME |

---

## 📂 Project Structure

```
emotion-xai/
│── app.py                # Streamlit dashboard for uploads
│── realtime_app.py       # Real-time webcam + mic mode
│── models/               # Place your .h5 models here (not uploaded in repo)
│── notebooks/            # Jupyter notebooks for model training
│── utils/                # Helper scripts (audio, video, xai)
│── outputs/              # LIME explanations & results
│── requirements.txt
│── README.md
```

> ⚠️ **Note:** Models (`.h5` files) are **not included in this repo** due to large size. You must manually place them in the `models/` folder before running.

---

## ⚡ Installation & Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-xai.git
cd emotion-xai
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place trained models

- Download your trained models (`serXAI.h5`, `ferVGG19.h5`, `maskedferVGG19.h5`)
- Put them inside the `models/` folder

```
emotion-xai/
│── models/
     ├── serXAI.h5
     ├── ferVGG19.h5
     └── maskedferVGG19.h5
```

### 5. Run the Streamlit app locally

**File Upload Dashboard:**

```bash
streamlit run app.py
```

**Real-Time Detection (Webcam + Mic):**

```bash
streamlit run realtime_app.py
```

After running, open the local URL shown in terminal (usually `http://localhost:8501`).

---

## 🎥 Demo

- **GitHub Repository**: [🔗 Add Link Here]
- **Streamlit App (Local)**: Runs via `streamlit run app.py`

---

## ✅ Limitations

- Models are not uploaded due to large size → must be placed manually
- Explanations from **LIME** depend on input quality
- Real-time mode may lag on low-end devices

---

## 🔮 Future Work

- [ ] Add **Grad-CAM** and **SHAP** explainability
- [ ] Improve fusion with attention-based models
- [ ] Extend dataset for multilingual & cultural variations
- [ ] Cloud-based deployment for broader access

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or feedback, please reach out to:

- **GitHub**: [your-username](https://github.com/your-username)
- **Email**: your.email@example.com

---

## 🙏 Acknowledgments

- LIME library for Explainable AI
- TensorFlow and Keras teams
- Streamlit for the amazing dashboard framework

---

**Made with ❤️ for transparent AI**