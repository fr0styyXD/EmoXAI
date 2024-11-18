# Speech Emotion Recognition with Explainable AI (XAI)
## Overview
This project delves into Speech Emotion Recognition using advanced Explainable Artificial Intelligence (XAI) techniques. The system processes speech data to identify emotions, ensuring transparent and interpretable predictions. It aims to improve real-world applications such as telehealth, education, and public safety, especially where understanding emotions is critical.

## Objectives
1. Accurately detect emotions from speech audio files.
2. Enhance interpretability using XAI tools to explain the decision-making process.
3. Build a robust system capable of handling noisy audio environments.
4. Facilitate practical applications like mental health support and remote learning.

## Features
1. Explainable AI: Integrates tools like LIME for clear insights into how predictions are made.
2. Advanced Preprocessing: Uses MFCC, CQT, and RASTA for robust feature extraction.
3. Deep Learning Model: Built with LSTM layers, optimized for sequence data processing.
4. Dynamic Training Strategies: Employs techniques like Early Stopping and learning rate reduction for efficient training.

## Dataset
### Toronto Emotional Speech Set (TESS):
1. Consists of 2,800 WAV files across seven emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, and neutral.
2. Two female actors (aged 26 and 64) performed a set of target phrases, organized by emotion and actor.

## Speech System Architecture
### Data Preprocessing
#### Feature Extraction:
1. MFCC (Mel-Frequency Cepstral Coefficients): Mimics human auditory perception.
2. CQT (Constant-Q Transform): Captures pitch characteristics on a logarithmic scale.
3. RASTA (Relative Spectral Transform): Filters out noise for cleaner feature representation.
4. Combined features stacked using NumPy for comprehensive input representation.

#### Label Encoding:
Converts categorical labels into integer formats for training.

#### Padding:
Ensures uniform feature dimensions across audio samples.

#### Visualization:
Utilizes waveplots, spectrograms, and CQT spectrograms to analyze audio data.

## Model Design
### Architecture:
LSTM layers for capturing temporal dependencies in speech.
Dense layers with ReLU and Softmax activations for classification.
Dropout (40%) to prevent overfitting.
Batch normalization for faster and stable training.

### Optimization:
Adam optimizer with a learning rate of 0.001.
Loss function: Categorical cross-entropy.
Metric: Accuracy.

### Callbacks:
EarlyStopping: Stops training when validation loss stagnates.
ReduceLROnPlateau: Lowers the learning rate to enhance convergence.

## Results
1. Training Metrics: High accuracy and low loss for multi-class emotion detection.
2. Model Explainability: XAI visualizations demonstrate feature importance and decision logic.

## Future Scope
1. Extend the system for real-time speech emotion recognition.
2. Explore multimodal integration with facial emotion recognition.
3. Improve robustness for diverse languages and accents.
