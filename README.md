# Gestic 🤟

### Real-Time Hand Gesture Recognition System for Deaf Communication

Gestic is a machine learning–based system that translates hand gestures into **text and speech in real time**.
The goal of the project is to assist **deaf or speech-impaired individuals** in communicating with others using simple hand gestures.

The system detects hand gestures through a webcam, classifies them using a trained ML model, and converts them into readable text and audible speech.

---

# Abstract

Communication barriers often exist between hearing individuals and people who rely on sign language. Gestic aims to reduce this gap by providing a simple and real-time gesture translation system.

Using computer vision and machine learning, the system detects hand landmarks from webcam input and classifies gestures into predefined commands. The predicted gestures are displayed as text and can also be spoken aloud through a text-to-speech engine.

The project demonstrates how machine learning can be applied to build assistive technologies for accessibility and inclusive communication.

---

# Features

* Real-time gesture recognition
* Webcam-based hand detection
* Machine learning gesture classification
* Gesture-to-text translation
* Gesture-to-speech output
* Confidence score for predictions
* Sentence builder from multiple gestures
* Gesture history panel
* Clear sentence control

---

# Supported Gestures (V1)

The current version supports the following gestures:

* HELLO
* YES
* NO
* STOP
* THANK YOU

These gestures represent common conversational expressions.

---

# System Architecture

```
Webcam
   ↓
OpenCV Frame Capture
   ↓
Mediapipe Hand Landmark Detection
   ↓
Feature Extraction (21 landmarks → 63 values)
   ↓
RandomForest Gesture Classifier
   ↓
Prediction + Confidence Score
   ↓
Streamlit Interface
   ↓
Text Output + Speech Output
```

---

# Tech Stack

### Programming Language

* Python

### Computer Vision

* OpenCV

### Hand Tracking

* Mediapipe

### Machine Learning

* Scikit-learn
* RandomForest Classifier

### Interface

* Streamlit

### Speech Output

* pyttsx3

### Data Processing

* NumPy
* Pandas

---

# Dataset

The dataset was created manually using webcam capture.

For each gesture:

* 200 samples were collected
* Each sample contains **21 hand landmarks**
* Each landmark contains **(x, y, z)** coordinates

Total features per sample:

```
21 landmarks × 3 coordinates = 63 features
```

Dataset format:

```
gesture,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
HELLO,0.42,0.31,-0.02,...
```

---

# Machine Learning Model

A **Random Forest Classifier** was used for gesture classification.

Reasons for choosing Random Forest:

* Works well with small datasets
* Handles non-linear relationships
* Requires minimal parameter tuning
* Provides probability estimates for confidence scoring

### Training Pipeline

```
Dataset
   ↓
Feature Extraction
   ↓
Train-Test Split (80/20)
   ↓
RandomForest Training
   ↓
Model Evaluation
   ↓
Saved Model (.pkl)
```

The trained model is stored as:

```
models/gesture_model.pkl
```

---

# User Interface

The system uses a **Streamlit-based UI** that provides:

* Live camera feed
* Gesture prediction display
* Confidence score display
* Sentence builder
* Gesture history
* Clear sentence button

Example interface output:

```
Gesture: HELLO
Confidence: 94%

Sentence:
HELLO THANK YOU

History:
HELLO
THANK YOU
```

---

# Project Structure

```
gesture-translator
│
├── data
│   └── raw
│
├── models
│   └── gesture_model.pkl
│
├── src
│   ├── capture
│   │   └── collect_data.py
│   │
│   ├── training
│   │   └── train_model.py
│   │
│   └── inference
│       └── predict_gesture.py
│
├── ui
│   └── streamlit_app.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone <repo-url>
cd gestic
```

Create a Python virtual environment:

```
python -m venv venv
```

Activate environment:

```
venv\Scripts\activate
```

Install Python dependencies:

```
pip install -r requirements.txt
```

Install frontend dependencies for Version 2 UI:

```
cd gestic-ui
npm install
cd ..
```

---

# Running the Project (Both Versions)

The project has two runnable versions.

## Version 1 (Landmark + RandomForest + Streamlit)

### 1) Collect landmark data (optional if dataset already exists)

```
python src/capture/collect_data.py
```

### 2) Train model (optional if model already exists)

```
python src/training/train_model.py
```

### 3) Run Streamlit app

```
streamlit run ui/streamlit_app.py
```

### 4) Alternative webcam inference (without Streamlit)

```
python src/inference/predict_gesture.py
```

---

## Version 2 (CNN + FastAPI + React)

### 1) Collect image dataset (optional if dataset already exists)

```
python v2/capture/collect_images.py
```

### 2) Train CNN model (optional if model already exists)

```
python v2/cnn_model/train_cnn.py
```

### 3) Start FastAPI backend

```
uvicorn v2.cnn_model.api:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

### 4) Start React frontend

Open a new terminal:

```
cd gestic-ui
npm run dev
```

Frontend runs at:

```
http://127.0.0.1:5173
```

### 5) Optional standalone CNN webcam inference

```
python v2/cnn_model/predict_cnn.py
```

---

# Version Summary

## Version 1

* Hand landmark based recognition pipeline
* RandomForest classifier
* Streamlit interface + speech support

## Version 2

* Image-based CNN classification
* FastAPI backend for prediction API
* React + Vite + Framer Motion modern UI
* Browser speech output support

---

# Future Improvements

Planned enhancements for upcoming versions include:

* Dynamic gesture recognition
* Sentence prediction and grammar correction
* Larger gesture vocabulary
* Deep learning models (CNN / LSTM)
* React-based animated UI
* Mobile camera integration
* Real-time sign language translation

---

# Limitations

* Limited gesture vocabulary in current dataset
* Lighting and camera quality can affect predictions
* Static gesture focus (no full temporal sentence model yet)
* API and UI currently tuned for local development setup

---

# Applications

* Assistive technology for deaf communication
* Human-computer interaction
* Educational tools for sign language learning
* Gesture-based control systems

---

Mini Project
**Gestic – Hand Gesture Recognition System**

Developed as part of an academic project exploring the use of **Machine Learning and Computer Vision for accessibility technologies**.

---

# License

This project is developed for educational purposes.
