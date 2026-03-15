# GESTIC — AI Powered Hand Gesture Recognition System for Assistive Communication

## Development Note: Multi-Version Evolution

GESTIC was intentionally developed in multiple versions, where each version improved the design, intelligence, and usability of the previous system. The project started as a practical prototype to validate real-time gesture detection, then evolved into a deep learning and web-based architecture for better scalability, prediction quality, and user experience.

- Version 1: Prototype (Static Gesture Recognition)
- Version 2: Deep Learning + Modern Web UI

This staged development approach allowed iterative problem-solving: first proving that the concept works, then redesigning the system to overcome accuracy, flexibility, and interface limitations.

---

## 1. Abstract

GESTIC is an assistive communication system designed to support deaf or speech-impaired users by translating hand gestures into readable text and optional speech output. The system captures live camera input, detects hand patterns, classifies gestures using machine learning models, and presents the recognized gesture in a user-friendly interface.

The project combines computer vision, machine learning, deep learning, and web technologies to provide real-time gesture interpretation. In the initial version, the focus was on fast prototype implementation using hand landmarks and a basic classifier pipeline. In the improved version, a Convolutional Neural Network (CNN) was introduced for image-based recognition, along with a modern frontend-backend architecture using React and FastAPI.

GESTIC demonstrates how AI-powered perception systems can be applied to accessibility-focused applications and highlights the engineering journey from prototype to production-style modular design.

---

## 2. Introduction

Human communication is highly dependent on speech and hearing, which creates barriers for people who primarily communicate through sign language or gesture-based expressions. Assistive technologies that interpret hand gestures can reduce this communication gap by enabling machine-mediated translation in real time.

Gesture recognition systems have important applications in:

- Assistive communication for deaf and speech-impaired users
- Human-computer interaction without physical contact
- Smart classrooms and accessibility tools
- Healthcare and rehabilitation interfaces
- Industrial environments where touchless control is preferred

Despite their potential, gesture recognition systems are technically challenging. Major challenges include:

- Variability in lighting, background, and camera quality
- Differences in hand shape, speed, and orientation across users
- Distinguishing similar gestures with high confidence
- Maintaining low latency for real-time usability
- Building interfaces that are simple and intuitive for end users

GESTIC addresses these challenges through staged engineering. The first stage emphasized rapid proof-of-concept and fundamental gesture detection. The second stage introduced data-driven deep learning and modern UI architecture to increase robustness and improve user interaction quality.

---

## 3. Project Overview

GESTIC is a camera-based AI system that observes hand gestures and converts them into meaningful output for communication support.

Overall workflow:

Camera Input
→ Gesture Detection
→ Machine Learning Model
→ Gesture Classification
→ Text Output
→ Optional Speech Output

Conceptually, the system has four core layers:

1. Input Layer: Captures video frames from webcam.
2. Perception Layer: Extracts visual hand information (landmarks or image crops).
3. Intelligence Layer: Classifies gestures using a trained model.
4. Interaction Layer: Displays recognized gesture and can speak it aloud.

This layered design makes the system modular, allowing independent upgrades to detection logic, model architecture, and user interface.

---

## 4. Version 1 – Prototype System

Version 1 was built as a basic working prototype to validate the feasibility of real-time hand gesture translation in a low-complexity environment. The objective was to quickly create an end-to-end system that could detect a hand, infer a gesture category, and provide visible and audible output.

### 4.1 Technologies Used

- Python: Core scripting language for rapid experimentation.
- OpenCV: Webcam capture and frame processing.
- MediaPipe: Real-time hand landmark detection.
- Streamlit: Fast interface development for proof-of-concept UI.
- pyttsx3: Offline text-to-speech output.

Technology selection rationale:

- Python ecosystem provides mature libraries for vision and ML.
- OpenCV and MediaPipe together provide a practical baseline for hand tracking.
- Streamlit significantly reduces UI development time during prototyping.
- pyttsx3 enables local speech output without external APIs.

### 4.2 System Workflow

Version 1 processing flow:

Camera Feed
→ MediaPipe Hand Landmark Detection
→ Feature Extraction
→ Classification
→ Gesture Display
→ Text-to-Speech Output

Detailed logic:

1. Webcam frames are captured using OpenCV.
2. Frames are converted from BGR to RGB for MediaPipe compatibility.
3. MediaPipe Hands detects up to one hand and returns 21 landmarks.
4. Each landmark contributes x, y, z coordinates, forming a 63-feature vector.
5. Feature vectors are used for gesture prediction through a trained classifier.
6. Predicted gesture is shown in UI and optionally spoken.

How MediaPipe contributes:

MediaPipe Hands estimates 3D keypoints of fingers and palm in real time. This representation is compact and relatively robust compared to raw image pixels, making it suitable for early-stage prototypes with limited data.

### 4.3 Folder Structure (Version 1)

Version 1 implementation uses the following structure:

- data/raw: Landmark dataset CSV files.
- src/capture: Webcam and data collection scripts.
- src/training: Model training script for landmark-based classifier.
- src/inference: Live prediction script.
- models: Saved trained model artifact.
- ui: Streamlit application layer.

### 4.4 Limitations of Version 1

Although functional, Version 1 had multiple constraints:

- Limited gesture vocabulary and constrained feature generalization.
- Dependence on landmark quality under varying environments.
- Limited scalability for adding complex gesture classes.
- Prototype-oriented architecture with tight coupling.
- Reduced flexibility for modern web deployment.

These limitations motivated Version 2, where image-based deep learning and API-driven architecture were introduced for better adaptability, maintainability, and user experience.

---

## 5. Version 2 – Deep Learning System

Version 2 redesigned GESTIC into a modular AI application with deep learning-based recognition and a modern web interface. The objective shifted from prototype validation to system maturity: better classification behavior, cleaner separation of concerns, and richer UI interaction.

### 5.1 Technologies Used

Backend:

- Python: Core backend and ML orchestration.
- TensorFlow/Keras: CNN model definition, training, and inference.
- OpenCV: Image capture and preprocessing.
- FastAPI: Lightweight, high-performance prediction API.

Frontend:

- React: Component-based interactive UI.
- Vite: Fast frontend build and development tooling.
- Framer Motion: UI animation and visual transitions.
- Web Speech API: Browser-native speech synthesis for gesture output.

Technology rationale:

- TensorFlow/Keras supports rapid CNN experimentation and deployment-ready model serialization.
- FastAPI provides low-overhead REST endpoints suitable for real-time frame inference.
- React and Vite enable maintainable frontend architecture with fast iteration.
- Framer Motion improves usability through animated visual feedback.

### 5.2 CNN Model Architecture

The Version 2 classifier uses a Convolutional Neural Network for image classification. CNNs are effective for vision tasks because they learn hierarchical spatial features directly from pixel data.

Model structure (high level):

1. Convolution Layer Block 1: Learns low-level edges and texture patterns.
2. Pooling Layer 1: Reduces spatial size and computation.
3. Convolution Layer Block 2: Learns richer shape-level features.
4. Pooling Layer 2: Further dimensionality reduction.
5. Convolution Layer Block 3: Captures high-level gesture patterns.
6. Pooling Layer 3: Compresses feature maps.
7. Flatten Layer: Converts feature maps into dense vector form.
8. Dense Layer: Learns class-specific decision boundaries.
9. Dropout Layer: Reduces overfitting risk.
10. Softmax Output Layer: Produces class probabilities.

Why CNN is suitable:

- Automatically extracts discriminative visual features.
- Handles variation in gesture appearance more effectively than manual features.
- Improves class separation for image-based hand gestures.

### 5.3 Dataset Creation

Version 2 dataset was collected using webcam capture with controlled frame extraction.

Dataset organization:

- v2/dataset/hello
- v2/dataset/no
- v2/dataset/stop
- v2/dataset/thankyou
- v2/dataset/yes

Collection process:

1. User performs a target gesture inside a capture box.
2. Frames are cropped from region of interest.
3. Images are resized to 128 x 128.
4. Images are saved class-wise into folder structure.
5. Collected dataset is used for supervised training.

Current class setup includes five classes with balanced samples per class, supporting stable early-stage CNN training.

### 5.4 Model Training Process

Training pipeline:

Dataset Loading
→ Image Preprocessing
→ Training/Validation Split
→ CNN Training
→ Model Evaluation
→ Saving Trained Model

Details:

- Keras dataset loader reads images directly from class folders.
- Data is split into training and validation subsets.
- Pixel values are normalized through rescaling.
- CNN trains for multiple epochs to optimize cross-entropy loss.
- Validation metrics are monitored for generalization behavior.
- Final model is saved in Keras format for inference reuse.

Training behavior expectation:

- Accuracy improves progressively with epochs.
- Validation trends indicate how well the model generalizes.
- Overfitting risk is partially addressed via dropout and validation monitoring.

### 5.5 Backend Architecture

Version 2 backend is implemented using FastAPI and exposes prediction as a REST service.

Primary endpoint:

- POST /predict

Request-response workflow:

React sends image frame
→ FastAPI receives image bytes
→ OpenCV decodes and resizes frame
→ CNN model predicts class probabilities
→ Backend returns JSON gesture and confidence

Advantages of backend design:

- Decouples model inference from frontend rendering.
- Enables future integration with mobile or external clients.
- Supports maintainable API contracts and modular scaling.

### 5.6 Frontend Architecture

Version 2 frontend is implemented as a React application.

Core frontend flow:

1. Browser webcam stream is initialized.
2. Hidden canvas captures periodic frame snapshots.
3. Snapshot is sent to backend API as multipart form data.
4. JSON prediction response updates gesture and confidence state.
5. User can trigger speech synthesis for recognized gesture.

Main interface capabilities:

- Live webcam preview
- Continuous prediction updates
- Confidence display
- Speech button for auditory output
- Animated UI transitions

Communication mechanism:

- Frontend uses HTTP fetch requests to call FastAPI endpoint.
- Backend responses are consumed as JSON and mapped to UI state.

### 5.7 UI Design and Animation

The Version 2 interface emphasizes clarity and engagement through a modern visual system.

Design elements include:

- Animated gradient-like atmospheric background
- Floating color blobs with motion paths
- Glassmorphism-inspired card layout
- Prominent gesture and confidence display panel
- Audio waveform animation during speaking state

Framer Motion usage:

- Continuous background element motion
- Gesture card transition effects on prediction updates
- Smooth interaction feedback to enhance perceived responsiveness

This design improves readability and usability while maintaining the real-time monitoring focus of the application.

### 5.8 Project Folder Structure

Final workspace structure (functional view):

gesture/
- data/
  - raw/
  - processed/
- models/
- src/
  - capture/
  - training/
  - inference/
- ui/
- v2/
  - dataset/
  - capture/
  - cnn_model/
    - saved_model/
- gestic-ui/
  - src/
  - public/
- docs/
- notebooks/

Folder purpose summary:

- src and ui represent Version 1 prototype pipeline.
- v2 contains deep learning dataset, training, and API logic.
- gestic-ui contains modern React frontend.
- models and saved_model store trained artifacts.
- data stores structured training inputs for classical pipeline.

---

## 6. System Architecture Diagram

Version 2 end-to-end pipeline:

Browser Camera
→ React Frontend
→ FastAPI Backend
→ CNN Model
→ Prediction Response
→ UI Update

Stage explanation:

1. Browser Camera: Captures real-time user gesture stream.
2. React Frontend: Extracts and transmits selected frames.
3. FastAPI Backend: Validates and preprocesses incoming frame.
4. CNN Model: Computes class probabilities from image tensor.
5. Prediction Response: Returns gesture label and confidence score.
6. UI Update: Displays result and enables optional speech output.

This architecture separates presentation, inference service, and model logic for cleaner extensibility.

---

## 7. Features of the Final System

- Real-time webcam-based gesture detection
- AI-based gesture recognition using CNN
- Confidence-aware prediction display
- Animated modern web interface
- Optional speech output for recognized gestures
- Modular frontend-backend architecture
- Multi-version development demonstrating system evolution
- Scalable baseline for future gesture set expansion

---

## 8. Challenges Faced

Key engineering challenges during development:

- Achieving consistent model accuracy with limited dataset size
- Handling frame quality variation due to lighting and camera noise
- Balancing prediction frequency with UI responsiveness
- Managing frontend-backend latency for near real-time inference
- Avoiding false positives between visually similar gestures
- Maintaining smooth UX while running frequent network requests
- Integrating speech output without disrupting continuous detection

These challenges informed architecture decisions such as buffering, confidence display, model refinement, and modular service separation.

---

## 9. Future Improvements

Potential high-impact enhancements:

- Dynamic sentence formation from gesture sequences
- Continuous sign-to-text translation with temporal modeling
- Larger and more diverse gesture vocabulary
- Data augmentation and transfer learning for better generalization
- Mobile deployment using TensorFlow Lite or ONNX runtime
- Model quantization and optimization for low-latency inference
- User calibration mode for personalized recognition behavior
- Authentication, role-based access, and secure API deployment
- Multilingual speech synthesis and caption export

---

## 10. Conclusion

GESTIC demonstrates a complete AI engineering lifecycle for an accessibility-centered application. Starting from a landmark-based prototype and evolving to a CNN-powered web system, the project shows clear technical progression in model design, architecture, and user interaction.

The final system delivers real-time gesture recognition with readable and spoken outputs, supported by modular components that are suitable for extension and deployment-oriented refinement. Beyond technical implementation, GESTIC highlights the social value of assistive AI, offering a practical foundation for future communication tools that can reduce barriers for deaf and speech-impaired communities.

In summary, GESTIC is both a successful academic project and a meaningful prototype for inclusive human-centered technology.