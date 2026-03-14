import streamlit as st # type: ignore
import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np # type: ignore
import pickle
import pyttsx3 # type: ignore
import time
import threading

# ------------------------------
# Load trained model
# ------------------------------
MODEL_PATH = "models/gesture_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ------------------------------
# Speech function
# ------------------------------
def speak_async(text):

    def speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()

    threading.Thread(target=speak).start()


# ------------------------------
# Mediapipe setup
# ------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🤟 Hand Gesture Translator")
st.write("Real-time gesture recognition for deaf communication.")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

gesture_display = st.empty()
confidence_display = st.empty()

st.subheader("Sentence Builder")
sentence_box = st.empty()

clear_button = st.button("Clear Sentence")

st.subheader("Gesture History")
history_box = st.empty()

# ------------------------------
# Session State Storage
# ------------------------------
if "sentence" not in st.session_state:
    st.session_state.sentence = []

if "history" not in st.session_state:
    st.session_state.history = []

if "last_added_gesture" not in st.session_state:
    st.session_state.last_added_gesture = None

# Clear sentence button
if clear_button:
    st.session_state.sentence = []

# ------------------------------
# Camera setup
# ------------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

last_spoken_time = 0
speak_delay = 2

gesture_buffer = []

while run:

    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        st.write("Camera error.")
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    predicted_gesture = "None"
    confidence = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            features = []

            for lm in hand_landmarks.landmark:
                features.extend([lm.x, lm.y, lm.z])

            features = np.array(features).reshape(1, -1)

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            predicted_gesture = prediction
            confidence = np.max(probabilities) * 100
            CONFIDENCE_THRESHOLD = 70

            # Smooth predictions
            gesture_buffer.append(predicted_gesture)

            if len(gesture_buffer) > 5:
                gesture_buffer.pop(0)

            if gesture_buffer.count(predicted_gesture) >= 4 and confidence > CONFIDENCE_THRESHOLD:

                if time.time() - last_spoken_time > speak_delay:

                    speak_async(predicted_gesture)
                    last_spoken_time = time.time()

                    # Add to history
                    st.session_state.history.append(predicted_gesture)

                    if len(st.session_state.history) > 10:
                        st.session_state.history.pop(0)

                    # Add to sentence only if new gesture
                    if predicted_gesture != st.session_state.last_added_gesture:
                        st.session_state.sentence.append(predicted_gesture)
                        st.session_state.last_added_gesture = predicted_gesture

    # ------------------------------
    # Update UI
    # ------------------------------
    gesture_display.markdown(f"### Gesture: **{predicted_gesture}**")
    confidence_display.markdown(f"Confidence: **{confidence:.2f}%**")

    sentence_box.markdown(" ".join(st.session_state.sentence))

    history_box.markdown("\n".join(st.session_state.history))

    # ------------------------------
    # Camera overlay
    # ------------------------------
    cv2.putText(
        frame,
        f"Gesture: {predicted_gesture}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    fps = 1 / (time.time() - start_time)

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
