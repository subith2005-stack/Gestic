import streamlit as st  # type: ignore
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
# Text-to-speech function
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

            predicted_gesture = model.predict(features)[0]

            # Smooth predictions
            gesture_buffer.append(predicted_gesture)

            if len(gesture_buffer) > 5:
                gesture_buffer.pop(0)

            # Check stable prediction
            if gesture_buffer.count(predicted_gesture) >= 4:

                if time.time() - last_spoken_time > speak_delay:
                    speak_async(predicted_gesture)
                    last_spoken_time = time.time()

    # Display gesture
    cv2.putText(
        frame,
        f"Gesture: {predicted_gesture}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # FPS counter
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
