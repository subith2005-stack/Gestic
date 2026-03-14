import cv2 # type: ignore
import mediapipe as mp # type: ignore
import csv
import os

# -------------------------------
# CHANGE THIS FOR EACH GESTURE
# -------------------------------
GESTURE_NAME = "THANK_YOU"
SAMPLES_TO_COLLECT = 200
# -------------------------------

# Dataset file
DATASET_PATH = "data/raw/gesture_data.csv"

# Create folder if not exists
os.makedirs("data/raw", exist_ok=True)

# Initialize mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Open webcam
cap = cv2.VideoCapture(0)

sample_count = 0

# Create CSV if not exists
if not os.path.exists(DATASET_PATH):
    with open(DATASET_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["gesture"]
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        writer.writerow(header)

print(f"Collecting data for gesture: {GESTURE_NAME}")

while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmark coordinates
            row = [GESTURE_NAME]

            for lm in hand_landmarks.landmark:
                row.extend([lm.x, lm.y, lm.z])

            # Save to CSV
            with open(DATASET_PATH, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            sample_count += 1

    cv2.putText(
        frame,
        f"Samples: {sample_count}/{SAMPLES_TO_COLLECT}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Dataset Collection", frame)

    if sample_count >= SAMPLES_TO_COLLECT:
        break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Data collection complete.")
