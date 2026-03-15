import cv2 #type: ignore
import numpy as np # type: ignore
import tensorflow as tf # type: ignore
from collections import deque
import pyttsx3 # type: ignore

MODEL_PATH = "v2/cnn_model/saved_model/gesture_cnn.keras"

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully.")

class_names = ['hello', 'no', 'stop', 'thankyou', 'yes']

IMG_SIZE = 128
prediction_buffer = deque(maxlen=5)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    h, w, _ = frame.shape

    box_size = 300

    x1 = w//2 - box_size//2
    y1 = h//2 - box_size//2
    x2 = x1 + box_size
    y2 = y1 + box_size

    hand = frame[y1:y2, x1:x2]

    hand_img = cv2.resize(hand,(IMG_SIZE,IMG_SIZE))

    img = hand_img / 255.0
    img = np.expand_dims(img,axis=0)

    predictions = model.predict(img,verbose=0)

    confidence = np.max(predictions)
    class_index = np.argmax(predictions)

    prediction_buffer.append(class_index)

    if len(prediction_buffer) == 5:
        class_index = max(set(prediction_buffer), key=prediction_buffer.count)

    gesture = class_names[class_index]

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.putText(
        frame,
        f"Gesture: {gesture}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence*100:.2f}%",
        (10,80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2
    )

    cv2.imshow("Gestic V2 CNN",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()