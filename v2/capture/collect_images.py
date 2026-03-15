import cv2 # type: ignore
import os
import time

# CHANGE THIS EACH TIME
GESTURE_NAME = "thankyou"
IMAGES_TO_COLLECT = 300

DATASET_PATH = f"v2/dataset/{GESTURE_NAME}"

os.makedirs(DATASET_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
collecting = False
last_capture_time = 0

CAPTURE_DELAY = 0.25   # seconds between captures

print(f"Ready to collect images for: {GESTURE_NAME}")
print("Press 'S' to start capture")

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

    hand = cv2.resize(hand,(128,128))

    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    if collecting and count < IMAGES_TO_COLLECT:

        current_time = time.time()

        if current_time - last_capture_time > CAPTURE_DELAY:

            filename = f"{DATASET_PATH}/{count}.jpg"
            cv2.imwrite(filename,hand)

            count += 1
            last_capture_time = current_time

    cv2.putText(
        frame,
        f"Images: {count}/{IMAGES_TO_COLLECT}",
        (10,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("Dataset Collection",frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        collecting = True

    if key == 27 or count >= IMAGES_TO_COLLECT:
        break

cap.release()
cv2.destroyAllWindows()

print("Collection complete.")