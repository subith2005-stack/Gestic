import cv2 # type: ignore
import os

# ------------------------------
# CHANGE GESTURE NAME EACH TIME
# ------------------------------
GESTURE_NAME = "thank you"
IMAGES_TO_COLLECT = 400
# ------------------------------

DATASET_PATH = f"v2/dataset/{GESTURE_NAME}"

os.makedirs(DATASET_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0
collecting = False

print(f"Ready to collect images for: {GESTURE_NAME}")
print("Press 'S' to start automatic capture")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    # Define crop box (center region)
    box_size = 300

    x1 = w//2 - box_size//2
    y1 = h//2 - box_size//2

    x2 = x1 + box_size
    y2 = y1 + box_size

    hand = frame[y1:y2, x1:x2]

    hand = cv2.resize(hand, (128,128))

    # Draw capture box
    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # Show instructions
    if not collecting:
        cv2.putText(
            frame,
            "Press S to start capture",
            (10,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    # Capture images automatically
    if collecting and count < IMAGES_TO_COLLECT:

        filename = f"{DATASET_PATH}/{count}.jpg"
        cv2.imwrite(filename, hand)

        count += 1

        cv2.putText(
            frame,
            f"Capturing: {count}/{IMAGES_TO_COLLECT}",
            (10,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )

    cv2.imshow("Dataset Collection", frame)

    key = cv2.waitKey(1)

    # Start collecting
    if key == ord('s'):
        collecting = True

    # Stop if enough images
    if count >= IMAGES_TO_COLLECT:
        break

    # Exit manually
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("Collection complete.")