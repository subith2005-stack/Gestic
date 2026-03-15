from fastapi import FastAPI, UploadFile, File # type: ignore
import numpy as np # type: ignore
import cv2 # type: ignore
import tensorflow as tf # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore

app = FastAPI()

# Allow React to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = tf.keras.models.load_model("v2/cnn_model/saved_model/gesture_cnn.keras")

class_names = ['hello', 'no', 'stop', 'thankyou', 'yes']

IMG_SIZE = 128


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)

    confidence = float(np.max(predictions))
    class_index = int(np.argmax(predictions))

    gesture = class_names[class_index]

    return {
        "gesture": gesture,
        "confidence": confidence
    }