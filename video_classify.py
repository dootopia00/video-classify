import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

def load_model():
    return MobileNetV2(weights='imagenet')


def classify_frame(model, frame):
    img = cv2.resize(frame, (224, 224))
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    top = decode_predictions(preds, top=1)[0][0]
    return top[1], float(top[2])


def process_video(input_path, output_path, frame_interval=30):
    model = load_model()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_interval == 0:
            label, prob = classify_frame(model, frame)
            text = f"{label} ({prob*100:.1f}%)"
            cv2.putText(frame, text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 255, 0), 2, cv2.LINE_AA)
        out.write(frame)

    cap.release()
    out.release()


if __name__ == "__main__":
    input_path = os.getenv("VIDEO_PATH", "input.mp4")
    output_path = os.getenv("OUTPUT_PATH", "output_annotated.mp4")
    process_video(input_path, output_path)
    print("✅ Processing complete:", output_path)
