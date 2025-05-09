# video_classify.py

import os
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

# 1) 모델 로드 (import 시 한 번만 실행)
model = MobileNetV2(weights='imagenet')
print("✅ MobileNetV2 모델 로드 완료")


def classify_and_annotate(
    video_path: str,
    output_path: str,
    frame_interval: int = 30
):
    """
    주어진 비디오 파일을 열어 frame_interval 프레임마다
    MobileNetV2로 분류된 라벨을 오버레이하고,
    스크린샷과 어노테이트된 영상을 저장합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")
    
    
    # 비디오 속성
    fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 스크린샷 저장 폴더 준비
    screenshot_dir = os.path.join(os.path.dirname(output_path), "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)
    
    base_name   = os.path.splitext(os.path.basename(video_path))[0]
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # 지정 간격마다 분류 및 스크린샷 저장
        if frame_count % frame_interval == 0:
            # 1) 모델 예측
            img   = cv2.resize(frame, (224, 224))
            x     = preprocess_input(np.expand_dims(img.astype('float32'), axis=0))
            preds = model.predict(x)
            top   = decode_predictions(preds, top=1)[0][0]

            # print(f"[top]: {top}");
            # return;

            label = f"{top[1]} ({top[2]*100:.1f}%)"
            print(f"[{base_name}:Frame count: {frame_count}] / label: {label}")

            # 2) 프레임에 라벨 오버레이
            cv2.putText(
                frame, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0, 255, 0), 2, cv2.LINE_AA
            )

            # 3) 스크린샷 저장
            #    파일명에 들어갈 수 없는 문자는 제거·치환
            safe_label = (
                label
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("%", "pct")
            )
            filename = f"{base_name}_frame{frame_count}_{safe_label}.jpg"
            path     = os.path.join(screenshot_dir, filename)
            success  = cv2.imwrite(path, frame)
            if success:
                print(f"✨ Saved screenshot: {path}")
            else:
                print(f"⚠️ Failed to save screenshot: {path}")

        # 어노테이트된 프레임을 비디오에 기록
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ 완료: {video_path} → {output_path}")


if __name__ == "__main__":
    # 직접 실행 시, 환경변수로 경로 지정 가능
    video_path  = os.getenv("VIDEO_PATH", "input.mp4")
    output_path = os.getenv("OUTPUT_PATH", "output_annotated.mp4")
    classify_and_annotate(video_path, output_path)
