import os
import cv2
import numpy as np
import tensorflow as tf
from transformers import AutoImageProcessor, TFAutoModelForImageClassification
from PIL import Image

# 1) 프로세서·모델 로드 (이미 ImageNet-21k 체크포인트)
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model     = TFAutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

# 실제 클래스 맵을 확인해 보세요
print("총 클래스 수:", len(model.config.id2label))  
print(f" model.config.id2label  - {model.config.id2label}")

# → 21843 등

def classify_and_annotate(images, output_dir="output_direct"):
    os.makedirs(output_dir, exist_ok=True)

    for img_path in images:
        # PIL로 로드 & 전처리
        img_pil = Image.open(img_path).convert("RGB")
        inputs  = processor(img_pil, return_tensors="tf")

        # 2) 모델 순전파
        outputs = model(**inputs)
        logits  = outputs.logits        # (1, num_classes)
        probs   = tf.nn.softmax(logits, axis=-1).numpy()[0]

        # 3) Top-5 인덱스 및 레이블/확률
        top5_idx   = probs.argsort()[-5:][::-1]
        top5_labels = [model.config.id2label[i] for i in top5_idx]
        top5_scores = [probs[i] for i in top5_idx]

        print(f"\n[{os.path.basename(img_path)}] Top-5 예측:")
        for label, score in zip(top5_labels, top5_scores):
            print(f"  - {label}: {score*100:.2f}%")

        # 4) 어노테이트 & 저장 (파일명에 safe_label 포함)
        best_label = f"{top5_labels[0]} ({top5_scores[0]*100:.1f}%)"
        safe_label = best_label.replace(" ", "_") \
                                .replace("(", "") \
                                .replace(")", "") \
                                .replace("%", "pct")

        img = cv2.imread(img_path)
        cv2.putText(
            img, best_label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA
        )

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f"{base}_{safe_label}.jpg")
        cv2.imwrite(out_path, img)
        print(f"✨ 저장: {out_path}")

if __name__ == "__main__":
    imgs = [
      "testImages/a1.jpeg",
      "testImages/a2.jpeg",
      "testImages/a3.jpeg",
      "testImages/a4.jpeg",
      "testImages/a5.jpeg",
      "testImages/a6.png",
    ]
    classify_and_annotate(imgs)
