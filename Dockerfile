FROM python:3.9-slim-buster

WORKDIR /app

# 시스템 라이브러리, ffmpeg, pipenv 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libglib2.0-0 libsm6 libxext6 libxrender-dev ffmpeg && \
    pip install --no-cache-dir pipenv && \
    rm -rf /var/lib/apt/lists/*

# Pipenv 의존성 설치 (lock 없이)
COPY Pipfile .
# 기본 + dev 패키지 모두 설치 (--dev 옵션 추가)
RUN pipenv install --ignore-pipfile --skip-lock --dev

# 한 줄로 MobileNetV2 가중치만 빌드 타임에 내려받기
RUN pipenv run python -c "from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2; MobileNetV2(weights='imagenet')"

# 애플리케이션 코드 복사
COPY . .

# 기본 커맨드
CMD ["pipenv", "run", "python", "video_classify.py"]
