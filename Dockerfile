FROM python:3.9-slim-buster
WORKDIR /app

# pipenv 설치
RUN pip install --no-cache-dir pipenv

# 시스템 라이브러리 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      libglib2.0-0 libsm6 libxext6 libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# Pipenv 의존성 설치 (lock 없이 Pipfile 기준)
COPY Pipfile ./
RUN pipenv install --system --skip-lock

# 코드 복사
COPY video_classify.py .
COPY tests ./tests

# 기본 테스트 실행 커맨드
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
