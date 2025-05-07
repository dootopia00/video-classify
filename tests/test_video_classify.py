import pytest
from video_classify import model

def test_model_loaded():
    # TensorFlow MobileNetV2 모델이 로드되었는지 확인
    assert model is not None
    # 최소 하나의 레이어가 있는지 체크
    assert len(model.layers) > 0
