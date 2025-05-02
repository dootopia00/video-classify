import os
import cv2
import numpy as np
import pytest
from video_classify import load_model, classify_frame, process_video

def test_load_model():
    model = load_model()
    assert model is not None


def test_classify_frame_returns_label_and_prob():
    model = load_model()
    # 검은 화면 프레임 (64x64) 생성
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    label, prob = classify_frame(model, dummy)
    assert isinstance(label, str)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_process_video_creates_output(tmp_path):
    # 간단한 검은색 비디오 생성
    input_path = tmp_path / "test_input.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(input_path), fourcc, 5.0, (32, 32))
    for _ in range(10):
        out.write(np.zeros((32,32,3), dtype=np.uint8))
    out.release()

    output_path = tmp_path / "test_output.mp4"
    process_video(str(input_path), str(output_path), frame_interval=5)
    assert output_path.exists()
    assert output_path.stat().st_size > 0