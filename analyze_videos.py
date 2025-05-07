#!/usr/bin/env python3
import os
from video_classify import classify_and_annotate  # 이제 import 해도 비디오 열기 안 함

if __name__ == "__main__":
    videos = [
        "/app/videos/test-video-01.MOV",
        "/app/videos/test-video-02.MOV",
    ]
    for vid in videos:
        base = os.path.splitext(os.path.basename(vid))[0]
        out  = f"/app/output/{base}_annotated.mp4"
        classify_and_annotate(vid, out)
