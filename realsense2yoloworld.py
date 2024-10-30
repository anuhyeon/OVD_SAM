
import pyrealsense2.pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLOWorld
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLOWorld("yolov8s-world.pt").to(device)
print(f"모델이 실행 중인 장치: {device}")

input_classes = input("추론할 클래스 이름을 쉼표로 구분하여 입력하세요 (예: person,fork,cup): ")
class_list = [cls.strip() for cls in input_classes.split(',')]
model.set_classes(class_list)
print(f"설정된 클래스: {class_list}")

# RealSense 파이프라인 설정
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 320, 240, rs.format.bgr8, 15)  # 성능 최적화를 위한 해상도와 FPS 조정
config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 15) # 현재 15fps

# 카메라 스트리밍 시작
pipeline.start(config)

try:
    while True:
        # 프레임 가져오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # RealSense 프레임을 NumPy 배열로 변환
        frame = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # YOLO 모델에 컬러 프레임 입력
        results = model.predict(frame)

        # 탐지 결과 처리
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 바운딩 박스 좌표와 클래스 정보 가져오기
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                label = f"{model.names[int(cls)]} {conf:.2f}"

                # 바운딩 박스 중심의 Depth 값 가져오기
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                depth_value = depth_frame.get_distance(cx, cy)

                # 바운딩 박스 및 레이블 표시
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} | Depth: {depth_value:.2f}m",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('Real-time Object Detection with RealSense', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 자원 해제
    pipeline.stop()
    cv2.destroyAllWindows()

