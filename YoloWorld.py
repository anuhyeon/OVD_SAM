import cv2
import torch
from ultralytics import YOLOWorld

model = YOLOWorld("yolov8s-world.pt")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)  # 모델을 MPS 또는 CPU로 이동

print(f"모델이 실행 중인 장치: {device}")

input_classes = input("추론할 클래스 이름을 쉼표로 구분하여 입력하세요 (예: person,fork,cup): ")

class_list = [cls.strip() for cls in input_classes.split(',')]

# YOLO 모델에 설정
model.set_classes(class_list)

print(f"설정된 클래스: {class_list}")

cap = cv2.VideoCapture(1)  # '0'은 기본 카메라

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #(OpenCV BGR 이미지를 그대로 전달)
    results = model.predict(frame)

    for result in results:
        boxes = result.boxes  # 바운딩 박스 정보
        for box in boxes:
            # 바운딩 박스 좌표 가져오기
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]  # 신뢰도
            cls = box.cls[0]  # 클래스 정보
            label = f"{model.names[int(cls)]} {conf:.2f}"
            
            # 바운딩 박스 및 레이블 표시
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('Real-time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
