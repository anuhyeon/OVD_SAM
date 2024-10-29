import cv2
import torch
from ultralytics import YOLOWorld

# 모델 로드 및 장치 설정
model = YOLOWorld("yolov8s-world.pt")
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model.to(device)  # 모델을 MPS 또는 CPU로 이동

# 모델이 어느 장치에서 실행되고 있는지 확인
print(f"모델이 실행 중인 장치: {device}")

# 사용자로부터 추론할 클래스 입력 받기
input_classes = input("추론할 클래스 이름을 쉼표로 구분하여 입력하세요 (예: person,fork,cup): ")

# 입력받은 문자열을 쉼표로 나눠서 리스트로 변환
class_list = [cls.strip() for cls in input_classes.split(',')]

# YOLO 모델에 설정
model.set_classes(class_list)

print(f"설정된 클래스: {class_list}")


# 카메라 초기화
cap = cv2.VideoCapture(1)  # '0'은 기본 카메라를 의미합니다.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 모델 예측 (OpenCV BGR 이미지를 그대로 전달)
    results = model.predict(frame)

    # 결과 후처리 및 시각화
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

    # 결과 프레임 표시
    cv2.imshow('Real-time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
