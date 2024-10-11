from ultralytics import YOLOWorld
import cv2
# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-world.pt")
# 모델이 어느 장치에서 실행되고 있는지 확인
device = next(model.parameters()).device
print(f"모델이 실행 중인 장치: {device}")
# Define custom classes
model.set_classes(["person","fork","headphone","cup","foot","notebook"])
# Execute prediction on an image
results = model.predict("/Users/an-uhyeon/OVD_SAM/images/starbucks2.jpg")
# Show results
results[0].show()
# 처리 속도는 전처리 9.4ms, 추론 175.5ms, 후처리 4.5ms가 소요
