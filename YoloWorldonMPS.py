import torch
from ultralytics import YOLOWorld

# MPS 장치 설정 (Apple Silicon GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS(GPU)를 사용합니다.")
else:
    device = torch.device("cpu")
    print("MPS를 사용할 수 없어 CPU를 사용합니다.")
print('111')
# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-world.pt").to(device)  # 모델을 MPS 장치로 이동
# 모델이 어느 장치에서 실행되고 있는지 확인
device = next(model.parameters()).device
print(f"모델이 실행 중인 장치: {device}")
print('222')
# Define custom classes
model.set_classes(["person","fork","headphone","cup","foot","notebook"])
print('333')
# Execute prediction on an image
# predict 함수가 이미 내부적으로 데이터를 MPS로 전송할 수 있는지 확인 필요
results = model.predict("/Users/an-uhyeon/OVD_SAM/images/starbucks2.jpg")
print('444')
# Show results
results[0].show()
