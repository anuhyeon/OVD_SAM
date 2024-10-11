import torch

# MPS 장치 설정 (Apple Silicon의 GPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS(GPU)를 사용합니다.")
else:
    device = torch.device("cpu")
    print("MPS를 사용할 수 없어 CPU를 사용합니다.")

# 텐서를 GPU(MPS)로 이동
x = torch.randn(3, 3).to(device)
print(x)
