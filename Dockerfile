# 1. Python 3.10이 설치된 베이스 이미지 선택
FROM python:3.10-slim-buster

# 2. 필요한 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive

# 3. 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    git cmake build-essential libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev \
    wget curl \
    && rm -rf /var/lib/apt/lists/*

# 4. 작업 디렉터리 설정
WORKDIR /app

# 5. 호스트의 파일들을 컨테이너로 복사
COPY .gitignore /app/
COPY requirements.txt /app/
COPY realsense2yoloworld.py /app/

# 6. Python 의존성 설치
RUN pip install -r requirements.txt



