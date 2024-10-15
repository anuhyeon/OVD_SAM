import cv2

# 카메라 초기화
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("프레임을 읽어올 수 없습니다.")
        break

    # 프레임 출력
    cv2.imshow("Camera Test", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
