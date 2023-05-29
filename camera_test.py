import cv2

cap = cv2.VideoCapture(0) # 0 对应的是默认第一个摄像头

while True: # 使用while循环来从视频流中读取每一帧图像，并对其进行处理
    # ret 布尔型(True或者False),代表有没有读取到图片  frame 表示截取到的一帧的图片的数据，是个三维数组
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 使用cv2.threshold函数对图像进行二值化处理
    ret, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    # 使用cv2.resize函数调整图像大小，使其变为特定尺寸的栅格化图像
    grid_image = cv2.resize(binary_image, (800, 800))
    # grid_image = cv2.resize(frame, (800, 800))
    # 使用cv2.imshow函数将处理后的图像显示出来，并使用cv2.waitKey函数等待指定的毫秒数，以便查看图像
    cv2.imshow('grid_image', grid_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
