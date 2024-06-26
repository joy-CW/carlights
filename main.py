import cv2
import numpy as np
import time

frame_width = 640
frame_height = 480

image_low = cv2.imread('ROI_low.bmp')
image_low = cv2.resize(image_low, (frame_width, frame_height))
image_high = cv2.imread('ROI_high.bmp')
image_high = cv2.resize(image_high, (frame_width, frame_height))

initial_image_low = cv2.imread('low.png')
initial_image_low = cv2.resize(image_low, (frame_width, frame_height))
initial_image_high = cv2.imread('high.png')
initial_image_high = cv2.resize(image_high, (frame_width, frame_height))

gray_image_low = cv2.cvtColor(image_low, cv2.COLOR_BGR2GRAY)
mask_low = np.zeros((480, 640, 3), dtype=np.uint8)
mask_low[gray_image_low > 0] = [255, 255, 255]

gray_image_high = cv2.cvtColor(image_high, cv2.COLOR_BGR2GRAY)
mask_high = np.zeros((480, 640, 3), dtype=np.uint8)
mask_high[gray_image_high > 0] = [255, 255, 255]

kernel = np.ones((16,16),np.uint8)
block_size = 16
threshold = 30
light = 0
nohigh = False
nohigh_change = nohigh
red = False
last_switch_time = time.time()
switch_delay = 0.5

cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_V4L)
# cap = cv2.VideoCapture('video.mp4')
result = cv2.VideoWriter('result_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    key = cv2.waitKey(1)

    if key == ord('w'):
        mask_high = cv2.dilate(mask_high, kernel, iterations=1)
    elif key == ord('s'):
        mask_high = cv2.erode(mask_high, kernel, iterations=1)

    gray_low = cv2.bitwise_and(mask_low, initial_image_low)
    gray_high = cv2.bitwise_and(mask_high, initial_image_high)
    
    frame = cv2.resize(frame, (frame_width, frame_height))
    f = frame.copy()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for y in range(0, frame_height, block_size):
        for x in range(0, frame_width, block_size):
            block = gray_frame[y:y + block_size, x:x + block_size]
            block_brightness = np.mean(block)

            block_low = gray_low[y:y + block_size, x:x + block_size]
            block_brightness_low = np.mean(block_low)

            block_high = gray_high[y:y + block_size, x:x + block_size]
            block_brightness_high = np.mean(block_high)

            if block_brightness_high > threshold:
                frame[y:y + block_size, x:x + block_size] = [0, 0, 50]
                if (block_brightness - block_brightness_high) > 180:
                    light += 1
                    frame[y:y + block_size, x:x + block_size] = [0, 255, 0]
            elif block_brightness_low > threshold:
                frame[y:y + block_size, x:x + block_size] = [0, 50, 50]

    if light > 0:
        nohigh = True
    else:
        nohigh = False

    if nohigh_change != nohigh:
        last_switch_time = time.time()
        nohigh_change = nohigh

    if time.time() - last_switch_time > switch_delay:
        if nohigh == True:
            red = True
        else:
            red = False

    if red:
        f[0:80, 0:80] = [0, 0, 255]
    else:
        f[0:80, 0:80] = [0, 255, 0]

    light = 0

    cv2.imshow('Marked Frame', frame)
    # cv2.imshow('high',image_high)
    cv2.imshow('video', f)
    # cv2.imshow('mask',mask_high)
    # cv2.imshow('gray', gray_high)
    # cv2.imshow('initial', initial_image_high)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    result.write(f)

cap.release()
cv2.destroyAllWindows()



