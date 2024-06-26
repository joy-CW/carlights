import cv2
import numpy as np

lsPointsChoose = []
tpPointsChoose = []
pointsCount = 0
pointsMax = 50
threshold = 100
key = -1
height = 539
width = 959

img = cv2.imread('/home/pi/python/21.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def binary_threshold(image_path, threshold):
    _, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow("initial Image", img)
    cv2.imshow("Binary Image", binary_img)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    return key, binary_img

def on_mouse(event, x, y, flags, param):

    global img, point1, point2, count, pointsMax
    global lsPointsChoose, tpPointsChoose  # 存入選擇的點
    global pointsCount, pointsMax  # 對鼠標點的計數
    global img2, ROI_bymouse_flag

    img2 = img.copy()  
    if event == cv2.EVENT_LBUTTONDOWN:  # 左鍵點擊
        if pointsCount < pointsMax:
            pointsCount = pointsCount + 1
            print('pointsCount:', pointsCount)
            point1 = (x, y)
            print(x, y)
            # 畫點擊的點
            cv2.circle(img2, point1, 3, (0, 255, 0), 2)
        # 將選取的點保存到list
        lsPointsChoose.append([x, y])  
        tpPointsChoose.append((x, y))  
    
    if pointsCount > 0:
        for i in range(len(tpPointsChoose) - 1):
            cv2.line(img2, tpPointsChoose[i], tpPointsChoose[i + 1], (0, 0, 255), 2)

        cv2.line(img2, tpPointsChoose[0], tpPointsChoose[pointsCount - 1], (0, 0, 255), 2)
    
    if event == cv2.EVENT_RBUTTONDOWN:  # 右鍵點擊
        pointsMax = pointsCount
    # ----------繪圖----------------
    if (pointsCount == pointsMax):
        # -----------繪製ROI區域----------
        ROI_byMouse()
        ROI_bymouse_flag = 1
        lsPointsChoose = []
    cv2.imshow('initial', img2)
    cv2.imshow('src', binary)

def ROI_byMouse():
    global src, ROI, ROI_flag, mask2
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array([lsPointsChoose], np.int32)  # 多邊形頂點列表
    pts = pts.reshape((-1, 1, 2))
    #-------------畫多邊形--------------------
    mask = cv2.polylines(mask, [pts], True, (255, 255, 255))
    #-------------填充多邊形--------------------
    mask2 = cv2.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imwrite('21.bmp', ROI)
    cv2.imshow('ROI', ROI)

while True:
    key, binary = binary_threshold(img, threshold)
    ROI = binary.copy()
    # 按下w鍵，增加亮度門檻值
    if key == ord('w'):
        threshold += 10
        print("Threshold increased to:", threshold)
    # 按下s鍵，減少亮度門檻值
    elif key == ord('s'):
        threshold -= 10
        print("Threshold decreased to:", threshold)
    # 按下q鍵，確定亮度門檻值並開始選取ROI辨識區
    elif key == ord('q'):
        break

cv2.namedWindow('src')
cv2.setMouseCallback('src', on_mouse)
cv2.imshow('src', binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

