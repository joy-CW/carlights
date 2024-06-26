import cv2

# current camera
cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_V4L)
#cap = cv2.VideoCapture('/home/pi/python/0601_video.mp4')
play = True
cnt = 0
print(cv2.__file__)

while (cap.isOpened()):
    ret, frame = cap.read()
    while(not ret):
        ret, frame = cap.read()
        #print("false")
    if not play: 
        while True:
            cmd = cv2.waitKey(1)
            if cmd == ord('w'):
                play = True
                break
            elif cmd == ord('e'):
                cv2.imwrite(str(cnt) + '.png', frame)
                cnt += 1
                break

            else:
                pass

    if ret == True:
        cv2.imshow('frame', frame)
        if play:
            key = cv2.waitKey(1)
			
        if key == ord("q") or key == ord("Q"):
            break;
        elif key == ord('w'):
            play = False
        else:
            pass

    else:
        break

cap.release()
cv2.destroyAllWindows()

