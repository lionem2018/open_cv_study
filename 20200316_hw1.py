import cv2
import matplotlib.pyplot as plt


def onMouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        plt.subplot(1, 2, 1)
        plt.plot(frame[y, :, 0], 'b')
        plt.plot(frame[y, :, 1], 'g')
        plt.plot(frame[y, :, 2], 'r')

        plt.subplot(1, 2, 2)
        plt.plot(frame[:, x, 0], 'b')
        plt.plot(frame[:, x, 1], 'g')
        plt.plot(frame[:, x, 2], 'r')

        plt.show()


cap = cv2.VideoCapture(0)
if cap.isOpened():
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('camera', frame)
            if cv2.waitKey(1) != -1:
                break
        else:
            print('no frame!')
            break

    cv2.imshow('camera', frame)
    cv2.setMouseCallback('camera', onMouse)
    cv2.waitKey(0)
else:
    print('no camera!')


