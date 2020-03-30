import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

blk_size = 9        # 블럭 사이즈
C = 5               # 차감 상수
img = cv2.imread('sudoku.png', cv2.IMREAD_GRAYSCALE) # 그레이 스케일로  읽기

# ---① 오츠의 알고리즘으로 단일 경계 값을 전체 이미지에 적용
ret, th1 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# ---② 어뎁티드 쓰레시홀드를 평균과 가우시안 분포로 각각 적용
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                                      cv2.THRESH_BINARY, blk_size, C)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                     cv2.THRESH_BINARY, blk_size, C)

###########################################################


def thresholdIntegral(inputMat,s,T = 0.15):
    # outputMat=np.uint8(np.ones(inputMat.shape)*255)
    outputMat=np.full(inputMat.shape, 255)
    nRows = inputMat.shape[0]
    nCols = inputMat.shape[1]
    S = int(max(nRows, nCols) / 8)

    s2 = int(S / 4)

    for i in range(nRows):
        y1 = i - s2
        y2 = i + s2

        if (y1 < 0) :
            y1 = 0
        if (y2 >= nRows):
            y2 = nRows - 1

        for j in range(nCols):
            x1 = j - s2
            x2 = j + s2

            if (x1 < 0) :
                x1 = 0
            if (x2 >= nCols):
                x2 = nCols - 1
            count = (x2 - x1)*(y2 - y1)

            sum=s[y2][x2]-s[y2][x1]-s[y1][x2]+s[y1][x1]

            if ((int)(inputMat[i][j] * count) < (int)(sum*(1.0 - T))):
                outputMat[i][j] = 0
                # print(i,j)
            # else:
            #     outputMat[j][i] = 0
    return outputMat


if __name__ == '__main__':
    ratio=1
    image = cv2.imdecode(np.fromfile('sudoku.png', dtype=np.uint8), 0)
    img = cv2.resize(image, (int(image.shape[1] / ratio), int(image.shape[0] / ratio)), cv2.INTER_NEAREST)

    # thresh = cv2.adaptiveThreshold(img,  255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
    # retval, thresh = cv2.threshold(img, 150, 255, cv2.THRESH_OTSU)
    # retval, thresh = cv2.threshold(img, retval, 255, cv2.THRESH_OTSU)

    time_start = time.time()
    roii = cv2.integral(img)
    time_end = time.time()
    print('integral cost', time_end - time_start)

    # time_start = time.time()
    thresh = 0
    for j in range(1):
        thresh = thresholdIntegral(img, roii)
    time_end = time.time()
    print('totally cost', time_end - time_start)
    # cv2.namedWindow('integral threshold',0)
    # cv2.imshow('integral threshold',thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


##############################

# ---③ 결과를 Matplot으로 출력
imgs = {'Original': img, 'Global-Otsu:%d'%ret:th1, \
        'Adapted-Mean':th2, 'Adapted-Gaussian': th3, 'Adapted-Integral': thresh}

plt.figure(figsize=(10, 30))

for i, (k, v) in enumerate(imgs.items()):
    plt.subplot(3,2,i+1)
    plt.title(k)
    plt.imshow(v,'gray')
    plt.xticks([]),plt.yticks([])

plt.show()
