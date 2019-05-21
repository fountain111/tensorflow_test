import cv2 as cv

import time

import  numpy as np



def read_vide():
    #SHAPE = (720,1280)
    EXIT_PTS = np.array([
        [[732,720],[732,590],[1280,520],[1280,720]],
        [[0,400],[645,400],[645,0],[0,0]]
    ])

    EXIT_PTS1 = np.array([
        [[700, 550],[950, 550], [950, 460],[700, 460]]
    ])
    video = "http://admin:admin@192.168.31.95:8081"

    cap = cv.VideoCapture(video)

    bg_subtractor = cv.createBackgroundSubtractorMOG2(history=500,detectShadows=True)

    #base = np.zeros((SHAPE+(3,)),dtype='uint8'

    #)

    #cv.imshow("ig",exit_mask)
    cv.waitKey(0)




    while(1):
        ret,frame = cap.read()
        frame = cv.fillPoly(frame, EXIT_PTS1, (255, 255, 255))  # 245,255,250
        print(frame.shape)
        cv.imshow("capture",frame)
        if cv.waitKey(1) & 0xFF ==ord('q'):
            break


def read_img():
    img_path = "/home/gg/Downloads/VOCdevkit/VOC2007/JPEGImages/000001.jpg"  #要用绝对路径，不能用相对路径

    img = cv.imread(img_path,0)

    cv.namedWindow("image",cv.WINDOW_NORMAL)
    cv.imshow('image',img)
    cv.waitKey(0)

    cv.imwrite('test.png',img)




    cv.destroyAllWindows()

if __name__ == '__main__':
    read_vide()