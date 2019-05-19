import cv2 as cv

import time



def read_vide():
    video = "http://admin:admin@192.168.31.95:8081"

    cap = cv.VideoCapture(video)


    while(1):
        ret,frame = cap.read()
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
