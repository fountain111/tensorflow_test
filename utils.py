import cv2
import logging
import logging.handlers
import math
import sys
import numpy as np

def save_frame(frame,file_name,flip=True):

    #flip BGR TO RGB
    if flip:
        cv2.imwrite(file_name,np.flip(frame,2))
    else:
        cv2.imwrite(file_name,frame)



def init_logging(to_file=False):
    main_logger = logging.getLogger()

    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d %(levelname)-8s [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler_stream =logging.StreamHandler(sys.stdout)
    handler_stream.setFormatter(formatter)
    main_logger.addHandler(handler_stream)

    if to_file:
        handler_file = logging.handlers.RotatingFileHandler("debug.log",maxBytes=1024*11024*400,backupCount=10) # 400 MB
        handler_file.setFormatter(fmt=formatter)
        main_logger.addHandler(handler_stream)

    main_logger.setLevel(logging.DEBUG)
    return main_logger
#=====================================================================

def distance(x,y,type='euclidean',x_weight=1.0,y_weight=1.0):
    # 欧式距离
    if type =='euclidean':
        return math.sqrt(float((x[0]-y[0])**2)/x_weight) + float( (x[1]-y[1])**2/y_weight )


def get_centroid(x,y,w,h):
    #获取中心

    x1 = int(w/2)
    y1 = int(h/2)

    cx = x+x1
    cy = y+y1

    return (cx,cy)


def skeleton(img):
    #这个函数有四个参数，第一个原图像，第二个进行分类的阈值，第三个是高于（低于）阈值时赋予的新值，第四个是一个方法选择参数.
    ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)) #理解为卷积核
    done = False
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)


    while (not done):
        eroded = cv2.erode(img, element) # 腐蚀
        temp = cv2.dilate(eroded, element) # 膨胀
        temp = cv2.subtract(img, temp) #
        skel = cv2.bitwise_or(skel, temp) # or操作 得到skel
        img = eroded.copy() #腐蚀过的img
        zeros = size - cv2.countNonZero(img)  # 非0像素点数
        if zeros == size:
            done = True


    return skel



