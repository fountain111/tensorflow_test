import tensorflow as tf

import numpy as np
print(tf.__version__)
import cv2
import numpy

#import matplotlib.pyplot as plot

#cap = cv2.VideoCapture(0)

#print(np.zeros((720,1280)+(3,),dtype='uint8'))
#while(1):
    # get a frame
 #   ret, frame = cap.read()
    # show a frame
  #  cv2.imshow("capture", frame)
   # if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

def centro_id():

    cx1 = np.random.randint(1,10)
    cx2 = np.random.randint(1,10)
    return (cx1,cx2)
matches = []

for i in range(10):
    (x,y,w,h) =  (np.random.randint(1,10) for j in range(4))

    centrid= centro_id()


    matches.append(
        ((x,y,w,h),centrid)

    )

points = np.array(matches)[:,0:2]
#points1 = np.array(matches)

points2 = points.tolist()

print(matches)
#print(points1)
print(points2)

paths=[]
for math in points2:
    paths.append([math])

for path in paths:

    for p in points2:
        if len(path)==1:
            print(p[1])
            pass
        else:
            print('len=2')


