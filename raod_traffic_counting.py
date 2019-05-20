import os
import logging
import logging.handlers
import random

import numpy as np
#import skvideo.io
import cv2
import matplotlib.pyplot as plt

import utils


def back_groud_test():
    #background函数测试
    cap = cv2.VideoCapture(0)
    mog = cv2.createBackgroundSubtractorMOG2(history=500,detectShadows=True)

    while True:
        ret,frame = cap.read()
        fg_mask = mog.apply(frame,None,0.001)
        cv2.imshow("frame",fg_mask)
        key = cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def get_centroid(x,y,w,h):

    x1 = int(w/2)
    y1 = int(h/2)

    cx=x+x1
    cy=y+y1
    return cx,cy

def detect_vehicles(fg_mask,min_countour_width=35,min_countour_heigh=35):
    matches = []

    # find external contours
    im,countours,hierarchy = cv2.findContours(
        fg_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_KCOS

    )

    for (i,countour) in enumerate(countours):
        (x,y,w,h) = cv2.boundingRect(countour)
        countour_valid = (x>=min_countour_width) and (h>min_countour_heigh)

        if not countour_valid:
            continue

        #getting center of the bounding box
        centroid = get_centroid(x,y,w,h)

        matches.append((x,y,w,h),centroid)

    return matches




def train_bg_subtractor(inst,cap,num=5000):

    i = 0
    for frame in cap:
        inst.apply(frame,None,0.001)
        i +=1
        if i>=num:
            return cap


class PipelineRunner(object):

    def __init__(self,pipeline=None,log_level=logging.DEBUG):
        self.pipeline = pipeline or []
        self.context = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.set_log_level()

    def set_context(self,data):
        self.context=data

    def add(self,processor):
        if not isinstance(processor,PipelineProcessor):
            raise Exception(
                'Processor should be isinstance of PipelineProcessor'

            )

            processor.log.setLevel(self.log_level)
            self.pipeline.append(processor)

    def remove(self,name):
        for i,p in enumerate(self.pipeline):
            if p.__class__.__name__ ==name:
                del self.pipeline[i]
                return True
        return False

    def set_log_level(self):
        for p in self.pipeline:
            p.log.setLevel(self.log_level)

    def run(self):
        for p in self.pipeline:
            self.context = p(self.context)

        return self.context


class PipelineProcessor(object):
    '''
         Base calss for processors

    '''
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)

class CountourDetection(PipelineProcessor):
    '''

        Detecting moving objects.
        Purpose of this processor is to subtrac background, get moving objects
        and detect them with a cv2.findContours method, and then filter off-by
        width and height.
        bg_subtractor - background subtractor isinstance.
        min_contour_width - min bounding rectangle width.
        min_contour_height - min bounding rectangle height.
        save_image - if True will save detected objects mask to file.
        image_dir - where to save images(must exist).

    '''

    def __init__(self,bg_subtractor,min_contour_width=35,min_countour_height=35,save_image=False,image_dir='images'):
        super(CountourDetection, self).__init__()
        self.bg_subtractor = bg_subtractor
        self.min_contour_width = min_contour_width
        self.min_contour_height = min_countour_height
        self.save_image = save_image
        self.image_dir = image_dir

    def filter_mask(self, img, a=None):
        '''
            This filters are hand-picked just based on visual tests
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

        # Fill any small holes
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # Remove noise
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        # Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening, kernel, iterations=2)

        return dilation

    def detect_vehicles(self, fg_mask, context):

        matches = []

        # finding external contours
        im2, contours, hierarchy = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

        for (i, contour) in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            contour_valid = (w >= self.min_contour_width) and (
                    h >= self.min_contour_height)

            if not contour_valid:
                continue

            centroid = utils.get_centroid(x, y, w, h)

            matches.append(((x, y, w, h), centroid))

        return matches



def main():
    log = logging.getLogger("main")

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500,detectShadows=True
    )
    cap = cv2.VideoCapture(0)

    ret,frame = cap.read()
    fg_mask = bg_subtractor.apply(frame,None,0.001)

    cv2.imshow("mask",fg_mask)
    cv2.imshow("img",frame)
    cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    back_groud_test()