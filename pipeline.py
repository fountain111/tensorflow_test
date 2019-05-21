import os
import logging
import csv
import numpy as np

import cv2

import utils


DIVIDER_COLOUR = (255,255,0)
BOUNDING_BOX_COLOUR = (255,255,0)
CENTROID_COLOUR = (0,0,255)
CAR_COLOURS = [(0,0,255)]
EXIT_COLOR = (66,183,42)


class PipelineProcessor(object):
    '''
        Base class for processors.
    '''

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)


class PipelineRunner(object):
    '''
        just for conveinet

        pipeline 里面存放的是每个processor,processor由外部传入，比如countourdetection就是一个pipeline，

    '''

    def __init__(self,pipeline=None,log_level=logging.DEBUG):
        self.pipeline = pipeline or []
        self.context = {}
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(log_level)
        self.log_level = log_level
        #self.set_log_level()

    def set_context(self,data):
        self.context=data

    def add(self,processor):
        if not isinstance(processor,PipelineProcessor):
            raise Exception(
                'Process should be an instance of PipelineProcess'

            )
        processor.log.setLevel(self.log_level)

    def remove(self,name):
        for i,p in enumerate(self.pipeline):
            if p.__class__.__name__==name:
                del self.pipeline[i]
                return True
        return False

    def set_log_level(self):
        for p in self.pipeline:
            p.log.setLevel(self.log_level)


    def run(self):
        for p in self.pipeline:
            self.context=p(self.context)

        self.log.debug('Frame{frame_number}'.format(frame_number=self.context['frame_number']))

        return self.context




class ContourDetection(PipelineProcessor):

    '''

        Detecting moving object

        1:subtract background
        2:get moving object
        3:filter



    '''


    def __init__(self,bg_subtractor,min_contour_width=35,min_contour_height=35,save_image=False,image_dir='images'):
        super(ContourDetection,self).__init__()

        self.bg_subtractor = bg_subtractor
        self.min_contour_width = min_contour_width
        self.min_contour_height = min_contour_height
        self.save_image = save_image
        self.image_dir = image_dir


    def filter_mask(self,img,a=None):
        '''

            filter
        :param img:
        :param a:
        :return:
        '''

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))

        #fill any small holes
        closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

        #remove noise
        opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)

        #Dilate to merge adjacent blobs
        dilation = cv2.dilate(opening,kernel,iterations=2)

        return dilation


    def detect_vehicles(self,fg_mask,context):

        matches = []

        im2,contours,hierarchy = cv2.findContours(fg_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_TC89_L1)

        for (i,contour) in enumerate(contours):
            (x,y,w,h) = cv2.boundingRect(contour)

            contour_valid = (w>=self.min_contour_width) and (h>=self.min_contour_height)

            if not contour_valid:
                continue


            centroid = utils.get_centroid(x,y,w,h)

            matches.append(
                ((x,y,w,h),centroid)
                           )

        return matches


    def __call__(self, context):
        frame = context['frame'].copy()
        frame_number = context['frame_number']
        fg_mask =self.bg_subtractor.apply(frame,None,0.001)  #函数可以apply

        #thresholding values
        fg_mask = fg_mask[fg_mask<240] = 0
        fg_mask = self.filter_mask(fg_mask,frame_number)  #frame_number应该没用到

        if self.save_image:
            utils.save_frame(fg_mask,self.image_dir+"mask_{frame_number}".format(frame_number=frame_number),flip=False)

        context['objects'] = self.detect_vehicles(fg_mask,context) #object存放的是一个matches，里面包含一个数组，数组里的元素是一个countour的坐标及中心坐标，
        context['fg_mask'] = fg_mask #filter之后的图像.

        return context




class VehicleCOunter(PipelineProcessor):
    '''
        COUNTING vehicles

    '''

    def __init__(self,exit_masks=[],path_size = 10,max_dst=30,x_weight=1.0,y_weight=1.0):
        super(VehicleCOunter,self).__init__()

        self.exit_masks = exit_masks
        self.vehicle_count = 0
        self.path_size = path_size
        self.pathes = []
        self.max_dst = max_dst
        self.x_weight = x_weight
        self.y_weight = y_weight



    def check(self,point):
        for exit_mask in self.exit_masks:
            try:
                if exit_mask[point[1]][point[0]] == 255:
                    return True
            except:
                return True

        return False

    def __call__(self, context):
        objects = context['objects']
        context['exit_masks'] = self.exit_masks
        context['pathes'] = self.pathes
        context['vehicle_count'] = self.vehicle_count

        if not objects:
            return context

        points = np.array(objects)[:,0:2] #搞清楚objects是什么,objects的0：2应该是两个中心坐标x,y ，运行的时候u验证一下,其实似乎没有必要这样写，全部取过来也行，要试一下,
        points = points.tolist() #目的是 把tuple换成List ，但是这样做有必要吗？


        #add new points if pathes is empty

        if not self.pathes:
            for match in points:
                self.pathes.append([match]) #match=[(1, 5, 8, 8), (9, 2)]，外面还要加一个括号,一个元素变成[[(1, 5, 8, 8), (9, 2)]]

        else:
            new_pathes = []

            for path in self.pathes:

                _min = 999999
                _match = None

                for p in points:
                    if len(path) == 1:
                        #distance from last point to current
                        d = utils.distance(p[0],path[-1][0])  #不管如何，放入distance，都只会取x和y,不放心就debug一下，一个个point和某一个path去匹配计算欧式距离
                    else:
                        xn = 2*path[-1][0][0]-path[-2][0][0]
                        yn= 2*path[-1][0][1] - path[-2][0][1]
                        d = utils.distance(p[0],(xn,yn),x_weight=self.x_weight,y_weight=self.y_weight)

                    if d< _min:
                        _min = d
                        _match = p  #包含坐标及中心

                if _match and _min <=self.max_dst:

                    points.remove(_match)
                    path.append(_match) #小于max_dst的加入path，这个特定的path是针对单个跟踪到的汽车，
                    new_pathes.append(path)


                if _match is None:
                    new_pathes.append(path)

            self.pathes = new_pathes


            # add new pathes
            if len(points):
                for p in points:

                    if self.check(p[1])
