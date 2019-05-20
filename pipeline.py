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
        for i,p in


