# Plotting utils

import glob
import math
import os
import random
from copy import copy
from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from scipy.signal import butter, filtfilt

from utils.general import xywh2xyxy, xyxy2xywh
from utils.metrics import fitness

# Settings
matplotlib.use('Agg')  # for writing to files only




def plot_one_box(x, img, color=None, label=None, line_thickness=None, counter=0):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]    
    
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))    
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.circle(img, (550, 350), 2, (255, 255,0), thickness=4)
    cv2.circle(img, (750, 178), 2, (255, 255,0), thickness=4)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        cv2.putText(img, str(counter), (200, 100), 0, tl / 3, [225, 0, 255], thickness=tf, lineType=cv2.LINE_AA)

    x1, y1 = c1
    x2, y2 = c2    
    xc, yc = (int((x2+x1)/ 2), int((y2+y2)/2))
    print(counter)
    
    if yc > 178 and yc < 350:
    	if yc == (823 - (0.86*xc)):
            counter += 1


    		

    				
    return counter
    		
    
    	

