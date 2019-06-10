#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:16:09 2018

@author: ialvarezillan
"""
from skimage.util import montage
import numpy as np
from matplotlib import pyplot as plt

def bmp_stack(image, axis=1):
    if len(image.shape)==4:
        plt.figure()
        if axis==1:
            imager=image[:,:,int(image.shape[2]/2),:]
        elif axis==2:
            imager=image[:,int(image.shape[1]/2),:,:]
        elif axis==3:
            imager=image[:,:,:,int(image.shape[3]/2)]
        plt.imshow(montage(imager),cmap='gray')
        plt.colorbar()
    else:
        plt.figure()
        if axis==1:
            plt.imshow(montage(np.transpose(image, (2,0,1))))#,cmap='gray')
        elif axis==2:
            plt.imshow(montage(image))#,cmap='gray')
        elif axis==3:
            plt.imshow(montage(np.transpose(image, (1,2,0))))#,cmap='gray')
        plt.colorbar()