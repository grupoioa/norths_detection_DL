import SimpleITK as sitk
import visualization.utilsviz as utilsviz
import math
import os
from os.path import join
import cv2
from skimage import measure
from scipy.ndimage.morphology import binary_fill_holes
from pandas import DataFrame
from inout.readData import *
from visualization.utilsviz import *
import pandas as pd
import numpy as np
from scipy import interpolate


def normalizeIntensitiesPercentile(imgs, minper=1, maxper=99):
    '''
    Normalizes an array of images (independently) between 0 and 1 using percentiles
    :param imgs: array of itk images
    :param minper: int def 1 Percentile to be used as 0
    :param maxper:  int def 99 Percentile to be used as 1
    :return:
    '''
    out = []

    normalizationFilter = sitk.IntensityWindowingImageFilter()

    # This part normalizes the images
    for idx in range(len(imgs)):
        img = imgs[idx]
        array = np.ndarray.flatten(sitk.GetArrayFromImage(img))

        # Gets the value of the specified percentiles
        upperPerc = np.percentile(array, maxper) #98
        lowerPerc = np.percentile(array, minper) #2

        normalizationFilter.SetOutputMaximum(1.0)
        normalizationFilter.SetOutputMinimum(0.0)
        normalizationFilter.SetWindowMaximum(upperPerc)
        normalizationFilter.SetWindowMinimum(lowerPerc)

        floatImg= sitk.Cast(img, sitk.sitkFloat32) # Cast to float

        # ALL images get normalized between 0 and 1
        outNormalization = normalizationFilter.Execute(floatImg) #Normalize to 0-1
        out.append(outNormalization)

        # If you want to see the differences before and after normalization
        # utilsviz.drawSeriesItk(floatImg, slices=[90], title='', contours=[], savefig='', labels=[])
        # utilsviz.drawSeriesItk(out[-1], slices=[90], title='', contours=[], savefig='', labels=[])
    return out


def interpolateNumpyArray(data, new_dims):
    '''
    Interpolates data from origin dims to new dims
    :param data:
    :param orig_dims:
    :param new_dims:
    :return:
    '''
    # print('Interpolating....')
    examples, variables, rows, cols = data.shape
    new_rows, new_cols = new_dims
    new_data = np.zeros((examples, variables, new_rows, new_cols))

    x = np.linspace(0,10,cols)
    y = np.linspace(0,10,rows)
    x_new = np.linspace(0,10,new_cols)
    y_new = np.linspace(0,10,new_rows)

    for c_example in range(examples):
        for c_var in range(variables):
            c_map = data[c_example][c_var]
            f = interpolate.interp2d(x,y,c_map, kind='cubic')
            new_map = f(x_new, y_new)
            # utilsviz.drawSingleMap(new_map)

            new_data[c_example, c_var,:,:] = new_map
    # print('Done!')

    return new_data
