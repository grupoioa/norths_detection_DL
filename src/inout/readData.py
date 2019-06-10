from pathlib import Path
import numpy as np
import SimpleITK as sitk
from os.path import join
from pandas import DataFrame
import nibabel as nib
import os
from netCDF4 import Dataset
from clint.textui import progress
from constants.readMode import ReadMode

import requests

def readDataTraining( files_to_read, variables):
    '''
    This funciton will read all the files and variables in an array of size [len(files_to_read),len(variables)]
    :param files_to_read:
    :param variables:
    :return:
    '''
    rows = -1
    cols = -1

    for file_idx, c_file in enumerate(files_to_read):
        netcdf_ds = Dataset(c_file, 'r', format='NETCDF4')

        allvars = netcdf_ds.variables  # Gets the variables inside the netcdf
        # TODO validate the desired variables are in the file
        for var_idx, cur_var_name in enumerate(variables):
            c_var = allvars.get(cur_var_name)  # To access a specific variable
            # print(c_var_np.get_dims())
            c_var_np = c_var[0]
            if rows == -1: # We haven't create the container
                rows, cols = c_var_np.shape
                data = np.zeros((len(files_to_read), len(variables), rows, cols))

            data[file_idx,var_idx,:] = c_var_np

        # Do not forge the close the file handler at the end
        netcdf_ds.close()

    # print('Done!')
    return data


def getDateFromFile(c_file):
    ''' THis is the one that gets the dates from the file names'''
    # TODO made them proper dates rather than strings
    return F"{c_file.split('_')[3]}T{c_file.split('_')[4].split('.')[0]}"

def getDatesFromYears(data_folder, years):
    dates = []
    for cur_year in years:
        cur_folder = join(data_folder,F'a{cur_year}','salidas')
        files = os.listdir(cur_folder)
        # c_files_dates = [getDateFromFile(c_file) for c_file in files]
        for c_file in files:
            dates.append( getDateFromFile(c_file) )

    dates.sort()
    return np.array(dates)

def getFileFromDate(data_folder, c_date):
    date_str = c_date.split('T')[0]
    c_year = date_str.split('-')[0]
    time_str = c_date.split('T')[1]
    file =  F'wrfout_c1h_d01_{date_str}_{time_str}.a{c_year}.nc'
    all_file = join(data_folder,F'a{c_year}','salidas',file)
    if os.path.exists(all_file):
        return all_file
    else:
        return -1

def readAllYears(data_folder):
    files = os.listdir(data_folder)
    years = [y.replace('a','') for y in files ]
    return years


def load_years_np(data_folder, years=ReadMode.ALL_YEARS, variables=['u','v']):
    """Reads specific year and variables in a dictionary of numpy arrays """

    # Init Dictionary
    if years == ReadMode.ALL_YEARS:
        years = readAllYears(data_folder)

    all_dates = getDatesFromYears(data_folder, years)
    data = {str(c_date):{str(cur_var):[] for cur_var in variables} for c_date in all_dates}

    print(F'Reading years: {years}')
    print(F'Reading dates: {all_dates}')
    for c_date in all_dates:
        c_file = getFileFromDate(data_folder, c_date)
        print(F'\tReading file: {c_file}')
        netcdf_ds = Dataset(c_file, 'r', format='NETCDF4')
    #         # TODO check the file contains the variable
        allvars = netcdf_ds.variables
        for cur_var_name in variables:
            c_var =  allvars.get(cur_var_name)  # To access a specific variable
            # print(c_var_np.get_dims())
            # TODO assuming that it is always 2D this may acuse a problem
            c_var_np = c_var[0]
            data[c_date][cur_var_name] = c_var_np

        # Do not forge the close the file handler at the end
        netcdf_ds.close()

    print('Done!')
    return data
