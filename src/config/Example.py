# Copy this file to MainConfig.py and edit for your own paths
from NN.metrics import *
from src.constants.readMode import ReadMode
from src.constants.visualizationModes import VizualizationModes
from src.constants.preprocMode import PreprocMode
from keras.metrics import mse
from keras.losses import mean_squared_error

_input_data = '/media/osz1/DATA/Dropbox/MyProjects/UNAM/Norths_Detection_DL/ExampleData/YEARS'

def getMapVisualizationConfig():
    '''These method creates configuration parameters for making visualizations'''
    cur_config = {
        'input_folder': _input_data,
        'output_folder': '/media/osz1/DATA_Old/NORTES/Maps',
        'mode': VizualizationModes.OZVIZ,
        'years': range(1979,1982), # It can be a range
        'display': True, # Indicate if we want to plot the results
        'save_figures': True, # Indicate if we want to save the images
        'variables': ['u','v']
    }

    return cur_config

# def getPreprocessingConfig():
#     '''This method creates a configuration for the preprocessing'''
#     cur_config = {
#         'mode': PreprocMode.MASK_AND_BBOX,
#         'input_folder': '/media/osz1/DATA/Dropbox/MyProjects/UNAM/Norths_Detection_DL/ExampleData',
#         'input_folder': '/media/osz1/DATA/Dropbox/MyProjects/UNAM/Norths_Detection_DL/ExampleData',
#         # 'output_folder': '/media/osz1/DATA/Dropbox/UMIAMI/WorkUM/KidneySegmentation_MICCAI/Kits19_Ours/info',
#         'output_folder': '/media/osz1/DATA/DATA/MICCAI_Kidney_CT/preproc',
#         'crop_size_cm': [17.96, 30.08, 36.6], # (NOT USED FOR NOW) Only for 'preproc' The images will be cropped to this size, starting from middle
#         # 'resample': [2.5, 0.9, 0.9],
#         'resample': [2.0, 2, 2], # Resampling size of the image. The final image should have more less the dimensions below
#         'crop_size_slices': [168, 168, 168], # Final resolution of the images (related with the 'resample' parameter)
#         'display_images': False # Indicates if we want to see intermediate images displayed
#         }
#
#     return cur_config

def getTrainingConfig():
    cur_config = {
        'input_folder':_input_data,
        'norths_file': '/media/osz1/DATA/Dropbox/MyProjects/UNAM/Norths_Detection_DL/ExampleData/nortes.csv',
        'output_folder': '/media/osz1/DATA/Dropbox/MyProjects/UNAM/Norths_Detection_DL/Output',
        'val_perc': .1,
        'test_perc': 0,
        'eval_metrics': [mse],  # Metrics to use in the training
        'loss_func': mean_squared_error,  # Loss function to use for the learning
        'variables': ['u','v'],
        'batch_size':  1,
        'epochs':  10000,
        'useDA':  False # Use data augmentation
    }

    return cur_config
