# Copy this file to MainConfig.py and edit for your own paths
from NN.metrics import *
from src.constants.planeTypes import PlaneTypes
from src.constants.visualizationModes import VizualizationModes
from src.constants.preprocMode import PreprocMode


def getImageContourVisualizer():
    '''These method creates configuration parameters for making visualizations'''
    cur_config = {
        'input_folder': './../data/input',
        'output_folder': './../data/output',
        'mode': VizualizationModes.OZVIZ,
        # If visualizing PREPROC DATA
        # 'input_folder': './../data/preproc',
        # 'mode': VizualizationModes.PREPROC,
        'years': range(209),
        'display': True,  # Indicate if we want to plot the results
        'save_figures': True,  # Indicate if we want to save the images
        # Indicate which slices we want to plot 'all', 'middle', range(30,50)
        'slices': range(100, 400, 10),
        'plane': PlaneTypes.AXIAL,  # Which plane we want to plot 'sag', 'ax', 'cor'
        # Indicate if we want to save only images that have some segmentation,
        'only_contours': True,
        # Github credentials required to download file from github under secure connection.
        'github_username': '---',
        'github_pass': '---',
        'github_data_url': 'https://github.com/neheller/kits19/blob/master/data'
    }

    return cur_config


def getPreprocessingConfig():
    '''This method creates a configuration for the preprocessing'''
    cur_config = {
        'mode': PreprocMode.ALL,  # Mode can be 'summary', 'mask_bbox', 'preproc'
        'input_folder': '../yourinput',
        'output_folder': '../outputfolder',
        'output_image_folder': '../outimagefolder',
        'crop_size_cm': [17.96, 30.08, 36.6], # (NOT USED FOR NOW) Only for 'preproc' The images will be cropped to this size, starting from middle
        'resample': [2.0, 2, 2], # Resampling size of the image. The final image should have more less the dimensions below
        'crop_size_slices': [168, 168, 168], # Final resolution of the images (related with the 'resample' parameter)
        'display_images': False # Indicates if we want to see intermediate images displayed
        }

    return cur_config

def getTrainingConfig():
    cur_config = {
        'input_folder': '../yourinput',
        'output_folder': '../youroutput',
        'tot_examples': 209, # NOT SURE HOW TO MOVE THIS AUTOMATIC
        'val_perc': .1,
        'test_perc': 0,
        'img_dims': [168, 168, 168],
        'eval_metrics': [real_dice_coef],  # Metrics to use in the training
        'loss_func': dice_coef_loss,  # Loss function to use for the learning
        'batch_size':  2,
        'epochs':  1000,
        'useDA':  False # Use data augmentation
    }

    return cur_config

def getClassifyConfig():
    # Name your run or classification
    _run_name= 'TodayRun'

    all_params = {
        # Where the years are. Enumerated and in .nrrd format
        'input_folder': '.../preproc', # Where the Cases are
        'output_folder': F'...outputfolder/{_run_name}', # Where to save segmentations
        'output_images': F'...outputimages/{_run_name}', # Where to save images from the segmentation
        'weights': '.....hdf5', # File name with the weights of the model
        'model_name': '3dm', # Name of the model (not used so far)
        'display_images': False, # Indicates if we should display the images or only save them
        'roi_slices': np.arange(0,160,5),  # Which slices to generate ('all', 'middle', or an array of numbers)
        # 'roi_slices': SliceMode.MIDDLE,
        'save_segmentations': False, # Indicates if the segmentations files are going to be saved (normally yes)
        'years': np.arange(0,208) # Which years to compute
    }

    return all_params
