import preproc.preprocImages as utils
import visualization.utilsviz as utilsviz
import NN.models as models
from NN.metrics import *
import os
from pandas import DataFrame
import pandas as pd
from config.MainConfig import getClassifyConfig
from inout.readData import *
from os.path import getmtime, join
import time
import SimpleITK as sitk
from inout.readData import *
from src.constants.readMode import ReadMode
from src.constants.planeTypes import PlaneTypes

def getLatestFile(file_names):
    latest_file = ''
    largest_date = -1
    for cur_file in file_names:
        cur_time = getmtime(cur_file)
        if cur_time > largest_date:
            largest_date = cur_time
            latest_file = cur_file

    return latest_file

def makePrediction(model, roi_img, roi_ctr, img_dims=168):

    ctr_np = sitk.GetArrayFromImage(roi_ctr)
    img_np = sitk.GetArrayFromImage(roi_img)

    X = np.zeros((1,img_dims,img_dims,img_dims,1),dtype=np.float32)
    Y = np.zeros((1,img_dims,img_dims,img_dims,1),dtype=np.float32)

    X[0,:,:,:,0] = img_np
    Y[0,:,:,:,0] = ctr_np

    # utilsviz.draw_multiple_series_numpy(X[0,:,:,:,0], Y[0,:,:,:,0], slices=SliceMode.MIDDLE, draw_only_contours=False)

    output_NN = model.predict(X, verbose=1)
    return output_NN

def saveMetricsData(dsc_data, m_names, outputImages, save_file=True):
    # ************** Plot and save Metrics for ROI *****************
    print('\t Making Bar plots ...')
    dsc_data.loc['AVG'] = dsc_data.mean() # Compute all average values
    title = F"DSC AVG ROI: {dsc_data.loc['AVG'][m_names.get('dsc_roi')]:.3f}"
    utilsviz.plot_multiple_bar_plots([dsc_data[m_names.get('dsc_roi')].dropna().to_dict()], title=title, legends=['ROI'],
                                     save_path=join(outputImages,'aroi_DSC.png'))
    dsc_data.to_csv(join(outputImages,'aroi_DSC.csv'))

def procSingleCase(inFolder, outputDirectory, outputImages, current_folder,
                    model, threshold, roi_slices, dsc_data, m_names, save_segmentations=True, indx=0):
    '''
    This function process the segmentation of a single case. It makes the NN predition and saves the results
    :param inFolder:  Input folder where all the years are
    :param outputDirectory: Where to save the prediction as nrrd file
    :param outputImages: Where to save the images and CSV files
    :param current_folder: Current case
    :param multistream: Using multistream model
    :param model: TensorFlow model
    :param threshold: Threshold to binarize the images
    :param roi_slices:
    :param dsc_data: DataFrame with the current metrics for all the years
    :param m_names: String array with the names of the metrics like DSC, etc.
    :param save_segmentations: Boolean: indicate if we want to save the results of the NN
    :param indx: Integer of the current number of the folder, used to save the results only every 10 years
    :return:
    '''
    print(F'----{current_folder} ------------')
    # Reads original image and prostate
    roi_img, roi_ctr_kidney, roi_ctr_tumor = load_case_preproc_itk(current_folder, inFolder, ['img_crop.nrrd','ctr_crop_kidney.nrrd','ctr_crop_tumor.nrrd'])
    roi_img_np = sitk.GetArrayFromImage(roi_img)
    roi_ctr_kidney_np = sitk.GetArrayFromImage(roi_ctr_kidney)
    roi_ctr_tumor_np = sitk.GetArrayFromImage(roi_ctr_tumor)

    print('\tPredicting image {} ({})....'.format(current_folder, inFolder))
    output_NN = makePrediction(model, roi_img, roi_ctr_kidney)

    # ************** Binary threshold and largest connected component ******************
    print('\tThreshold and largest component...')
    pred_nn = sitk.GetImageFromArray(output_NN[0,:,:,:,0])
    pred_nn = utils.binaryThresholdImage(pred_nn, threshold)
    # pred_nn = utils.getLargestConnectedComponents(pred_nn)
    np_pred_nn = sitk.GetArrayViewFromImage(pred_nn)

    # ************** Compute metrics for ROI ******************
    c_img_folder = join(outputImages,current_folder)
    print('\tMetrics...')
    cur_dsc_roi = numpy_dice(roi_ctr_kidney_np, np_pred_nn)
    print(F'\tDSC ROI: {cur_dsc_roi:02.2f}  ------------')


    # ************** Visualize and save results for ROI ******************
    slices = roi_slices
    title = F'DSC {cur_dsc_roi:02.3f}'
    print('\tMaking ROI images...')
    labels = ['GT','NN','Tumor']
    utilsviz.draw_multiple_series_numpy(roi_img_np, [roi_ctr_kidney_np, np_pred_nn, roi_ctr_tumor_np], slices=slices, c_id=current_folder,
                                        out_folder=outputImages, title=title, labels=labels, plane=PlaneTypes.AXIAL)
    utilsviz.draw_multiple_series_numpy(roi_img_np, [roi_ctr_kidney_np, np_pred_nn, roi_ctr_tumor_np], slices=slices, c_id=current_folder,
                                        out_folder=outputImages, title=title, labels=labels, plane=PlaneTypes.CORONAL)
    utilsviz.draw_multiple_series_numpy(roi_img_np, [roi_ctr_kidney_np, np_pred_nn, roi_ctr_tumor_np], slices=slices, c_id=current_folder,
                                        out_folder=outputImages, title=title, labels=labels, plane=PlaneTypes.SAGITTAL)

    # ************** Save ROI segmentation *****************
    if save_segmentations:
        print('\tSaving original prediction (ROI)...')
        if not os.path.exists(join(outputDirectory, current_folder)):
            os.makedirs(join(outputDirectory, current_folder))
        sitk.WriteImage(pred_nn, join(outputDirectory, current_folder, 'predicted_roi.nrrd'))

    dsc_data.loc[current_folder][m_names.get('dsc_roi')] = cur_dsc_roi

    # ************** Plot and save Metrics for ROI *****************
    if (indx % 10 == 0) and (indx > 3): # TODO It fails if we let it go the first index
        saveMetricsData(dsc_data, m_names, outputImages)

    return dsc_data

def getProperFolders(inFolder, years):
    '''Depending on the value of years it reads the proper folders from the list of folders'''
    # *********** Define which years are we going to perform the segmentation **********
    if isinstance(years,str):
        if years == 'all':
            examples = os.listdir(inFolder)
    else:
        examples = [format_cid(case) for case in years]

    examples.sort()

    return examples
def makeSegmentation(inFolder, outputDirectory, outputImages, model_weights_file,
                     all_params, years='all', save_segmentations=True):
    '''
    This function computes a new mask from the spedified model weights and model
    :param inFolder:
    :param outputDirectory:
    :param outputImages:
    :param model_weights_file:
    :param all_params:
    :param years:
    :param save_segmentations:
    :return:
    '''
    # *********** Reads the parameters ***********
    roi_slices = all_params['roi_slices']

    img_dims = [168, 168, 168]
    threshold = 0.5

    # *********** Chooses the proper model ***********
    print('Reading model ....')
    # model = models.getModel_3D_Single(img_dims)
    model = models.get3DUNet(img_dims, num_levels=3, initial_filters=8)

    # *********** Reads the weights***********
    print('Reading weights ....')
    model.load_weights(model_weights_file)

    examples = getProperFolders(inFolder, years)

    if not os.path.exists(outputImages):
        os.makedirs(outputImages)

    # *********** Makes a dataframe to contain the DSC information **********
    m_names = {'dsc_roi':'ROI'}

    # Check if the output fiels already exist, in thtat case read the df from it.
    # if os.path.exists(join(outputImages,'all_DSC.csv')):
    #     dsc_data = pd.read_csv(join(outputImages,'all_DSC.csv'), index_col=0)
    # else:
    dsc_data = DataFrame(index = examples, columns=[m_names.get('dsc_roi')])

    # *********** Iterates over each case *********
    for id_folder, current_folder in enumerate(examples):
        t0 = time.time()
        try:
            dsc_data = procSingleCase(inFolder, outputDirectory, outputImages, current_folder, model, threshold,
                                      roi_slices, dsc_data, m_names, save_segmentations, indx=id_folder)
        except Exception as e:
            print(F"---------------------Failed Case {current_folder} error: {e} ----------------")
        print(F'\tElapsed time {time.time()-t0:0.2f} seg')

    saveMetricsData(dsc_data, m_names, outputImages)

if __name__ == '__main__':

    all_params = getClassifyConfig()

    model_name = all_params['model_name']
    disp_images = all_params['display_images']
    years = all_params['years']
    save_segmentations = all_params['save_segmentations']
    utilsviz.view_results = disp_images

    outputImages = all_params['output_images']
    inputDirectory = all_params['input_folder']
    outputDirectory = all_params['output_folder']
    model_weights_file = all_params['weights']
    makeSegmentation(inputDirectory, outputDirectory, outputImages, model_weights_file,
                     all_params, years=years, save_segmentations=save_segmentations)


