from config.MainConfig import getPreprocessingConfig
from inout.readData import *
from inout.io_common import saveImage
from preproc.preprocImages import *
from visualization.utilsviz import *
# import visualization.utilsviz
from src.constants.preprocMode import PreprocMode
from src.constants.fileNames import FileNames

if __name__ == '__main__':
    # TODO aqui tal vez seria necesario generar archivos netcdf SOLO con las variables de interes

    # Things that need to be done
    # 1) Flip with respect to Y (lattitude)
    # 2) Interpolate both axes to a size that can be divided by 8 (like 168)
    # 3) Select only desired variables
