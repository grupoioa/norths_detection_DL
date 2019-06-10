
from src.config.MainConfig import getMapVisualizationConfig
from src.constants.visualizationModes import VizualizationModes
import visualization.utilsviz as utilviz
from inout.readData import *
from pathlib import Path
import os

if __name__ == '__main__':

    options = getMapVisualizationConfig()
    data_folder = str(Path(options['input_folder']).resolve().absolute())
    years = options['years']
    vars = options['variables']

    mode = options['mode']

    if mode == VizualizationModes.OZVIZ:
        # This options plots images on specific plane and specific slices
        # This is linked to a variable inside utilsviz
        utilviz.view_results = options['display']

        if options['save_figures']:
            out_img_folder = str(Path(options['output_folder']).resolve().absolute())
        else:
            out_img_folder = ''

        data = load_years_np(data_folder, years=years, variables=vars)
        if (options['mode'] == VizualizationModes.OZVIZ):
            utilviz.draw_multiple_years(data, years, vars)

