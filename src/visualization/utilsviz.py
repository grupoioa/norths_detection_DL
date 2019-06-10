import matplotlib.pyplot as plt
from src.constants.readMode import ReadMode

colors = ['y', 'r', 'c', 'b', 'g', 'w', 'k', 'y', 'r', 'c', 'b', 'g', 'w', 'k']

view_results = True

def display_images():
    if view_results:
        plt.show()
    else:
        plt.close()


def drawSingleMap(map):
    plt.imshow(map)
    display_images()


def draw_multiple_years(data, years: list, variables=ReadMode.ALL_YEARS, title='', out_folder='', labels=[]):
    '''
    Main function to draw some variables as maps
    :param data:
    :param years:
    :param variables:
    :param title:
    :param out_folder:
    :param labels:
    :return:
    '''
    years_str = [str(y) for y in years]
    for c_year in years_str:
        dates_per_year = [x for x in data.keys() if x.find(c_year) != -1]
        for c_date in dates_per_year:
            if variables == ReadMode.ALL_YEARS:
                variables = data[c_date].keys()
            tot_vars = len(variables)

            fig, axarr = plt.subplots(1,tot_vars, squeeze=True)
            fig.suptitle(F'{c_date}', fontsize=25)
            for var_idx, c_var in enumerate(variables):
                # ------ Here is where we should define how to plot each map -----------
                c_map = data[c_date][c_var]
                print(F'Resolution is {c_map.shape}')
                axarr[var_idx].imshow(c_map)
                ctitle = F'{title} {c_year} {c_var}'
                axarr[var_idx].set_title(ctitle, fontsize=20)
            display_images()


