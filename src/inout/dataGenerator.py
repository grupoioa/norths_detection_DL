import numpy as np
from preproc.preprocImages import *
from inout.readData import *
import visualization.utilsviz as utilsviz

def dataGenerator(path, all_dates_to_read, Y, variables, img_dims, data_augmentation=False, batch_size=1, normalize=True):

    """
    Generator to yield inputs and their labels in batches.
    """
    print(F"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALLED Generator {all_dates_to_read} %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    Y_orig = np.array(Y).astype(np.int)

    curr_idx = -1 # First index to use
    files_to_read = [getFileFromDate(path,x) for x in all_dates_to_read]
    np.random.shuffle(files_to_read) # We shuffle the folders every time we have tested all the examples

    while True:
        # If we are not at the end, just get the next batch
        if curr_idx < (len(all_dates_to_read) - batch_size):
            curr_idx += batch_size
        else:
            curr_idx = 0
            np.random.shuffle(files_to_read) # We shuffle the folders every time we have tested all the examples

        try:

            data_batch = readDataTraining(files_to_read[curr_idx:curr_idx+batch_size], variables)
            data_batch_norm = interpolateNumpyArray(data_batch, img_dims)

            # Reordering the axes,
            X = np.swapaxes(data_batch_norm,1,3)
            Y = np.expand_dims(Y_orig[curr_idx:curr_idx+batch_size], axis=1)
            # print(F'Final dims: {X.shape} and {Y.shape}')

            yield X, Y

        except Exception as e:
            print("----- Not able to generate for: ", files_to_read, " ERROR: ", str(e))


