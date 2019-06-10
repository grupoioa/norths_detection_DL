import NN.models as models
from datetime import datetime
import pandas as pd
from NN.ozcallbacks import *
from keras.optimizers import *
import NN.utilsNN as utilsNN
from os.path import join
from inout.dataGenerator import *
from NN.metrics import *
from inout.io_common import *
from config.MainConfig import getTrainingConfig
from pandas import DataFrame
import pandas as pd

if __name__ == '__main__':

    config = getTrainingConfig()

    input_folder = config['input_folder']
    output_folder = config['output_folder']
    val_perc = config['val_perc']
    test_perc = config['test_perc']
    eval_metrics = config['eval_metrics']
    loss_func = config['loss_func']
    batch_size = config['batch_size']
    epochs = config['epochs']
    useDA = config['useDA']
    vars = config['variables']

    all_years = readAllYears(input_folder)
    all_dates = getDatesFromYears(input_folder, all_years)
    tot_examples = len(all_dates)
    # img_dims = [73,127]
    img_dims = [128,200]

    Y = np.genfromtxt(config['norths_file'],delimiter=',',dtype=np.str)

    splitInfoFolder = join(output_folder,'Splits')
    parametersFolder = join(output_folder,'Parameters')
    weightsFolder= join(output_folder,'models')
    logsFolder = join(output_folder,'logs')
    createFolder(splitInfoFolder)
    createFolder(parametersFolder)
    createFolder(weightsFolder)

    train_ids = []
    val_ids = []
    test_ids = []
    # ================ Split definition
    [train_ids, val_ids, test_ids] = utilsNN.splitTrainValidationAndTest(tot_examples, val_perc=val_perc, test_perc=test_perc)

    print("Train examples (total:{}) :{}".format(len(train_ids), all_dates[train_ids]))
    print("Validation examples (total:{}) :{}:".format(len(val_ids), all_dates[val_ids]))
    print("Test examples (total:{}) :{}".format(len(test_ids), all_dates[test_ids]))

    # ================ Reads the model
    print("Setting the model....")
    now = datetime.utcnow().strftime("%Y_%m_%d_%H_%M")

    # model = models.get2DUNet(img_dims, num_vars=len(vars), num_levels=3, initial_filters=8)
    model = models.getSimple2DNet(img_dims, num_vars=len(vars), num_levels=3, initial_filters=8)
    model_name = F'Name_{now}'

    # ================ Saving splits
    file = open(join(splitInfoFolder,'{}.txt'.format(model_name)),'w')
    file.write("\n\nTrain examples (total:{}) :{}".format(len(train_ids), all_dates[train_ids]))
    file.write("\n\nValidation examples (total:{}) :{}:".format(len(val_ids), all_dates[val_ids]))
    file.write("\n\nTest examples (total:{}) :{}".format(len(test_ids), all_dates[test_ids]))
    file.close()

    # ================ Configures the optimization
    print("Configuring optimization....")

    # optimizer = SGD() # Default values lr=0.01, momentum=0., decay=0.,
    optimizer = Adam() # Default values lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0., amsgrad=False
    # optimizer = SGD(lr=0.03, momentum=0.3, decay=0.0, nesterov=False)
    # optimizer='sgd' #'adam', 'sgd'

    model.compile(loss=loss_func, optimizer=optimizer, metrics=eval_metrics)

    [logger, save_callback, stop_callback] = getAllCallbacks(model_name, F'val_{eval_metrics[0].__name__}', weightsFolder, join(output_folder))

    # ================ Trains the model
    generator = True

    if generator:
        options={ 'input_folder': input_folder, }

        train_generator = dataGenerator(input_folder, all_dates[train_ids], Y[train_ids][:,1], vars, img_dims,
                                        data_augmentation=useDA, batch_size=batch_size, normalize=True)
        val_generator = dataGenerator(input_folder, all_dates[val_ids], Y[val_ids][:,1], vars, img_dims,
                                      data_augmentation=useDA, batch_size=batch_size, normalize=True)

        model.fit_generator(train_generator, steps_per_epoch=min(50,len(train_ids)),
                            validation_data= val_generator,
                            validation_steps=min(30,len(val_ids)),
                             use_multiprocessing=False,
                             workers=1,
                            # use_multiprocessing=True,
                            # workers=4,
                            epochs=epochs, callbacks=[logger, save_callback, stop_callback] )
