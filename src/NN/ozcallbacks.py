import keras.callbacks as callbacks
from os.path import join

def getAllCallbacks(model_name, early_stopping, weights_folder='.', logs_folder='.', min_delta=.001, patience=70):
    root_logdir = join(logs_folder,'logs')
    logdir = "{}/run-{}/".format(root_logdir, model_name)

    logger = callbacks.TensorBoard(
        log_dir=logdir,
        write_graph=True,
        write_images=False,
        histogram_freq=0
    )

    # Saving the model every epoch
    filepath_model = join(weights_folder, model_name+'-{epoch:02d}-{val_loss:.2f}.hdf5')
    save_callback = callbacks.ModelCheckpoint(filepath_model,monitor=early_stopping, save_best_only=True,
                                              mode='max',save_weights_only=True)

    # Early stopping
    stop_callback = callbacks.EarlyStopping(monitor=early_stopping, min_delta=min_delta, patience=patience, mode='max')
    return [logger, save_callback, stop_callback]


