from keras.layers import *
from keras.models import *
from NN.modelBuilder import *

def getSimple2DNet(imgs_dims, num_vars=vars, num_levels=3, initial_filters=8):

    filter_size = 3
    filterFactor = 1
    [w, h] = imgs_dims
    d = num_vars
    print(F'\tMaking model with initial dimensions: {(w,h,d)}')

    #  [128,200]
    inputs = Input((h,w,d))
    conv1 = Conv2D(initial_filters * filterFactor, (filter_size, filter_size), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(initial_filters * filterFactor, (filter_size, filter_size), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    #  [64,100]
    conv3 = Conv2D(initial_filters * filterFactor * 2, (filter_size, filter_size), activation='relu', padding='same')(pool1)
    conv4 = Conv2D(initial_filters * filterFactor * 2, (filter_size, filter_size), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    #  [32,50]
    conv5 = Conv2D(initial_filters * filterFactor * 4, (filter_size, filter_size), activation='relu', padding='same')(pool2)
    conv6 = Conv2D(initial_filters * filterFactor * 4, (filter_size, filter_size), activation='relu', padding='same')(conv5)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    #  [16,25]
    conv7 = Conv2D(initial_filters * filterFactor * 8, (filter_size, filter_size), activation='relu', padding='same')(pool3)
    conv8 = Conv2D(initial_filters * filterFactor * 8, (filter_size, filter_size), activation='relu', padding='same')(conv7)
    flat = Flatten()(conv8)

    # Single output value
    out = Dense(1, activation='sigmoid')(flat)
    model = Model(inputs=[inputs], outputs=[out])

    print('\tDone creating the model!')
    return model


def get2DUNet(imgs_dims, num_vars, num_levels, initial_filters):
    filter_size = 3
    [w, h] = imgs_dims
    d = num_vars

    c_filters = initial_filters
    inputs = Input((d, h, w))
    lastconv = make2DUNet(inputs, c_filters, filter_size=filter_size, num_levels=num_levels)

    model = Model(inputs=[inputs], outputs=[lastconv])
    return model

def get3DUNet(imgs_dims, num_levels, initial_filters):
    filter_size = 3

    [w, h, d] = imgs_dims

    c_filters = initial_filters
    inputs = Input((d, h, w, 1))
    lastconv = make3DUNet(inputs, c_filters, fsize=filter_size, num_levels=num_levels)

    model = Model(inputs=[inputs], outputs=[lastconv])
    return model
