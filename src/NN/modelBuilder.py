from keras.layers import *
from keras.models import *

def two_cnn_3d(input, num_filters, filter_size, make_pool=True, batch_normalization=False):
    '''
    Makes a 'pair' of CNN, it may be using batch normalization and/or maxpool
    :param input:
    :param num_filters:
    :param filter_size:
    :param make_pool:
    :param batch_normalization:
    :return:
    '''
    if batch_normalization:
        conv1 = Conv3D(num_filters, (filter_size, filter_size,filter_size), padding='same', activation='relu')(input)
        b1= BatchNormalization(axis=4)(conv1)
        conv3 = Conv3D(num_filters, (filter_size, filter_size,filter_size), padding='same', activation='relu')(b1)
        last = BatchNormalization(axis=4)(conv3)
    else:
        conv1 = Conv3D(num_filters, (filter_size, filter_size,filter_size), padding='same', activation='relu')(input)
        last = Conv3D(num_filters, (filter_size, filter_size,filter_size), padding='same', activation='relu')(conv1)
    if make_pool:
        maxpool = MaxPooling3D(pool_size=(2, 2, 2))(last)
    else:
        maxpool = []
    return last, maxpool

def two_cnn_2d(input, num_filters, filter_size, make_pool=True, batch_normalization=False):
    '''
    Makes a 'pair' of CNN, it may be using batch normalization and/or maxpool
    :param input:
    :param num_filters:
    :param filter_size:
    :param make_pool:
    :param batch_normalization:
    :return:
    '''
    if batch_normalization:
        conv1 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation='relu')(input)
        b1= BatchNormalization(axis=4)(conv1)
        conv3 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation='relu')(b1)
        last = BatchNormalization(axis=4)(conv3)
    else:
        conv1 = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation='relu')(input)
        last = Conv2D(num_filters, (filter_size, filter_size), padding='same', activation='relu')(conv1)
    if make_pool:
        maxpool = MaxPooling2D(pool_size=(2, 2))(last)
    else:
        maxpool = []
    return last, maxpool

def make_up_2d(input, match_lay, num_filters, filter_size, batch_normalization=True):
    '''
    :param input: Is the CNN we will upsample
    :param match_lay: Is the CNN layer in the other side of the U (we will concatenate)
    :param num_filters:
    :param filter_size:
    :return:
    '''
    convT = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), activation='relu', padding='same')(input)
    print(F' Concatenating {convT.shape} with {match_lay.shape}')

    up = concatenate([ convT, match_lay])

    [conv, dele] = two_cnn_2d(up, int(num_filters), filter_size, make_pool=False, batch_normalization=batch_normalization)
    return conv



def make_up_3d(input, match_lay, num_filters, filter_size, batch_normalization=True):
    '''
    :param input: Is the CNN we will upsample
    :param match_lay: Is the CNN layer in the other side of the U (we will concatenate)
    :param num_filters:
    :param filter_size:
    :return:
    '''
    convT = Conv3DTranspose(num_filters, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(input)
    print(F' Concatenating {convT.shape} with {match_lay.shape}')

    up = concatenate([ convT, match_lay])

    [conv, dele] = two_cnn_3d(up, int(num_filters), filter_size, make_pool=False, batch_normalization=batch_normalization)
    return conv

def make2DUNet(inputs, num_filters, filter_size, num_levels):
    convs = []
    maxpools = []

    # Going down
    print(F"Building analysis path.... input shape {inputs.shape}")
    c_input = inputs
    for level in range(num_levels+1):
        if level == num_levels:
            m_pool = False
        else:
            m_pool = True

        print()
        filters = num_filters*(2**level)
        convT, poolT = two_cnn_2d(c_input, filters, filter_size, make_pool=m_pool, batch_normalization=True)

        if level != num_levels:
            print(F"Filters {filters} Conv: {convT.shape} Pool: {poolT.shape} ")
        else:
            print(F"Filters {filters} Conv: {convT.shape} ")

        convs.append(convT)
        maxpools.append(poolT)
        c_input = maxpools[-1] # Set the next input as the last output

    print("\n ------------- Building synthesis path....")
    for level in range(num_levels):
        convT = make_up_3d(convs[-1], convs[num_levels-level-1], num_filters*(2**(num_levels-level-1)), filter_size, batch_normalization=True)
        convs.append(convT)
        print(F"Output shape {convT.shape}")

    lastconv = Conv3D(1, (1, 1, 1), activation='sigmoid')(convs[-1])
    print(F"Final shape {lastconv.shape}")

    return lastconv


def make3DUNet(inputs, num_filters, filter_size, num_levels):
    convs = []
    maxpools = []

    # Going down
    print(F"Building analysis path.... input shape {inputs.shape}")
    c_input = inputs
    for level in range(num_levels+1):
        if level == num_levels:
            m_pool = False
        else:
            m_pool = True

        print()
        filters = num_filters*(2**level)
        convT, poolT = two_cnn_3d(c_input, filters, filter_size, make_pool=m_pool, batch_normalization=True)

        if level != num_levels:
            print(F"Filters {filters} Conv: {convT.shape} Pool: {poolT.shape} ")
        else:
            print(F"Filters {filters} Conv: {convT.shape} ")

        convs.append(convT)
        maxpools.append(poolT)
        c_input = maxpools[-1] # Set the next input as the last output

    print("\n ------------- Building synthesis path....")
    for level in range(num_levels):
        convT = make_up_3d(convs[-1], convs[num_levels-level-1], num_filters*(2**(num_levels-level-1)), filter_size, batch_normalization=True)
        convs.append(convT)
        print(F"Output shape {convT.shape}")

    lastconv = Conv3D(1, (1, 1, 1), activation='sigmoid')(convs[-1])
    print(F"Final shape {lastconv.shape}")

    return lastconv

