from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Convolution2D, Deconvolution2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np


def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x


def lambda_output(input_shape):
    return input_shape[:2]


def conv_block_unet(x, f, name, bn_mode, bn_axis, bn=True, subsample=(2,2)):

    x = LeakyReLU(0.2)(x)
    x = Convolution2D(f, 3, 3, subsample=subsample, name=name, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)

    return x


def conv_block_fire(x, f, lol, name, bn_mode, bn_axis, bn=True, subsample=(2,2)):
    print f, "lololo"
    # x = MaxPooling2D(
    #     pool_size=(2, 2), strides=(2, 2))(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(lol, 3, 3, activation='relu', subsample=(2, 2),
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(
        f/8, 1, 1,
        border_mode='same', name=name)(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = LeakyReLU(0.2)(x)
    e1 = Convolution2D(
        f/2, 1, 1,
        border_mode='same')(x)
    e1 = BatchNormalization(mode=bn_mode, axis=bn_axis)(e1)
    # e1 = LeakyReLU(0.2)(e1)
    e2 = Convolution2D(
        f/2, 3, 3,
        border_mode='same')(x)
    e2 = BatchNormalization(mode=bn_mode, axis=bn_axis)(e2)
    # e2 = LeakyReLU(0.2)(e2)
    x = merge(
        [e1, e2], mode='concat', concat_axis=bn_axis)
    return x


def conv_block_god(x, f, name, bn_mode, bn_axis, bn=True, subsample=(2,2)):
    print f, "lord"
    x = LeakyReLU(0.2)(x)
    x = Convolution2D(
        f, 3, 3, subsample=subsample,
        border_mode='same', name=name)(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # x = Convolution2D(
    #     f/8, 1, 1, activation='relu', init='glorot_uniform',
    #     border_mode='same', name=name)(x)
    # x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # e1 = Convolution2D(
    #     f/2, 1, 1, activation='relu', init='glorot_uniform',
    #     border_mode='same')(x)
    # e2 = Convolution2D(
    #     f/2, 3, 3, activation='relu', init='glorot_uniform',
    #     border_mode='same')(x)
    # x = merge(
    #     [e1, e2], mode='concat', concat_axis=bn_axis)
    return x

def up_conv_block_unet(x, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(f, 3, 3, name=name, border_mode="same")(x)
    if bn:
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x

def up_conv_block_fire(x, x2, f, name, bn_mode, bn_axis, bn=True, dropout=False):
    print f, "upupup"
    print x, x2
    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(
        f/8, 1, 1,
        border_mode='same', name=name)(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Activation("relu")(x)
    e1 = Convolution2D(
        f/2, 1, 1,
        border_mode='same')(x)
    e1 = BatchNormalization(mode=bn_mode, axis=bn_axis)(e1)
    e2 = Convolution2D(
        f/2, 3, 3,
        border_mode='same')(x)
    e2 = BatchNormalization(mode=bn_mode, axis=bn_axis)(e2)
    x = merge(
        [e1, e2], mode='concat', concat_axis=bn_axis)
    if dropout:
        x = Dropout(0.5)(x)
    x = merge([x, x2], mode='concat', concat_axis=bn_axis)

    return x

def generator_bullshit(img_dim, bn_mode, model_name="generator_fire_upsampling"):

    nb_filters = 64
    print "HIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
    if K.image_dim_ordering() == "th":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    m0 = Convolution2D(32, 3, 3, activation='relu', border_mode='same', name='block0_conv2')(unet_input)

    # x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(unet_input)
    m5 = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2', subsample=(2, 2))(m0)

    # Block 2
    m4 = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2', subsample=(2, 2))(m5)

    # Block 3
    m3 = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3', subsample=(2, 2))(m4)

    # Block 5
    m1 = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3', subsample=(2, 2))(m3)

    x = Activation("relu")(m1)
    x = Convolution2D(512, 3, 3, border_mode="same", subsample=(2, 2))(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Dropout(0.5)(x)

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(512, 3, 3, border_mode="same")(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Dropout(0.5)(x)
    x = merge([x, m1], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(256, 3, 3, border_mode="same")(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Dropout(0.5)(x)
    x = merge([x, m3], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(128, 3, 3, border_mode="same")(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Dropout(0.5)(x)
    x = merge([x, m4], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(64, 3, 3, border_mode="same")(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Dropout(0.5)(x)
    x = merge([x, m5], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(32, 3, 3, border_mode="same")(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    x = Dropout(0.5)(x)
    x = merge([x, m0], mode='concat', concat_axis=bn_axis)

    x = Convolution2D(3, 3, 3, border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_vgg = Model(input=[unet_input], output=[x])

    return generator_vgg


def generator_fire_upsampling(img_dim, bn_mode, model_name="generator_fire_upsampling"):

    nb_filters = 64
    print "HIIIIIIIIIIIIIIIIIIIIIIIIIIIIII"
    if K.image_dim_ordering() == "th":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    last_god = list_nb_filters.pop()

    # Encoder
    list_encoder = [Convolution2D(list_nb_filters[0], 3, 3,
                                  subsample=(2, 2), name="unet_conv2D_1", border_mode="same")(unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "fire%s_squeeze" % (i + 2)
        conv = conv_block_fire(list_encoder[-1], f, list_nb_filters[i], name, bn_mode, bn_axis)
        list_encoder.append(conv)

    print list_encoder

    conv = conv_block_god(list_encoder[-1], last_god, "conv", bn_mode, bn_axis)
    list_encoder.append(conv)

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-1][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)


    # Decoder
    list_decoder = [up_conv_block_fire(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], "unet_up_fire_1", bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_up_fire_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_fire(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(nb_channels, 3, 3, name="last_conv", border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(input=[unet_input], output=[x])

    return generator_unet

def generator_fire_squeezenet_reverse(img_dim, bn_mode, model_name="generator_fire_squeezenet_reverse"):
    nb_filters = 64

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    # Copied from squeezenet keras github
    unet_input = Input(shape=img_dim, name="unet_input")
    conv1 = Convolution2D(
        96, 7, 7, activation='relu', init='glorot_uniform',
        subsample=(2, 2), border_mode='same', name='conv1')(unet_input)
    maxpool1 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name='maxpool1')(conv1)
    fire2_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_squeeze')(maxpool1)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(fire2_squeeze)
    fire2_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand1')(x)
    fire2_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand2')(x)
    merge2 = merge([fire2_expand1, fire2_expand2], mode='concat', concat_axis=bn_axis)

    fire3_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_squeeze')(merge2)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(fire3_squeeze)
    fire3_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand1')(x)
    fire3_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand2',
        )(x)
    merge3 = merge([fire3_expand1, fire3_expand2], mode='concat', concat_axis=bn_axis)

    fire4_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_squeeze')(merge3)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(fire4_squeeze)
    fire4_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand1',
        )(x)
    fire4_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand2',
        )(x)
    merge4 = merge([fire4_expand1, fire4_expand2], mode='concat', concat_axis=bn_axis)
    maxpool4 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name='maxpool4',
        )(merge4)

    fire5_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_squeeze',
        )(maxpool4)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(fire5_squeeze)
    fire5_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand1',
        )(x)
    fire5_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand2',
        )(x)
    merge5 = merge([fire5_expand1, fire5_expand2], mode='concat', concat_axis=bn_axis)

    fire6_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_squeeze',
        )(merge5)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(fire6_squeeze)
    fire6_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand1',
        )(x)
    fire6_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand2',
        )(x)
    merge6 = merge([fire6_expand1, fire6_expand2], mode='concat', concat_axis=bn_axis)

    fire7_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_squeeze',
        )(merge6)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(fire7_squeeze)
    fire7_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand1',
        )(x)
    fire7_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand2',
        )(x)
    merge7 = merge([fire7_expand1, fire7_expand2], mode='concat', concat_axis=bn_axis)

    fire8_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_squeeze',
        )(merge7)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(fire8_squeeze)
    fire8_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand1',
        )(x)
    fire8_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand2',
        )(x)
    merge8 = merge([fire8_expand1, fire8_expand2], mode='concat', concat_axis=bn_axis)

    maxpool8 = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2), name='maxpool8',
        )(merge8)
    fire9_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_squeeze',
        )(maxpool8)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(fire9_squeeze)
    fire9_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand1',
        )(x)
    fire9_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand2',
        )(x)
    merge9 = merge([fire9_expand1, fire9_expand2], mode='concat', concat_axis=bn_axis)
    print(merge9)

    x = Dropout(0.5, name='fire9_dropout')(merge9)

    x = Activation("relu")(x)
    x = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    e1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same')(x)
    e2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same')(x)
    x = merge(
        [e1, e2], mode='concat', concat_axis=bn_axis)
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = merge([x, merge9], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(
        512, 1, 1,
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = merge([x, merge8], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = Convolution2D(
        192*2, 1, 1,
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = merge([x, merge7], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = Convolution2D(
        192*2, 1, 1,
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = merge([x, merge6], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = Convolution2D(
        256, 1, 1,
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = merge([x, merge5], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(
        256, 1, 1,
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = merge([x, merge4], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = Convolution2D(
        128, 1, 1,
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = merge([x, merge3], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = Convolution2D(
        128, 1, 1,
        border_mode='same')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # if dropout:
    #     x = Dropout(0.5)(x)
    x = merge([x, merge2], mode='concat', concat_axis=bn_axis)

    x = Activation("relu")(x)
    x = UpSampling2D(size=(4, 4))(x)
    x = Convolution2D(nb_channels, 3, 3, name="last", border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_fire_squeezenet = Model(input=[unet_input], output=[x])
    return generator_fire_squeezenet



def generator_unet_upsampling(img_dim, bn_mode, model_name="generator_unet_upsampling"):

    nb_filters = 64

    if K.image_dim_ordering() == "th":
        bn_axis = 1
        nb_channels = img_dim[0]
        min_s = min(img_dim[1:])
    else:
        bn_axis = -1
        nb_channels = img_dim[-1]
        min_s = min(img_dim[:-1])

    unet_input = Input(shape=img_dim, name="unet_input")

    # Prepare encoder filters
    nb_conv = int(np.floor(np.log(min_s) / np.log(2)))
    list_nb_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # Encoder
    list_encoder = [Convolution2D(list_nb_filters[0], 3, 3,
                                  subsample=(2, 2), name="unet_conv2D_1", border_mode="same")(unet_input)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_conv2D_%s" % (i + 2)
        conv = conv_block_unet(list_encoder[-1], f, name, bn_mode, bn_axis)
        list_encoder.append(conv)

    # Prepare decoder filters
    list_nb_filters = list_nb_filters[:-2][::-1]
    if len(list_nb_filters) < nb_conv - 1:
        list_nb_filters.append(nb_filters)

    # Decoder
    list_decoder = [up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                                       list_nb_filters[0], "unet_upconv2D_1", bn_mode, bn_axis, dropout=True)]
    for i, f in enumerate(list_nb_filters[1:]):
        name = "unet_upconv2D_%s" % (i + 2)
        # Dropout only on first few layers
        if i < 2:
            d = True
        else:
            d = False
        conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i + 3)], f, name, bn_mode, bn_axis, dropout=d)
        list_decoder.append(conv)

    x = Activation("relu")(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Convolution2D(nb_channels, 3, 3, name="last_conv", border_mode="same")(x)
    x = Activation("tanh")(x)

    generator_unet = Model(input=[unet_input], output=[x])

    return generator_unet


def DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name="DCGAN_discriminator", use_mbd=True):
    """
    Discriminator model of the DCGAN

    args : img_dim (tuple of int) num_chan, height, width
           pretr_weights_file (str) file holding pre trained weights

    returns : model (keras NN) the Neural Net model
    """

    list_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]

    if K.image_dim_ordering() == "th":
        bn_axis = 1
    else:
        bn_axis = -1

    nb_filters = 64
    nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
    list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]

    # First conv
    x_input = Input(shape=img_dim, name="discriminator_input")
    # x = Convolution2D(list_filters[0], 3, 3, subsample=(2, 2), name="disc_conv2d_1", border_mode="same")(x_input)
    # x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    # x = LeakyReLU(0.2)(x)

    x = MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(x_input)
    x = Convolution2D(
        list_filters[0]/8, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='disc_conv2d_1')(x)
    x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
    e1 = Convolution2D(
        list_filters[0]/2, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same')(x)
    e2 = Convolution2D(
        list_filters[0]/2, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same')(x)
    x = merge(
        [e1, e2], mode='concat', concat_axis=bn_axis)

    # Next convs
    for i, f in enumerate(list_filters[1:]):
        name = "disc_conv2d_fire_%s" % (i + 2)
        # x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same")(x)
        # x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        # x = LeakyReLU(0.2)(x)
        x = MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2))(x)
        x = Convolution2D(
            f/8, 1, 1, activation='relu', init='glorot_uniform',
            border_mode='same', name=name)(x)
        x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
        e1 = Convolution2D(
            f/2, 1, 1, activation='relu', init='glorot_uniform',
            border_mode='same')(x)
        e2 = Convolution2D(
            f/2, 3, 3, activation='relu', init='glorot_uniform',
            border_mode='same')(x)
        x = merge(
            [e1, e2], mode='concat', concat_axis=bn_axis)

    x_flat = Flatten()(x)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)

    PatchGAN = Model(input=[x_input], output=[x, x_flat], name="PatchGAN")
    print("PatchGAN summary")
    PatchGAN.summary()

    x = [PatchGAN(patch)[0] for patch in list_input]
    x_mbd = [PatchGAN(patch)[1] for patch in list_input]

    if len(x) > 1:
        x = merge(x, mode="concat", name="merge_feat")
    else:
        x = x[0]

    if use_mbd:
        if len(x_mbd) > 1:
            x_mbd = merge(x_mbd, mode="concat", name="merge_feat_mbd")
        else:
            x_mbd = x_mbd[0]

        num_kernels = 100
        dim_per_kernel = 5

        M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
        MBD = Lambda(minb_disc, output_shape=lambda_output)

        x_mbd = M(x_mbd)
        x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
        x_mbd = MBD(x_mbd)
        x = merge([x, x_mbd], mode='concat')

    x_out = Dense(2, activation="softmax", name="disc_output")(x)

    discriminator_model = Model(input=list_input, output=[x_out], name=model_name)

    return discriminator_model


# def DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name="DCGAN_discriminator", use_mbd=True):
#     """
#     Discriminator model of the DCGAN
#     args : img_dim (tuple of int) num_chan, height, width
#            pretr_weights_file (str) file holding pre trained weights
#     returns : model (keras NN) the Neural Net model
#     """
#
#     list_input = [Input(shape=img_dim, name="disc_input_%s" % i) for i in range(nb_patch)]
#
#     if K.image_dim_ordering() == "th":
#         bn_axis = 1
#     else:
#         bn_axis = -1
#
#     nb_filters = 64
#     nb_conv = int(np.floor(np.log(img_dim[1]) / np.log(2)))
#     list_filters = [nb_filters * min(8, (2 ** i)) for i in range(nb_conv)]
#
#     # First conv
#     x_input = Input(shape=img_dim, name="discriminator_input")
#     x = Convolution2D(list_filters[0], 3, 3, subsample=(2, 2), name="disc_conv2d_1", border_mode="same")(x_input)
#     x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
#     x = LeakyReLU(0.2)(x)
#
#     # Next convs
#     for i, f in enumerate(list_filters[1:]):
#         name = "disc_conv2d_%s" % (i + 2)
#         x = Convolution2D(f, 3, 3, subsample=(2, 2), name=name, border_mode="same")(x)
#         x = BatchNormalization(mode=bn_mode, axis=bn_axis)(x)
#         x = LeakyReLU(0.2)(x)
#
#     x_flat = Flatten()(x)
#     x = Dense(2, activation='softmax', name="disc_dense")(x_flat)
#
#     PatchGAN = Model(input=[x_input], output=[x, x_flat], name="PatchGAN")
#     print("PatchGAN summary")
#     PatchGAN.summary()
#
#     x = [PatchGAN(patch)[0] for patch in list_input]
#     x_mbd = [PatchGAN(patch)[1] for patch in list_input]
#
#     if len(x) > 1:
#         x = merge(x, mode="concat", name="merge_feat")
#     else:
#         x = x[0]
#
#     if use_mbd:
#         if len(x_mbd) > 1:
#             x_mbd = merge(x_mbd, mode="concat", name="merge_feat_mbd")
#         else:
#             x_mbd = x_mbd[0]
#
#         num_kernels = 100
#         dim_per_kernel = 5
#
#         M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
#         MBD = Lambda(minb_disc, output_shape=lambda_output)
#
#         x_mbd = M(x_mbd)
#         x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
#         x_mbd = MBD(x_mbd)
#         x = merge([x, x_mbd], mode='concat')
#
#     x_out = Dense(2, activation="softmax", name="disc_output")(x)
#
#     discriminator_model = Model(input=list_input, output=[x_out], name=model_name)
#
#     return discriminator_model

def DCGAN(generator, discriminator_model, img_dim, patch_size, image_dim_ordering):

    gen_input = Input(shape=img_dim, name="DCGAN_input")

    generated_image = generator(gen_input)

    if image_dim_ordering == "th":
        h, w = img_dim[1:]
    else:
        h, w = img_dim[:-1]
    ph, pw = patch_size

    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(h / ph)]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(w / pw)]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            if image_dim_ordering == "tf":
                x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(generated_image)
            else:
                x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])(generated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriminator_model(list_gen_patch)

    DCGAN = Model(input=[gen_input],
                  output=[generated_image, DCGAN_output],
                  name="DCGAN")

    return DCGAN


def load(model_name, img_dim, nb_patch, bn_mode, use_mbd, batch_size):

    if model_name == "generator_unet_upsampling":
        model = generator_unet_upsampling(img_dim, bn_mode, model_name=model_name)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "generator_bullshit":
        model = generator_bullshit(img_dim, bn_mode, model_name=model_name)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "generator_fire_upsampling":
        model = generator_fire_upsampling(img_dim, bn_mode, model_name=model_name)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "generator_fire_squeezenet_reverse":
        model = generator_fire_squeezenet_reverse(img_dim, bn_mode, model_name=model_name)
        print model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model

    if model_name == "DCGAN_discriminator":
        model = DCGAN_discriminator(img_dim, nb_patch, bn_mode, model_name=model_name, use_mbd=use_mbd)
        model.summary()
        from keras.utils.visualize_util import plot
        plot(model, to_file='../../figures/%s.png' % model_name, show_shapes=True, show_layer_names=True)
        return model


if __name__ == '__main__':

    # load("generator_unet_deconv", (256, 256, 3), 16, 2, False, 32)
    load("generator_unet_upsampling", (256, 256, 3), 16, 2, False, 32)
