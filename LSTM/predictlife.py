import tensorflow as tf
import numpy as np
from keras.layers import Conv2Dolution
from keras import backend as k

model = tf.keras.models.load_model('battery-life-model.h5')

def create_deconv_layer(layer):

    filter,kernel_size ,strides =layer.get_weights()

    deconv_layer = Conv2Dolution(filter=filter, 
                                 kernel_size=kernel_size,
                                 strides=strides,
                                 padding = 'same',
                                 activation ='real')

    deconv_layer.set_weights([filter.T,np.ones(filter.shape)])
    return deconv_layer

