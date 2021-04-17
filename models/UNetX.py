"""
This module defines the class UNetX for working with a U-Net Xception-style model
"""

import tensorflow as _tf
from tensorflow import keras as _keras

from .types import *
from typing import List,Tuple

class UNetX(_keras.Model):
    """
    Class for creating and working with a U-Net Xception-style model
    Args:
        img_size: ColorImageSize with the shape of the images that will be fit in
            A None will be preapended for batch feeding
        n_filters: List of integers that contain the number of filters per
            convolutional layer, it has to have 8 elements
    """
    
    def __init__(self, img_size: ColorImageSize, n_filters: List[int]) -> None:
        super().__init__()
        if not isinstance(img_size,tuple) or not isinstance(img_size[0],int) or img_size[2] != 3:
            raise TypeError("img_size must be a ColorImageSize")
        if not isinstance(n_filters,list) or not isinstance(n_filters[0],int):
            raise TypeError("n_filters must be a list of ints")
        if len(img_size) != 3:
            raise TypeError("img_size must be a ColorImageSize")
        if len(n_filters) != 8:
            raise ValueError("n_filters must have 8 elements")
        ## Imput of the network
        in_l = _keras.Input(shape=(None,img_size[0],img_size[1],img_size[2]))

        ## First half of the network
        ## Downsampling
        # Entry
        # x = _keras.layers.Conv2D(n_filters[0],3,strides=2,padding="same")(in_l)
        # x = _keras.layers.BatchNormalization()(x)
        # x = _keras.layers.Activation("relu")(x)

        # Residual saved
        # prev_block = x

        # for filter_s in n_filters[1:4]:
            # for _ in range(2):
                # x = _keras.layers.Activation("relu")(x)
                # x = _keras.layers.SeparableConv2D(filter_s,3,padding="same")(x)
                # x = _keras.layers.BatchNormalization()

            # # Residual
            # residual  = _keras.layers.Conv2D(filter_s,1,strides=2,padding="same")(prev_block)
            # x = _keras.layers.add([x,residual])
            # prev_block = x

        ## Second half
        ## Upsampling
        # for filter_s in n_filters[4:]



if __name__ == "__main__":
    pass
