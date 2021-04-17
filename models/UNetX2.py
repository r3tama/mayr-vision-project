import tensorflows as _tf
from tensorflow import keras as _keras

from .types import *
from typing import List, Tuple


class UNetXception(_keras.Model):
    """
    Classs for creating and working with a U-Net Xception-style model
    """

    def __init__(self, nFilters=None):
        super(UNetXception, self).__init__()
        self.nFilters = nFilters
        
        self.conv1 = _keras.layers.Conv2D(filters+self.nFilters,
                                          kernel_size=3,
                                          strides=2,
                                          padding="same")
        self.batchNorm1 = _keras.layers.BatchNormalization()
        self.act1 = _keras.layers.Activation("relu")

        self.sepConv1 = _keras.layers.SeparableConv2D(self.nFilters, 
                                                      kernel_size=3, 
                                                      padding="same")
        self.sepBatchNorm1 = _keras.layers.BatchNormalization()
        self.sepAct1 = _keras.layers.Activation("relu")


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchNorm1(x)
        x = self.act1(x)

        x = self.sepConv1(x)
        x = self.sepBatchNorm1(x)
        x = self.sepAct1(x)
        return x


if __name__ == "__main__":
    pass
