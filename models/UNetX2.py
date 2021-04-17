import tensorflow as _tf
from tensorflow import keras as _keras
import numpy as _np

from .types import *
from typing import List, Tuple


class UNetXception(_keras.Model):
    """
    Classs for creating and working with a U-Net Xception-style model
    Args:
        nFilters -> vector with filters for different layers
        nClasses -> number of classes for final layer
    """

    def __init__(self, nFilters: List[int], nClasses: int) -> None:
        super(UNetXception, self).__init__()

        # Handle errors
        if not isinstance(nFilters, list) or not isinstance(nFilters[0], int):
            raise TypeError("nFilters must be a list of ints")
        if not isinstance(nClasses, int):
            raise TypeError("nClasses mus be of type int")
        if len(nFilters) != 8:
            raise ValueError("nFilters must have 8 elements")
             
        self.nFilters = nFilters
        self.nClasses = nClasses

        _inputs = _keras.Input((480,720,3))
        
        # Create convolutions for first layer
        self.conv1 = _keras.layers.Conv2D(filters=self.nFilters[0],
                                          kernel_size=3,
                                          strides=2,
                                          padding="same")
        self.batchNorm1 = _keras.layers.BatchNormalization()
        self.act1 = _keras.layers.Activation("relu")


        # Initialize arrays for layers with different filter size
         
        # self.sepConv = _np.empty(3, dtype=SepConv2DKeras)
        # self.sepAct = _np.empty(3, dtype=ActivationKeras)
        # self.sepBatch = _np.empty(3, dtype=BatchNormKeras)
        # self.residualDown = _np.empty(3, dtype=Conv2DKeras)
        # self.transConv = _np.empty(4, dtype=TranspConv2DKeras)
        # self.transAct = _np.empty(4, dtype=ActivationKeras)
        # self.transBatch = _np.empty(4, dtype=BatchNormKeras)
        # self.residualUp = _np.empty(4, dtype=Conv2DKeras)
         
        # self.sepConv = _np.empty(len(self.nFilters), dtype=int)
        # self.sepAct = _np.empty(len(self.nFilters), dtype=int)
        # self.sepBatch = _np.empty(len(self.nFilters), dtype=int)
        # self.residualDown - _np.empty(len(self.nFilter), dtype=int)
        # self.transConv - _np.empty(len(self.nFilter), dtype=int)
        # self.transAct = _np.empty(len(self.nFilters), dtype=int)
        # self.transBatch = _np.empty(len(self.nFilters), dtype=int)
        # self.residualUp - _np.empty(len(self.nFilter), dtype=int)

        # First half of the network. Downsampling convolutions

        # First convolution x2
        self.sepConv1 = _keras.layers.SeparableConv2D(filters=self.nFilters[1],
                                                            kernel_size=3,
                                                            padding="same")
        self.sepAct1 = _keras.layers.Activation("relu")
        self.sepBatch1 = _keras.layers.BatchNormalization()
         
        self.sepConv2 = _keras.layers.SeparableConv2D(filters=self.nFilters[1],
                                                            kernel_size=3,
                                                            padding="same")
        # self.sepAct1 = _keras.layers.Activation("relu")
        self.sepBatch2 = _keras.layers.BatchNormalization()
        # Create pooling
        self.pooling1 = _keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  padding="same")
        self.residualDown1 = _keras.layers.Conv2D(filters=self.nFilters[1],
                                                    kernel_size=1,
                                                    padding="same",
                                                    strides=2)
         
        # Second convolution x2
        self.sepConv3 = _keras.layers.SeparableConv2D(filters=self.nFilters[2],
                                                            kernel_size=3,
                                                            padding="same")
        self.sepAct2 = _keras.layers.Activation("relu")
        self.sepBatch3 = _keras.layers.BatchNormalization()
         
        self.sepConv4 = _keras.layers.SeparableConv2D(filters=self.nFilters[2],
                                                            kernel_size=3,
                                                            padding="same")
        # self.sepAct2 = _keras.layers.Activation("relu")
        self.sepBatch4 = _keras.layers.BatchNormalization()
        self.pooling2 = _keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  padding="same")
        self.residualDown2 = _keras.layers.Conv2D(filters=self.nFilters[2],
                                                    kernel_size=1,
                                                    padding="same",
                                                    strides=2)
         
        # Thrid convolution x2
        self.sepConv5 = _keras.layers.SeparableConv2D(filters=self.nFilters[3],
                                                            kernel_size=3,
                                                            padding="same")
        self.sepAct3 = _keras.layers.Activation("relu")
        self.sepBatch5 = _keras.layers.BatchNormalization()
         
        self.sepConv6 = _keras.layers.SeparableConv2D(filters=self.nFilters[3],
                                                            kernel_size=3,
                                                            padding="same")
        # self.sepAct3 = _keras.layers.Activation("relu")
        self.sepBatch7 = _keras.layers.BatchNormalization()
        self.pooling3 = _keras.layers.MaxPooling2D(pool_size=3,
                                                  strides=2,
                                                  padding="same")
        self.residualDown3 = _keras.layers.Conv2D(filters=self.nFilters[3],
                                                    kernel_size=1,
                                                    padding="same",
                                                    strides=2)
        # for filt in self.nFilters[1:4]:
            # self.sepConv[count] = _keras.layers.SeparableConv2D(filters=filt,
                                                                # kernel_size=3,
                                                                # padding="same")
            # self.sepAct[count] = _keras.layers.Activation("relu")
            # self.sepBatch[count] = _keras.layers.BatchNormalization()
            # self.residualDown[count] = _keras.layers.Conv2D(filters=filt,
                                                        # kernel_size=1,
                                                        # padding="same",
                                                        # strides=2)
            # count += 1
        # Second half. Updampling

        # First deconvolution x2
        self.transConv1 = _keras.layers.Conv2DTranspose(filters=self.nFilters[4],
                                                   kernel_size=3,
                                                   padding="same")
        self.transAct1 = _keras.layers.Activation("relu")
        self.transBatch1 = _keras.layers.BatchNormalization()
        # Create upsampling
        self.upsampling1 = _keras.layers.UpSampling2D(size=2)
        self.upsamplingResidual1 = _keras.layers.UpSampling2D(size=2)
        self.residualUp1 = _keras.layers.Conv2D(filters=self.nFilters[4],
                                                      kernel_size=1,
                                                      padding="same")

        # Second deconvolution x2 
        self.transConv2 = _keras.layers.Conv2DTranspose(filters=self.nFilters[5],
                                                   kernel_size=3,
                                                   padding="same")
        self.transAct2 = _keras.layers.Activation("relu")
        self.transBatch2 = _keras.layers.BatchNormalization()
        self.upsampling2 = _keras.layers.UpSampling2D(size=2)
        self.upsamplingResidual2 = _keras.layers.UpSampling2D(size=2)
        self.residualUp2 = _keras.layers.Conv2D(filters=self.nFilters[5],
                                                      kernel_size=1,
                                                      padding="same")
         
        # Thrid deconvolution x2
        self.transConv3 = _keras.layers.Conv2DTranspose(filters=self.nFilters[6],
                                                   kernel_size=3,
                                                   padding="same")
        self.transAct3 = _keras.layers.Activation("relu")
        self.transBatch3 = _keras.layers.BatchNormalization()
        self.upsampling3 = _keras.layers.UpSampling2D(size=2)
        self.upsamplingResidual3 = _keras.layers.UpSampling2D(size=2)
        self.residualUp3 = _keras.layers.Conv2D(filters=self.nFilters[6],
                                                      kernel_size=1,
                                                      padding="same")
         
        # Fourth deconvolution x2
        self.transConv4 = _keras.layers.Conv2DTranspose(filters=self.nFilters[7],
                                                   kernel_size=3,
                                                   padding="same")
        self.transAct4 = _keras.layers.Activation("relu")
        self.transBatch4 = _keras.layers.BatchNormalization()
        self.upsampling4 = _keras.layers.UpSampling2D(size=2)
        self.upsamplingResidual4 = _keras.layers.UpSampling2D(size=2)
        self.residualUp4 = _keras.layers.Conv2D(filters=self.nFilters[7],
                                                      kernel_size=1,
                                                      padding="same")
        # count = 0
        # # Second half of the network. Upsampling convolutions
        # for filt in self.nFilters[4:]:
            # self.transConv[count] = _keras.layers.Conv2DTranspose(filters=filt,
                                                       # kernel_size=3,
                                                       # padding="same")
            # self.transAct[count] = _keras.layers.Activation("relu")
            # self.transBatch[count] = _keras.layers.BatchNormalization()
            # self.residualUp[count] = _keras.layers.Conv2D(filters=filt,
                                                          # kernel_size=1,
                                                          # padding="same")
            # count += 1

        # Create final layer with classes number as size
        self._output = _keras.layers.Conv2D(filters=self.nClasses,
                                        kernel_size=3,
                                        activation="softmax",
                                        padding="same")

        self.out = self.call(_inputs)
        super().__init__(inputs=_inputs, outputs=self.out)

    def call(self, inputs):
        # Create first convolutional layer
        x = self.conv1(inputs)
        x = self.batchNorm1(x)
        x = self.act1(x)

        # Save layer for residual movement
        prevLayer = x
        # for _ in range(2):
        x = self.sepAct1(x)
        x = self.sepConv1(x)
        x = self.sepBatch1(x)
         
        x = self.sepAct1(x)
        x = self.sepConv2(x)
        x = self.sepBatch2(x)
         
        x = self.pooling1(x)
        residual = self.residualDown1(prevLayer)
        x = _keras.layers.add([x, residual])
        prevLayer = x
             
        # for _ in range(2):
        x = self.sepConv3(x)
        x = self.sepBatch3(x)
        x = self.sepAct1(x)
         
        # x = self.sepConv2(x)
        # x = self.sepBatch2(x)
        # x = self.sepAct2(x)
         
        # x = self.pooling(x)
        # residual = self.residualDown2(prevLayer)
        # x = _keras.layers.add([x, residual])
        # prevLayer = x
         
        # for _ in range(2):
            # x = self.sepConv3(x)
            # x = self.sepBatch3(x)
            # x = self.sepAct3(x)
        # x = self.pooling(x)
        # residual = self.residualDown3(prevLayer)
        # x = _keras.layers.add([x, residual])
        # prevLayer = x
         
        # for _ in range(2):
            # x = self.transConv1(x)
            # x = self.transBatch1(x)
            # x = self.transAct1(x)
        # x = self.upsampling(x)
        # residual = self.upsampling(prevLayer)
        # residual = self.residualUp1(residual)
        # x = _keras.layers.add([x, residual])
        # prevLayer = x
         
        # for _ in range(2):
            # x = self.transConv2(x)
            # x = self.transBatch2(x)
            # x = self.transAct2(x)
        # x = self.upsampling(x)
        # residual = self.upsampling(prevLayer)
        # residual = self.residualUp2(residual)
        # x = _keras.layers.add([x, residual])
        # prevLayer = x
         
        # for _ in range(2):
            # x = self.transConv3(x)
            # x = self.transBatch3(x)
            # x = self.transAct3(x)
        # x = self.upsampling(x)
        # residual = self.upsampling(prevLayer)
        # residual = self.residualUp3(residual)
        # x = _keras.layers.add([x, residual])
        # prevLayer = x
         
        # for _ in range(2):
            # x = self.transConv4(x)
            # x = self.transBatch4(x)
            # x = self.transAct4(x)
        # x = self.upsampling(x)
        # residual = self.upsampling(prevLayer)
        # residual = self.residualUp4(residual)
        # x = _keras.layers.add([x, residual])
        # prevLayer = x
         
         
        # # Convolutional downsampling (filters number = 3)
        # for count in range(3):
            # for _ in range(2):
                # x = self.sepConv[count](x)
                # x = self.sepBatch[count](x)
                # x = self.sepAct[count](x)
            # x = self.pooling(x)
            # residual = self.residualDown[count](prevLayer)
            # x = _keras.layers.add([x, residual])
            # prevLayer = x

        # # Convolutional upsampling (filters number = 4)
        # for count in range(4):
            # for _ in range(2):
                # x = self.transConv[count](x)
                # x = self.transBatch[count](x)
                # x = self.transAct[count](x)
            # x = self.upsampling(x)
            # residual = self.upsampling(prevLayer)
            # residual = self.residualUp[count](residual)
            # x = _keras.layers.add([x, residual])
            # prevLayer = x


        # Final layer
        # x = self.out(x)

        return x


if __name__ == "__main__":
    pass
