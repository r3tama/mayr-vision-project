"""
Module to predict with previously trained model
"""
import cv2
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from numba import njit, vectorize, prange
from typing import List, Any, Tuple,Dict
from nptyping import NDArray
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from ctypes import *
from sklearn.metrics import confusion_matrix

# User-defined modules
from models import *

# Definition of types for typing
DataSources = Dict[str,str]
Image = NDArray[(Any, Any, 3), int]
ImageSeg = NDArray[(Any, Any, 3), int]
ImageSegBinary = NDArray[(Any, Any, 1), int]
ImageCollection = NDArray[(Any), Image]
ImageSegCollection = NDArray[(Any), ImageSeg]
ImageSegBinaryCollection = NDArray[(Any), ImageSegBinary]

# For printing all nmumpy array values
np.set_printoptions(threshold=np.inf)

def getItems(data_src: List[str],lbl_src: List[str]) -> Tuple[ImageCollection, ImageSegBinaryCollection]:
    count = 0
    x = np.zeros((len(data_src),224,224,3),dtype=np.int32)
    for j,path in enumerate(data_src):
        count += 1
        print("{}: {}".format(count,path))
        x[j] = tf.keras.preprocessing.image.load_img(path,target_size=(224,224))
    count = 0
    y = np.zeros((len(data_src),224,224,1),dtype=np.int32)
    for j,path in enumerate(lbl_src):
        count += 1
        print("{}: {}".format(count,path))
        img = tf.keras.preprocessing.image.load_img(path,color_mode="grayscale",target_size=(224,224))
        y[j] = np.expand_dims(img,2)
        y[j] -= 1
        
    return x,y
 
def loadCsvFile(filename: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
        Function to load original images and ground truth images from .csv file
        Args:
            filename: .csv file name

        Returns:
            a tuple with a list of original images  and a list of ground truth images
    """
    if not isinstance(filename, str):
        raise TypeError("Name is not a string")
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=";")
        next(reader)
        data_storage = list(reader)
        train_dict, test_dict = train_test_split(data_storage,test_size=0.3,train_size=0.7,random_state=69)
        train_data_src = [d[0] for d in train_dict]
        train_labl_src = [d[1] for d in train_dict]
        test_data_src = [d[0] for d in test_dict]
        test_labl_src = [d[1] for d in test_dict]
        return train_data_src,train_labl_src, test_data_src,test_labl_src
     
if __name__ == "__main__":

    tf.keras.backend.clear_session()
    # Limit GPU memory. Uncomment -> limit to 2GB
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # tf.config.experimental.set_virtual_device_configuration(
                    # gpu,
                    # [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
        except RuntimeError:
            print("Invalid GPU configuration")

    # Set where the channels are specified
    tf.keras.backend.set_image_data_format("channels_last")

    train_data_src,train_labl_src, test_data_src,test_labl_src = loadCsvFile('dogCat.csv')

    # Net params
    numClasses = 3

    net: UNetX = UNetX(img_size=(224,224,3),n_filters=[32,64,128,256,256,128,64,32], n_classes=numClasses)
    net.summary()

    net.load_weights('resultTraining/bestModel.hdf5')


    # Load test values
    dataTest, lblTest = getItems(test_data_src, test_labl_src)
    # Normalize data and convert lbl from 3 to 1 dimension
    dataTest = dataTest / 255.0

    # Print random predicted image, mask, ground truth for testing
    test1 = dataTest[5,:,:,:]
    lbl1 = lblTest[5,:,:,:]
    test = np.expand_dims(test1, 0)
    dataPredict = net.predict(test)
    mask = np.argmax(dataPredict[0], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    predicted = dataPredict[0,:,:,:]

    plt.imshow(mask)
    plt.show()
    plt.imshow(predicted)
    plt.show()
    plt.imshow(lbl1)
    plt.show()
