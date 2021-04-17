"""
Module to tinker with models, data loading and trainings
"""
import cv2
import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from typing import List, Any, Tuple,Dict
from nptyping import NDArray
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from ctypes import *

# User-defined modules
from models import *

# Definition of types for typing
DataSources = Dict[str,str]
Image = NDArray[(Any, Any, 3), int]
ImageSeg = NDArray[(Any, Any, 3), int]
ImageSegBinary = NDArray[(Any, Any, 1), int]
ImageSegCollection = NDArray[(Any), ImageSeg]
ImageSegBinaryCollection = NDArray[(Any), ImageSegBinary]

# For printing all nmumpy array values
np.set_printoptions(threshold=np.inf)

def loadFromDataSources(d_list: List[DataSources]) -> Tuple[List[Image], List[ImageSeg]]:
    """
    Function to load the original images and ground truth images from the paths
    given in the DataSources
    Args:
        d_list: List of DataSources

    Returns:
        A tuple with a list of original images and a list of ground truth images
    """
    if len(d_list) == 0:
        raise ValueError("Param is empty")
    if not isinstance(d_list[0], dict):
        raise TypeError("Param is not a dictionary list")
    data = []
    lbl = []
    count = 0
    for row in d_list:
        if not "data" in row.keys() or not "label" in row.keys():
            raise ValueError("Param dictionaries do not contain the desired keys")
        try:
            count += 1
            if count % 15 == 0:
                print(count)
            auxData = cv2.imread(row["data"])
            auxLbl = cv2.imread(row["label"])
            auxData = cv2.resize(auxData, (480,720))
            auxLbl = cv2.resize(auxLbl, (480,720))
            if auxData is not None:
                data.append(auxData)
            if auxLbl is not None:
                lbl.append(auxLbl)
        except ValueError:
            sys.stderr.write("Could not load an image")
    return data, lbl

def loadCsvFile(filename: str) -> Tuple[List[Image], List[ImageSeg],List[DataSources]]:
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
        data_storage = list(csv.DictReader(csvfile, delimiter=";"))
        # train_dict, test_dict = train_test_split(data_storage,test_size=0.3,train_size=0.7,random_state=69)
        train_dict, test_dict = train_test_split(data_storage,test_size=0.95,train_size=0.05,random_state=69)
        data,lbl = loadFromDataSources(train_dict)
        return data,lbl,test_dict

def oneDim2rgbLabel(imgBin: ImageSegBinaryCollection) -> ImageSegCollection:
    """
    Function to convert 1D ground truth values to 3D images
    Input:
        Numpy array of 1D ground truth values. Input must be (numImg, widht, height, 1)
    Returns:
        Numpy array with 3D images (numImg, widht, height, 3) with the followig codification:
        00 -> unlabeled
        01 -> paved-grass
        02 -> dirt
        03 -> grass
        04 -> gravel
        05 -> water
        06 -> rocks
        07 -> pool
        08 -> vegetation
        09 -> roof
        10 -> wall
        11 -> window
        12 -> door
        13 -> fence
        14 -> fence-pole
        15 -> person
        16 -> dog
        17 -> car
        18 -> bicycle
        19 -> tree
        20 -> bald-tree
        21 -> ar-marker
        22 -> obstacle
        23 -> conflicting

    """
    lblDict = {
        "unlabeled": [0, 0, 0], # 0
        "paved-area": [128, 64, 128], # 1
        "dirt": [0, 76, 130], # 2
        "grass": [0, 102, 0], # 3
        "gravel": [87, 103, 112], # 4
        "water": [168, 42, 28], # 5
        "rocks": [30, 41, 48], # 6
        "pool": [89, 50, 0], # 7
        "vegetation": [35, 142, 107], # 8 
        "roof": [70, 70, 70], # 9
        "wall": [156, 102, 102], # 10
        "window": [12, 228, 254], # 11
        "door": [12, 148, 254], # 12
        "fence": [153, 153, 190], # 13 
        "fence-pole": [153, 153, 153], # 14
        "person": [96, 22, 255], # 15
        "dog": [0, 51, 102], # 16 
        "car":[150, 143, 9], # 17
        "bicycle": [32, 11, 119], # 18
        "tree": [0, 51, 51],  # 19
        "bald-tree": [190, 250, 190], # 20
        "ar-marker": [146, 150, 112], # 21
        "obstacle": [115, 135, 2], # 22
        "conflicting": [0, 0, 255]  # 23
    }
     
    if len(imgBin.shape) != 4:
        raise TypeError("Array is not 4D")
    if imgBin.shape[3] != 1:
        raise ValueError("Array must have format (numImg, width, height, 1)")
     
    img = np.zeros((imgBin.shape[0],imgBin.shape[1],imgBin.shape[2], 3), dtype=np.int16)
    count = 0
    for i in range(imgBin.shape[0]):
        for j in range(imgBin.shape[1]):
            for k in range(img.shape[2]):
                if imgBin[i, j, k] == 0:
                    img[i, j, k, :]  = lblDict["unlabeled"]
                elif imgBin[i, j, k] == 1:
                    img[i, j, k, :] = lblDict["paved-area"]
                elif imgBin[i, j, k] == 2:
                    img[i, j, k, :] = lblDict["dirt"]
                elif imgBin[i, j, k] == 3:
                    img[i, j, k, :] = lblDict["grass"]
                elif imgBin[i, j, k] == 4:
                    img[i, j, k, :] = lblDict["gravel"]
                elif imgBin[i, j, k] == 5:
                    img[i, j, k, :] = lblDict["water"]
                elif imgBin[i, j, k] == 6:
                    img[i, j, k, :] = lblDict["rocks"]
                elif imgBin[i, j, k] == 7:
                    img[i, j, k, :] = lblDict["pool"]
                elif imgBin[i, j, k] == 8:
                    img[i, j, k, :] = lblDict["vegetation"]
                elif imgBin[i, j, k] == 9:
                    img[i, j, k, :] = lblDict["roof"]
                elif imgBin[i, j, k] == 10:
                    img[i, j, k, :] = lblDict["wall"]
                elif imgBin[i, j, k] == 11:
                    img[i, j, k, :] = lblDict["window"]
                elif imgBin[i, j, k] == 12:
                    img[i, j, k, :] = lblDict["door"]
                elif imgBin[i, j, k] == 13:
                    img[i, j, k, :] = lblDict["fence"]
                elif imgBin[i, j, k] == 14:
                    img[i, j, k, :] = lblDict["fence-pole"]
                elif imgBin[i, j, k] == 15:
                    img[i, j, k, :] = lblDict["person"]
                elif imgBin[i, j, k] == 16:
                    img[i, j, k, :] = lblDict["dog"]
                elif imgBin[i, j, k] == 17:
                    img[i, j, k, :] = lblDict["car"]
                elif imgBin[i, j, k] == 18:
                    img[i, j, k, :] = lblDict["bicycle"]
                elif imgBin[i, j, k] == 19:
                    img[i, j, k, :] = lblDict["tree"]
                elif imgBin[i, j, k] == 20:
                    img[i, j, k, :] = lblDict["bald-tree"]
                elif imgBin[i, j, k] == 21:
                    img[i, j, k, :] = lblDict["ar-marker"]
                elif imgBin[i, j, k] == 22:
                    img[i, j, k, :] = lblDict["obstacle"]
                elif imgBin[i, j, k] == 23:
                    img[i, j, k] = lblDict["conflicting"]
        count += 1
        if count % 15 == 0:
            print("Converted {} images to 3D".format(count))
    return img

def rgb2oneDimLabel(img: ImageSegCollection) -> ImageSegBinaryCollection:
    """
    Function to convert 3D ground truth images to 1D numeric values
    Input:
        Numpy array of 3D ground truth images. Input must be (numImg, widht, height, 3)
    Returns:
        Numpy array with 1D images (numImg, widht, height, 1) with the followig codification:
        00 -> unlabeled
        01 -> paved-grass
        02 -> dirt
        03 -> grass
        04 -> gravel
        05 -> water
        06 -> rocks
        07 -> pool
        08 -> vegetation
        09 -> roof
        10 -> wall
        11 -> window
        12 -> door
        13 -> fence
        14 -> fence-pole
        15 -> person
        16 -> dog
        17 -> car
        18 -> bicycle
        19 -> tree
        20 -> bald-tree
        21 -> ar-marker
        22 -> obstacle
        23 -> conflicting

    """
    if len(img.shape) != 4:
        raise TypeError("Array is not 4D")
    if img.shape[3] != 3:
        raise ValueError("Array must have format (numImg, width, height, 3)")
    imgBin = np.zeros((img.shape[0],img.shape[1],img.shape[2], 1), dtype=np.int16)
    count = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                if np.array_equal(img[i, j, k, :], [0, 0, 0]):
                    imgBin[i,j] = 0
                elif np.array_equal(img[i, j, k, :], [128, 64, 128]):
                    imgBin[i,j] = 1
                elif np.array_equal(img[i, j, k, :], [0, 76, 130]):
                    imgBin[i,j] = 2
                elif np.array_equal(img[i, j, k, :], [0, 102, 0]):
                    imgBin[i,j] = 3
                elif np.array_equal(img[i, j, k, :], [87, 103, 112]):
                    imgBin[i,j] = 4
                elif np.array_equal(img[i, j, k, :], [168, 42, 28]):
                    imgBin[i,j] = 5
                elif np.array_equal(img[i, j, k, :], [30, 41, 48]):
                    imgBin[i,j] = 6
                elif np.array_equal(img[i, j, k, :], [89, 50, 0]):
                    imgBin[i,j] = 7
                elif np.array_equal(img[i, j, k, :], [35, 142, 107]):
                    imgBin[i,j] = 8
                elif np.array_equal(img[i, j, k, :], [70, 70, 70]):
                    imgBin[i,j] = 9
                elif np.array_equal(img[i, j, k, :], [156, 102, 102]):
                    imgBin[i,j] = 10
                elif np.array_equal(img[i, j, k, :], [12, 228, 254]):
                    imgBin[i,j] = 11
                elif np.array_equal(img[i, j, k, :], [12, 148, 254]):
                    imgBin[i, j] = 12
                elif np.array_equal(img[i, j, k, :], [153, 153, 190]):
                    imgBin[i, j] = 13
                elif np.array_equal(img[i, j, k, :], [153, 153, 153]):
                    imgBin[i, j] = 14
                elif np.array_equal(img[i, j, k, :], [96, 22, 255]):
                    imgBin[i, j] = 15
                elif np.array_equal(img[i, j, k, :], [0, 51, 102]):
                    imgBin[i, j] = 16
                elif np.array_equal(img[i, j, k, :], [150, 143, 9]):
                    imgBin[i, j] = 17
                elif np.array_equal(img[i, j, k, :], [32, 11, 119]):
                    imgBin[i, j] = 18
                elif np.array_equal(img[i, j, k, :], [0, 51, 51]):
                    imgBin[i, j] = 19
                elif np.array_equal(img[i, j, k, :], [190, 250, 190]):
                    imgBin[i, j] = 20
                elif np.array_equal(img[i, j, k, :], [146, 150, 112]):
                    imgBin[i, j] = 21
                elif np.array_equal(img[i, j, k, :], [115, 135, 2]):
                    imgBin[i, j] = 22
                elif np.array_equal(img[i, j, k, :], [0, 0, 255]):
                    imgBin[i, j] = 23
        count += 1
        if count % 15 == 0:
            print("Converted {} images to 1D".format(count))
    return imgBin


if __name__ == "__main__":
     
    # Set where the channels are specified
    tf.keras.backend.set_image_data_format("channels_last")
     
    data, lbl, test_dict = loadCsvFile('img.csv')
    # Normalize data
    data = np.array(data, dtype=np.float32)
    data = data / 255.0
    # Convert labels from 3 to 1 dimension
    lbl = np.array(lbl, dtype=np.int32)
    # lblBin = rgb2oneDimLabel(lbl)

    print(lbl.shape)
    # convertDimensions = cdll.LoadLibrary("libconvertDimension.so")
    convertDimensions = np.ctypeslib.load_library("libconvertDimension.so",".")
    lblBin_c = convertDimensions.rgb2oneDimLabel(c_void_p(lbl.ctypes.data), lbl.shape[0], lbl.shape[1], lbl.shape[2])
    print("uncasted")
    print(lblBin_c)
    ptr = np.ctypeslib.ndpointer(c_int32,1,(lbl.shape[0]*720*480))
    lblBin_c = cast(lblBin_c,POINTER(c_int32))
    print("casted")
    print(lblBin_c)
    # lblBin = np.ctypeslib.as_array(lblBin_c,shape=(lbl.shape[0]*720*480,))
    lblBin = np.ctypeslib.as_array(ptr,shape=(lbl.shape[0]*720*480,))
    lblBin.reshape((lbl.shape[0],720,480,1))
    print(lblBin)
    print("lblBin type: {}, lbl shape: {}".format(type(lblBin), lblBin.shape))

    numClasses = 24
    nEpochs = 20



    # plt.axis('off')
    # plt.imshow(cv2.cvtColor(lbl[1], cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.imshow(img.astype(np.uint8))
    # plt.show()
    # data, lbl,test_dict = loadCsvFile('img.csv')
    net: UNetX = UNetX(img_size=(480,720,3),n_filters=[32,64,128,256,256,128,64,32],n_classes=24)
    net.summary()

    net.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


    # Create checkpoints to save differente models
    path = "weightsEpoch_{epoch:02d}_valLoss_{val_loss:.2f}.hdf5"
    path2 = "bestModel.hdf5"
    checkpoint = ModelCheckpoint(path, monitor="val_loss", verbose=1, save_best_only=True)
    checkpoint2 = ModelCheckpoint(path2, monitor="val_loss", verbose=1, save_best_only=True)
    callbackList = [checkpoint, checkpoint2]
     
    history = net.fit(data, lblBin, epochs=nEpochs, batch_size=16, callbacks=callbackList)

    # Evaluation
    # score = net.evaluate(data, lblBin, verbose=0)
    # print("Test Error: %.2f%%" % (100-score[1]*100))
    # print("%s: %.2f%%" % (net.metrics_names[1], score[1]*100))

    # # generate predictions for test
    # testPredict = net.predict(X[test])
    # ytestPredict = []
    # for element in testPredict:
        # index, value = max(enumerate(element), key=operator.itemgetter(1))
        # ytestPredict.append(index)

    # print('\n\n----------------------------------------------------')
    # print('Confusion Matrix')
    # testConf=confusion_matrix(y[test], ytestPredict)
    # print(testConf)

    # Loss Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    # save the losses figure
    plt.tight_layout()
    plt.savefig('losses.png')
    plt.close()
      
    # Accuracy Curves
    plt.figure(figsize=[8,6])
    plt.plot(history.history['accuracy'],'r',linewidth=3.0)
    plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

    # save the accuracies figure
    plt.tight_layout()
    plt.savefig('accs.png')
    plt.close()
     
    # # Save confusion matrix in file
    # with open('results.txt', '+a') as file:
        # file.write('\n\n-------------------------------------------')
        # file.write('Confusion Matrix fold '+str(count))
        # file.write(testConf)
