"""
Module to tinker with models, data loading and trainings
"""
import cv2
import numpy as np
import csv
import sys
from typing import List, Any, Tuple,Dict
from nptyping import NDArray
from sklearn.model_selection import train_test_split

# User-defined modules
from models import UNetX

DataSources = Dict[str,str]
"""
Dictionary form string to string with the keys "data" and "label"
"""

Image = NDArray[(Any, Any, 3), int]
"""
Color image with the data
"""

ImageSeg = NDArray[(Any, Any, 3), int]
"""
Color image with the mask
"""

def loadFromDataSources(d_list: List[DataSources]) -> Tuple[List[Image], List[ImageSeg]]:
    """
    Function to load the original images and ground truth images from the paths
    given in the DataSources
    Input:
        List of DataSources
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
        Input:
            .csv file name
        Returns:
            a tuple with a list of original images  and a list of ground truth images
    """
    if not isinstance(filename, str):
        raise TypeError("Name is not a string")
    with open(filename) as csvfile:
        data_storage = list(csv.DictReader(csvfile, delimiter=";"))
        train_dict, test_dict = train_test_split(data_storage,test_size=0.3,train_size=0.7,random_state=69)
        data,lbl = loadFromDataSources(train_dict)
        return data,lbl,test_dict



if __name__ == "__main__":
    # data, lbl,test_dict = loadCsvFile('img.csv')
    net: UNetX = UNetX()
