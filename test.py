import cv2
import numpy as np
import csv
from typing import List, Any, Tuple
from nptyping import NDArray


Image = NDArray[(Any, Any, 3), int]
ImageSeg = NDArray[(Any, Any, 3), int]



def loadCsvFile(filename: str) -> Tuple[List[Image], List[ImageSeg]]:
    """
        Function to load original images and ground truth images from .csv file
        Input:
            .csv file name
        Returns:
            a tuple with a list of original images  and a list of ground truth images
    """
    if not isinstance(filename, str):
        raise TypeError("Name is not a string")
        return 
    data = []
    lbl = []
    count=0
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=";")
        for row in reader:
            try:
                auxData = cv2.imread(row["data"])
                auxLbl = cv2.imread(row["label"])
                auxData = cv2.resize(auxData, (480,720))
                auxLbl = cv2.resize(auxLbl, (480,720))
                count +=1
                if auxData is not None:
                    data.append(auxData)
                if auxLbl is not None:
                    lbl.append(auxLbl)
                    print(count)
            except ValueError:
                error = 1
    return data, lbl


if __name__ == "__main__":
    data, lbl = loadCsvFile('img.csv')
