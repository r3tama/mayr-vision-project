import cv2
import numpy as np
import csv
from typing import List, Any
from nptyping import NDArray


Image = NDArray[(Any, Any, 3), Int]
ImageSeg = NDArray[(Any, Any, 3), Int]



def loadCsvFile(filename: Str): -> List[Image, ImageSeg]
   data = []
   lbl = []
   with open(filename) as csvfile:
       reader = csv.DictReader(csvfile, delimiter=";")
       for row in reader:
           try:
               auxData = cv2.imread(row["data"])
               auxLbl = cv2.imread(row["label"])
               if auxData is not None:
                   data.append(auxData)
                if auxLbl is not None:
                    lbl.append(auxLbl)
            except ValueError:
                error = 1
    return data, lbl
