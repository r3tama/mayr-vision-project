#include <stdio.h>
#include <stdlib.h>

int**** imgArrayCreate(int numImg, int imgWidth, int imgHeight){
    int**** ret;
    ret = (int****)calloc(numImg,sizeof(int***));
    for(int i = 0; i < numImg; i++){
        ret[i] = (int***)calloc(imgWidth,sizeof(int**));
        for(int j = 0; j < imgWidth;j++){
            ret[i][j] = (int**)calloc(imgHeight,sizeof(int*));
            for(int k = 0; k < imgHeight; k++){
                ret[i][j][k] = (int*)calloc(1,sizeof(int));
            }
        }
    }
    return ret;
}

void imgArrayDestroy(int**** array, int numImg, int imgWidth, int imgHeight){
    for(int i = 0; i < numImg; i++){
        for(int j = 0; j < imgWidth; j++){
            for(int k = 0; k < imgHeight; k++){
                free(array[i][j][k]);
            }
            free(array[i][j]);
        }
        free(array[i]);
    }
    free(array);
}

int remap(int* pixel){
    if ((pixel[0] == 0) && (pixel[1] == 0) && (pixel[2] == 0))
        return 0;
    /*if ((pixel[0] == 1) && (pixel[0] == 2) && (pixel[0] == 3))*/
        /*return 1;*/
    // default case
    return 0;
}

int**** rgb2oneDimLabel(int**** img, int numImg, int imgWidth, int imgHeight)
{
    int**** imgBin = imgArrayCreate(numImg,imgWidth,imgHeight);
    for(int i = 0; i < numImg; i++)
    {
        for(int j = 0; j < imgWidth; j++)
        {
            for(int k = 0; k < imgHeight; k++)
            {
                imgBin[i][j][k][0] = remap(img[i][j][k]);
            }
        }
    }

    return imgBin;
}
