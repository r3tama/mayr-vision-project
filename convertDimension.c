#include <stdio.h>
#include <stdlib.h>

int32_t* imgArrayCreate(int numImg, int imgWidth, int imgHeight){
    int32_t* ret = (int32_t*)calloc(numImg*imgWidth*imgHeight*1,sizeof(int32_t));
    return ret;
}

void imgArrayDestroy(int32_t* array){
    free(array);
}

int reindex3(int i, int j, int k, int ii, int jj, int kk){
    return i*jj*kk+j*kk+k;
}

int reindex4(int i, int j, int k, int l, int ii, int jj, int kk, int ll){
    return i*jj*kk*ll+j*kk*ll+k*ll+l;
}

int remap(const int32_t* pixel){
    if ((pixel[0] == 0) && (pixel[1] == 0) && (pixel[2] == 0))
        return 0;
    if ((pixel[0] == 128) && (pixel[1] == 64) && (pixel[2] == 128))
        return 1;
    if ((pixel[0] == 0) && (pixel[1] == 76) && (pixel[2] == 130))
        return 2;
    if ((pixel[0] == 0) && (pixel[1] == 102) && (pixel[2] == 0))
        return 3;
    if ((pixel[0] == 87) && (pixel[1] == 103) && (pixel[2] == 112))
        return 4;
    if ((pixel[0] == 168) && (pixel[1] == 42) && (pixel[2] == 28))
        return 5;
    if ((pixel[0] == 30) && (pixel[1] == 41) && (pixel[2] == 28))
        return 6;
    if ((pixel[0] == 89) && (pixel[1] == 50) && (pixel[2] == 0))
        return 7;
    if ((pixel[0] == 35) && (pixel[1] == 142) && (pixel[2] == 107))
        return 8;
    if ((pixel[0] == 70) && (pixel[1] == 70) && (pixel[2] == 70))
        return 9;
    if ((pixel[0] == 156) && (pixel[1] == 102) && (pixel[2] == 102))
        return 10;
    if ((pixel[0] == 12) && (pixel[1] == 228) && (pixel[2] == 254))
        return 11;
    if ((pixel[0] == 12) && (pixel[1] == 148) && (pixel[2] == 254))
        return 12;
    if ((pixel[0] == 153) && (pixel[1] == 153) && (pixel[2] == 190))
        return 13;
    if ((pixel[0] == 153) && (pixel[1] == 153) && (pixel[2] == 153))
        return 14;
    if ((pixel[0] == 96) && (pixel[1] == 22) && (pixel[2] == 255))
        return 15;
    if ((pixel[0] == 0) && (pixel[1] == 51) && (pixel[2] == 102))
        return 16;
    if ((pixel[0] == 150) && (pixel[1] == 143) && (pixel[2] == 9))
        return 17;
    if ((pixel[0] == 32) && (pixel[1] == 11) && (pixel[2] == 119))
        return 18;
    if ((pixel[0] == 0) && (pixel[1] == 51) && (pixel[2] == 51))
        return 19;
    if ((pixel[0] == 190) && (pixel[1] == 250) && (pixel[2] == 190))
        return 20;
    if ((pixel[0] == 146) && (pixel[1] == 150) && (pixel[2] == 112))
        return 21;
    if ((pixel[0] == 115) && (pixel[1] == 135) && (pixel[2] == 2))
        return 22;
    if ((pixel[0] == 0) && (pixel[1] == 0) && (pixel[2] == 255))
        return 23;
    /*if ((pixel[0] == 1) && (pixel[0] == 2) && (pixel[0] == 3))*/
        /*return 1;*/
    // default case
    return 0;
}

int32_t* rgb2oneDimLabel(const int32_t* img, int numImg, int imgWidth, int imgHeight)
{
    int32_t* imgBin = imgArrayCreate(numImg,imgWidth,imgHeight);

    for(int i = 0; i < numImg; i++){
        for(int j = 0; j < imgWidth; j++){
            for(int k = 0; k < imgHeight; k++){
                imgBin[reindex3(i,j,k,numImg,imgWidth,imgHeight)] = remap(&img[reindex4( i, j, k, 0, numImg, imgWidth, imgHeight, 3)]);
            }
        }
    }
    /*printf("prints?\n");*/
    /*printf("[C] N: %d %d %d\n",img[reindex4(0,45,34,0,numImg,imgWidth,imgHeight,3)],img[reindex4(0,45,34,1,numImg,imgWidth,imgHeight,3)],img[reindex4(0,45,34,2,numImg,imgWidth,imgHeight,3)]);*/
    /*printf("[C] N: %d\n",imgBin[reindex3(0,45,34,numImg,imgWidth,imgHeight)]);*/
    return imgBin;
}
