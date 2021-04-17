#include <stdio.h>
int**** rgb2oneDimLabel(int**** img, int numImg, int imgWidht, int imgHeight)
{
    int imgBin[numImg][imgWidht][imgHeight][1] = {};
    for(int i = 0; i < numImg; i++)
    {
        for(int j = 0; j < imgWidth; j++)
        {
            for(int k = 0; k < imgHeight; k++)
            {
                if ((img[i][j][k][0] == 0) && (img[i][j][k][1] == 0) && (img[]i][j][k][2] == 0))
                    imgBin[i][j][k][1] = 0;
            }
        }
    }





}
