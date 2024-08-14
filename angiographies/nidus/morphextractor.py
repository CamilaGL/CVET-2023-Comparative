"""
Nidus extraction tools:
    - Morphological closing-opening (adapted from CHENOUNE 2019)

Running environment requirements: 

    numpy
    sitk
    json


"""

import SimpleITK as sitk
import numpy as np
import argparse
import time
from angiographies.utils.formatconversionsitk import numpyToSITK, SITKToNumpy
from angiographies.utils.imageprocessing import thresholdImageSITK, getLargestConnected
from angiographies.utils.iositk import readNIFTIasSITK, writeSITK



def extractNidusSphere(img, radius):
    '''Perform binary closing and opening to extract avm nidus from segmentation
    img: SITKImage
    radius: morphological sphere radius
    returns sitk binary image with nidus voxels True'''
    #diameter=4.66 #ICA diameter in mm
    start_time = time.time()
    npimg = SITKToNumpy(img)
    #binary opening and closing as chenoune2019 does
    vectorRadius = (radius,radius,radius)
    kernel = sitk.sitkBall
    nidusitk = sitk.BinaryMorphologicalClosing(img, vectorRadius, kernel)
    closing = thresholdImageSITK(nidusitk, 1, 1)
    closing.SetOrigin(img.GetOrigin())
    closing.SetSpacing(img.GetSpacing())
    closing.SetDirection(img.GetDirection())

    nidusitk = sitk.BinaryMorphologicalOpening(nidusitk, vectorRadius, kernel)
    nidusitk = thresholdImageSITK(nidusitk, 1, 1)
    nidusitk.SetOrigin(img.GetOrigin())
    nidusitk.SetSpacing(img.GetSpacing())
    nidusitk.SetDirection(img.GetDirection())
    realnidus = npimg * SITKToNumpy(nidusitk)
    realnidusitk = numpyToSITK(realnidus)
    nidusitkconn = getLargestConnected(realnidusitk)
    nidusitkconn.SetOrigin(img.GetOrigin())
    nidusitkconn.SetSpacing(img.GetSpacing())
    nidusitkconn.SetDirection(img.GetDirection())
    print("--- %s seconds ---" % (time.time() - start_time))
    return nidusitkconn

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", help="path to segmentation or centerline", default="", required=True)
    parser.add_argument("-ofile", help="path to output nidus", default="", required=True)
    parser.add_argument("-rad", help="sphere radius for morphological extraction", type=int, default=30, required=False)


    args = parser.parse_args()
    inputf = args.ifile
    outputf = args.ofile
    rad = args.rad

    img = readNIFTIasSITK(inputf)
    masksitk = extractNidusSphere(img, rad)
    masksitk.SetOrigin(img.GetOrigin())
    masksitk.SetSpacing(img.GetSpacing())
    writeSITK(masksitk, outputf)



if __name__ == "__main__":
    main()
