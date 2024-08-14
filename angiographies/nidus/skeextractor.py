"""
Process angiography skeletonisation to perform nidus extraction (adapted from BABIN 2018)

- Skeleton method alternatives:
    - "skeleton": thinning performed according to euclidean distance weighting
    - "vmtk": skeleton obtained with vmtknetworkextraction

- Skeleton spider translation alternatives:
    - "spheres": sphere-based returns sphere mask
    - "hull": convex hull-based returns anchor points to get convex hull with other library (not compatible with this environment)
    - "boundingbox": bounding box that encloses all spider points

Running environment requirements: 

    numpy
    scipy
    skimage
    sitk

"""
import os
import argparse
import time
import numpy as np
from angiographies.utils.iositk import writeSITK, readNIFTIasSITK
from angiographies.utils.iovtk import writeVTKPolydataasVTP, readNIFTIasVTK, readVTPPolydata
from angiographies.utils.formatconversionsitk import numpyToSITK, SITKToNumpy
from angiographies.nidus.nidusextractor import avmMask
from angiographies.utils.overlapimg import overlapMask




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", help="path to segmentation", default="", required=True)
    parser.add_argument("-iske", help="path to vtp with skeleton", default="", required=True)
    parser.add_argument("-method", help="which skeletonisation are we using (skeleton, vmtk)", default="", required=True)
    parser.add_argument("-spider", help="which nidus extraction method based on spiders are we using (boundingbox, spheres, hull)", default="", required=True)
    parser.add_argument("-next", help="discard the n first spiders with max degree", type=int, default=0, required=False)
    parser.add_argument("-ofile", help="path to save nidus mask", default="", required=True)


    args = parser.parse_args()
    segmfile = args.ifile
    method = args.method
    spider = args.spider
    iske = args.iske
    ofile = args.ofile
    next = args.next

    origin = None
    spacing = None
    shape = None


    if os.path.isfile(segmfile): #if we don't have a segmentation file, we're doing nothing

        start_time = time.time()

        if method == "vmtk" or method == "skeleton": #we're doing spiders
            polydata = None
            origin = (0,0,0) #origin is the same for both (vmtk loses origin and skeleton doesn't have real world coordinates)
            img = readNIFTIasSITK(segmfile) #read segmentation to get volume information.
            shape = img.GetSize()[::-1]
            polydata = readVTPPolydata(iske)

            if method == "vmtk":
                spacing = img.GetSpacing()
            else:
                spacing = (1,1,1)
            if spider == "spheres" or spider == "boundingbox" or spider == "hull":
                if polydata is not None:
                    mask = avmMask(shape, origin, spacing, polydata, spider, next) 
                    if mask is not None:
                        masksitk = numpyToSITK(mask.astype(np.int32))
                        nidus = overlapMask(img, masksitk)
                        writeSITK(nidus, ofile)
                    else:
                        print("Couldn't find nidus")
                else:
                    print("No polydata")
            else:
                print("Invalid spider method")

            
        print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
