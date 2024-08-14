"""
Based on the thinning algorithm loop described by Larrabide 2007 
Voxel consideration order is given by the squared euclidean distance to the nearest background voxel
edt: https://github.com/seung-lab/euclidean-distance-transform-3d

Running environment requirements: 

    numpy
    sitk
    edt

"""


import numpy as np
import argparse
import time
import edt
import heapq
from angiographies.utils.iositk import readNIFTIasSITK, writeSITK
from angiographies.utils.formatconversionsitk import numpyToSITK, SITKToNumpy
from angiographies.utils.imageprocessing import binClosing, gaussianSmoothDiscrete
from angiographies.skeletonisation.skeletongraph import binSketoSke
from angiographies.utils.iovtk import writeVTKPolydataasVTP
from angiographies.skeletonisation.polydatamerger import skeToPolyline

# --------- ordered thinning with different criterions ---------------


def binarySegmToBinarySkeleton(npimg, npimgorig = None, thresh=None):
    '''Binary thinning. Order is squared euclidean distance to background.
    img: numpy image
    returns numpy image'''

    if thresh is not None:
        npimg[npimg<=thresh]=0
        npimg[npimg>thresh]=1
        npimg = npimg.astype(np.int8)

    start_time = time.time()
    npimgpadded = np.pad(npimg, 1)
    if npimgorig is None:
        
        weighted = getWeightedImageEuclideanDistanceTransform(npimgpadded)
        print("--- %s seconds ---" % (time.time() - start_time))
        
    else:
        
        minval = np.amin(npimgorig)
        npimgorig = npimgorig - minval + 1
        npimgorigpadded = np.pad(npimgorig, 1)
        
        if npimg.shape == npimgorig.shape: #check segmentation and grayscale images have same dimentions
            weighted = np.where(npimgpadded, npimgorigpadded, 0)
        print("--- %s seconds ---" % (time.time() - start_time))
    orderedThinning(npimgpadded, weighted) #we're editing npimg here!
    print("--- %s seconds ---" % (time.time() - start_time))
    
    distske = np.where(npimgpadded, weighted, 0)

    return distske[1:-1,1:-1,1:-1]


def intersection(lst1, lst2):
    '''Returns the intersection between two lists'''
    temp = set(lst2)
    return [value for value in lst1 if value in temp]


def orderedThinning(img, weightedImg):


    def isBoundary(img, v):
        '''Check if the 26-neighbourhood of v has at least on background voxel'''
        for i in range(-1, 2): 
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if img[v[0]+i,v[1]+j,v[2]+k] == 0:
                        return True
        return False


    def isBackgroundSimple(img, v):
        '''Check if the voxels of *value* in the *neigh* neighbourhood of *v* voxel are a connected component'''

        indexes18 = []
        indexes6 = 0
        #hardcoding neighbourhood extraction to avoid function calling
        for i in range(-1,2): 
            for j in range(-1,2):
                for k in range(-1,2):
                    if (abs(i) + abs(j) + abs(k)) != 0: #excluding center v
                        if (abs(i) + abs(j) + abs(k)) <=2: #can only move in two directions at a time for 18-neigh
                            if img[v[0]+i,v[1]+j,v[2]+k] == 0:
                                indexes18.append((v[0]+i,v[1]+j,v[2]+k)) #we only care about 18-neigh that's background
                                if (abs(i) + abs(j) + abs(k)) <= 1: #can only move in one direction at a time for 6-neigh
                                    indexes6 = indexes6 + 1
        
        if indexes6 == 0: #there are no background in the 6 connected
            return False
        
        else: #checked that part, now check if 18-neigh background is connected (flood fill)
            visit = [indexes18.pop()]
            while len(visit) != 0: #go over all the connected neighbours
                p = visit.pop()
                #print(p)
                for e in indexes18[:]: #check if any of the not visited is a neighbour (iterate over a copy so we can delete elements)
                    #print(e)
                    if max(abs(p[0]-e[0]), abs(p[1]-e[1]), abs(p[2]-e[2])) == 1:
                        visit.append(e)
                        indexes18.remove(e)
            if len(indexes18) == 0: #I visited all the background voxels in one connected pass
                return True
            else:
                return False



    def isForegroundSimple(img, v):
        '''Check if the voxels of foreground in the 26 neighbourhood of *v* voxel are a connected component
        (which means, the removal of v does not affect the foreground connectivity)'''
        numneigh=0
        indexes=[]
        for i in range(-1, 2): 
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if img[v[0]+i,v[1]+j,v[2]+k] == 1:
                        numneigh=numneigh+1
                        if (abs(i) + abs(j) + abs(k)) != 0: #we're excluding the center point v
                            indexes.append((v[0]+i,v[1]+j,v[2]+k))

        #if no foreground, v is not simple. If only one additional foreground, v is endline. If all neigh are foreground, v is not simple.
        if numneigh<=2 or numneigh == 27: 
            return False
        
        visit = [indexes.pop()] #pick one voxel to study connectivity
        while len(visit) != 0: #go over all the connected neighbours (flood fill)
            p = visit.pop()
            #print(p)
            for e in indexes[:]: #check if any of the not visited is a neighbour (iterate over a copy so we can delete elements in indexes)
                #print(e)
                if max(abs(p[0]-e[0]),abs(p[1]-e[1]),abs(p[2]-e[2])) == 1:
                    visit.append(e)
                    indexes.remove(e)
        if len(indexes) == 0: #I visited all the foreground voxels in one connected pass
            return True
        else:
            return False


    '''Ordered thinning algorithm considering the weightedImg values as order priority.
    Loop implemented according to the Larrabide 2007 paper.
    Optimised with heapq and some function calling reduction'''
    nz = np.nonzero(img)#img != 0
    queued = np.ascontiguousarray(np.where(img!=0, 0, -1)) #no voxel has been queued
    
    markedheap = []
    start_time = time.time()
    innerloop_start_time = time.time()

    for (x,y,z) in zip(*nz): #iterate over nonzero and get redundant boundaries
        if isBoundary(img, (x,y,z)):
            if isForegroundSimple(img, (x,y,z)) and isBackgroundSimple(img, (x,y,z)):
                heapq.heappush(markedheap, (weightedImg[x,y,z],(x,y,z))) #add to heap with priority
                queued[x,y,z] = 1 #mark as queued

    #print("--- selecting redundant boundary %s seconds ---" % (time.time() - innerloop_start_time))
    outerloop_start_time = time.time()
    while markedheap: #heap not empty
        t = heapq.heappop(markedheap)[-1]
        
        if isForegroundSimple(img, t) and isBackgroundSimple(img, t):
            #if not isEndPoint2(img, t):
            img[t] = 0 #deleting
            for i in range(-1,2): 
                for j in range(-1,2):
                    for k in range(-1,2):
                        if (abs(i) + abs(j) + abs(k)) != 0: #hardcoding 26-neigh to avoid function calling?
                            if queued[t[0]+i,t[1]+j,t[2]+k] == 0: #is foreground but not queued
                                if isForegroundSimple(img, (t[0]+i,t[1]+j,t[2]+k)) and isBackgroundSimple(img, (t[0]+i,t[1]+j,t[2]+k)):
                                    heapq.heappush(markedheap, (weightedImg[t[0]+i,t[1]+j,t[2]+k],(t[0]+i,t[1]+j,t[2]+k)))
                                    queued[t[0]+i,t[1]+j,t[2]+k] = 1 #mark as queued
            

    print("--- the complete loop took %s seconds ---" % (time.time() - start_time))



def getWeightedImageEuclideanDistanceTransform(img):
    '''Generate a new image where the foreground voxels are weighted according to the squared euclidean distance
    (relevant for their ordered consideration to perform a binary thinning)'''
    profiled = edt.edtsq(np.ascontiguousarray(img), black_border=True, parallel=1, order="K")
    return profiled


def getWeightedImageEuclidean(img, profiled):
    '''Generate a new image where the foreground voxels are weighted according to the squared euclidean distance
    (relevant for their ordered consideration to perform a binary thinning)'''
    specified = np.nonzero(img) #get nonzero voxels
    for (x,y,z) in zip(*specified): #iterate over nonzero voxels
        profiled[x,y,z] = getMinEuclideanToBackground(img, (x,y,z))
        #profiled[x,y,z] = getProfileMeasure(img, (x,y,z), getMinEuclideanToBackground(img, (x,y,z)))

def getWeightedImageGrayscale(img, original):
    '''Generate a new image where the foreground voxels are assigned their (original) grayscale value
    (relevant for their ordered consideration to perform a binary thinning)'''
    weighted = np.zeros(img.shape, dtype=np.intc)
    specified = np.nonzero(img) #get nonzero voxels in segmentation
    for (x,y,z) in zip(*specified): #iterate over nonzero voxels
        weighted[x,y,z] = original[x,y,z]
    return weighted

def getWeightedProfileMeasure(img, profiled):
    '''Generate a new image where the foreground voxels are weighted according to the profile measure proposed by BABIN Thesis
    (relevant for their ordered consideration to perform a binary thinning)'''
    specified = np.nonzero(img) #get nonzero voxels
    for (x,y,z) in zip(*specified): #iterate over nonzero voxels
        profiled[x,y,z] = getProfileMeasure(img, (x,y,z), getMinEuclideanToBackground(img, (x,y,z)))

def inside_sphere(_x, _y, _z, center, radius):
    '''Check if x,y,z is inside the sphere of *center* and *radius*'''
    x = np.array([_x, _y, _z]) # The point of interest.
    return (np.linalg.norm(x - center) <= radius)

def getProfileMeasure(img, v, delta):
    '''Count the number of foreground voxels included in a sphere of radius square root *delta* and center in v (excluding v)'''
    sqrtdelta = np.sqrt(delta)
    intsqrtdelta = int(np.ceil(sqrtdelta))
    neigh_min, neigh_max = getNeighLimits(img, v, [intsqrtdelta, intsqrtdelta, intsqrtdelta]) #get cube of side 2*ceil(sqrt(delta))
    profilevalue = -1 #instead of zero, to account for the center (voxel of interest) that we have to ignore
    neighbours = np.nonzero(img[neigh_min[0]:neigh_max[0]+1,neigh_min[1]:neigh_max[1]+1,neigh_min[2]:neigh_max[2]+1])
    for i,j,k in zip(*neighbours):
        if inside_sphere(i+neigh_min[0],j+neigh_min[1],k+neigh_min[2],v,sqrtdelta):# and img[i,j,k]==1: #distance to voxel is inside sphere (cube is generated to make stuff easier)
            profilevalue = profilevalue + 1
    return profilevalue

def getNeighLimits(img, v, radius): #v as tuple, radius as list
    '''For certain radius value, get valid indexes in img of a cube with center in v'''
    list_max = [x + y for x, y in zip(v, radius)]
    list_min = [x - y for x, y in zip(v, radius)]
    neigh_min = [max((line_min, 0)) for line_min in list_min]
    neigh_max = [min((line_max, line_shape-1)) for line_max, line_shape in zip(list_max, img.shape)]
    return neigh_min, neigh_max

def stretchMask(img, v, radius): #v as tuple, radius as list
    '''Stretch neighbourhood mask radius and get valid limits inside img with center in v'''
    newradius = [x + 1 for x in radius] 
    neigh_min, neigh_max = getNeighLimits(img, v, newradius)
    return neigh_min, neigh_max, newradius

def getMinEuclideanToBackground(img, v):
    '''Returns the minimal squared Euclidean distance (delta) from foreground voxel v to the background (nearest background voxel)'''
    radius = [0,0,0]
    found = False
    delta = np.dot(np.asarray(img.shape).T,np.asarray(img.shape)) #distance from first to last voxel (max distance between two voxels in this volume)
    while not found: #look for a background voxel
        neigh_min, neigh_max, radius = stretchMask(img, v, radius) #look on a bigger neighbourhood
        neighbourhood = img[neigh_min[0]:neigh_max[0]+1,neigh_min[1]:neigh_max[1]+1,neigh_min[2]:neigh_max[2]+1] #get submatrix corresponding to neighbourhood
        background = np.where(neighbourhood==0) #get background surrounding our voxel
        
        if len(list(zip(*background)))>0: #I have found background
            
            found = True
            for (x,y,z) in zip(*background):
                vector = np.asarray([vox-(back+minbias) for vox, back, minbias in zip(v,[x,y,z],neigh_min)]) #get distance vector
                
                newdelta = np.dot(vector.T, vector) #Euclidean distance squared
                if newdelta < delta: delta = newdelta #keep min distance, I don't need to keep the voxel indexes
            
    return delta


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", help="path to case segmentation", default="", required=True)
    parser.add_argument("-ofile", help="path to output folder+case", default="", required=True)
    parser.add_argument("-gsfile", help="path to original grayscale image", default=None, required=False)
    parser.add_argument("--closing", help="perform morphological closing", action="store_true", required=False, default=False)
    parser.add_argument("-rp", help="repeat closing", default=1, type=int, required=False)
    parser.add_argument("-rad", help="repeat closing", default=2, type=int, required=False)
    parser.add_argument("--gaussian", help="perform gaussian smoothing", action="store_true", required=False, default=True)


    args = parser.parse_args()
    inputf = args.ifile
    outputf = args.ofile
    closing = args.closing
    inputgrayscale = args.gsfile
    rp = args.rp
    rad = args.rad
    gaussian = args.gaussian
    img = readNIFTIasSITK(inputf)
    npimgorig = SITKToNumpy(readNIFTIasSITK(inputgrayscale)) if inputgrayscale is not None else None
    if closing:
        for _ in range(rp):
            img = binClosing(img, int(rad))
    if gaussian:
        img = gaussianSmoothDiscrete(img)
        npimg = SITKToNumpy(img)
        npthinned = binarySegmToBinarySkeleton(npimg, npimgorig, 0.25)
    else:
        npimg = SITKToNumpy(img)
        npthinned = binarySegmToBinarySkeleton(npimg, npimgorig)

    ske, graph = binSketoSke(npthinned, inputgrayscale is not None)
    polydata = skeToPolyline(ske)
    writeVTKPolydataasVTP(polydata, outputf)

if __name__ == "__main__":
    main()

