"""
Nidus extraction tools:
    - Morphological closing-opening (adapted from CHENOUNE 2019)
    - Skeleton spider location (adapted from BABIN 2018)
        - sphere-based returns sphere mask
        - convex hull-based returns anchor points to get convex hull with other library (not compatible with this environment)

Running environment requirements: 

    numpy
    sitk
    vtk
    json
    scipy


"""

import vtk
import SimpleITK as sitk
import numpy as np
import argparse
import collections
import math
import time
import os
import scipy
from angiographies.utils.iositk import readNIFTIasSITK, writeSITK
from angiographies.utils.iojson import writejson, readjson
from angiographies.utils.iovtk import readVTPPolydata
from angiographies.utils.formatconversionsitk import numpyToSITK, SITKToNumpy
from angiographies.utils.imageprocessing import thresholdImageSITK, getLargestConnected


def inside_sphere(_x, _y, _z, center, radius):
    '''Check if a point falls inside a sphere of center and radius'''
    x = np.array([_x, _y, _z]) # The point of interest.
    return (np.linalg.norm(x - center) < radius)

def sphere(mat, rad, c):
    '''Create sphere of rad radius in c center inside mat matrix
    mat: numpy matrix (z y x)
    rad: scalar
    c: tuple with center coordinates (x y z)'''

    spcenter = [rad, rad, rad] #center of the sphere
    for x in range(rad*2+1):
        for y in range(rad*2+1):
            for z in range(rad*2+1):
                #inside sphere and inside matrix
                if inside_sphere(x, y, z, spcenter, rad) and (z+c[2]-rad<mat.shape[0] and y+c[1]-rad<mat.shape[1] and x+c[0]-rad<mat.shape[2]) \
                        and (z+c[2]-rad>=0 and y+c[1]-rad>=0 and x+c[0]-rad>=0):
                    mat[z+c[2]-rad,y+c[1]-rad,x+c[0]-rad] = 1

def minus(p1, p2):
    return (p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2])

def sum(p1, p2):
    return (p1[0]+p2[0],p1[1]+p2[1],p1[2]+p2[2])

def divT(p1, p2): #divide tuple
    return (round(p1[0]/p2[0]),round(p1[1]/p2[1]),round(p1[2]/p2[2]))

def divS(e, p):#scalar to tuple
    return (round(e/p[0]),round(e/p[1]),round(e/p[2]))

def divTS(p, e):#tuple by scalar
    return (p[0]/e,p[1]/e,p[2]/e)

def mulTS(p, e):#tuple by scalar
    return (p[0]*e,p[1]*e,p[2]*e)

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


def isSpiderCenter(skeleton, v):
    '''Check if vertex is the center of a spider or just connected to many spurious endlines
    skeleton: polydata with lines and points
    v: point id'''

    cellIds = vtk.vtkIdList()
    skeleton.GetPointCells(v, cellIds) #get cells incident
    degree=cellIds.GetNumberOfIds()
    incident = [cellIds.GetId(c) for c in range(degree)]
    numEndlines = 0
    #Peek at the adjacent cells and check if they have endlines.
    for c in incident:
        cellIds2=vtk.vtkIdList()
        if (skeleton.GetPoint(skeleton.GetCell(c).GetPointId(0)) == skeleton.GetPoint(v)): #center is first point, check last point is endline
            skeleton.GetPointCells(skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-1), cellIds2)
            if cellIds2.GetNumberOfIds() <2: #the point only belongs to one cell
                numEndlines=numEndlines+1
        elif (skeleton.GetPoint(skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-1)) == skeleton.GetPoint(v)): #center is last point, check first point is endline
            skeleton.GetPointCells(skeleton.GetCell(c).GetPointId(0), cellIds2)
            if cellIds2.GetNumberOfIds() <2: #the point only belongs to one cell
                numEndlines=numEndlines+1

    #if half my lines are endlines, this isn't an avm spider but a spurious branch end
    if numEndlines>=degree/2:
        return False
    return True


def getAdjacentPoints(skeleton, v):
    '''Creates a list with the points connected to v
    skeleton: polydata with lines and points
    v: point id'''
    adjacent = []
    cellIds = vtk.vtkIdList()
    skeleton.GetPointCells(v, cellIds) #get cells incident
    incident = [cellIds.GetId(c) for c in range(cellIds.GetNumberOfIds())]
    for c in incident:
        if (skeleton.GetPoint(skeleton.GetCell(c).GetPointId(0)) == skeleton.GetPoint(v)): #first point, find next point
            adjacent.append(skeleton.GetPoint(skeleton.GetCell(c).GetPointId(1)))
        elif (skeleton.GetPoint(skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-1)) == skeleton.GetPoint(v)): #last point, find second to last point
            adjacent.append(skeleton.GetPoint(skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-2)))
    return adjacent


def getAdjacentPointsExtended(skeleton, v, arrayname = "Radius"):
    '''Creates a list with the points connected to v, but considering the line extended according to radius | edt
    skeleton: polydata with lines and points
    v: point id'''
    if skeleton.GetPointData().GetArray(arrayname) is not None:
        adjacent = []
        cellIds = vtk.vtkIdList()
        p1 = skeleton.GetPoint(v)
        skeleton.GetPointCells(v, cellIds) #get cells incident
        incident = [cellIds.GetId(c) for c in range(cellIds.GetNumberOfIds())]
        for c in incident:
            if (skeleton.GetPoint(skeleton.GetCell(c).GetPointId(0)) == skeleton.GetPoint(v)): #first point, find next point
                p2id = skeleton.GetCell(c).GetPointId(1)
            elif (skeleton.GetPoint(skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-1)) == skeleton.GetPoint(v)): #last point, find second to last point
                p2id = skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-2)
            p2 = skeleton.GetPoint(p2id)
            lenAB = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(p1, p2))
            vectAB = minus(p2, p1)
            normAB = divTS(vectAB, lenAB)
            extendto = skeleton.GetPointData().GetArray(arrayname).GetValue(p2id)
            p3 = sum(mulTS(normAB, extendto), p2)
            adjacent.append(p3)
        adjacent.append(p1)
        return adjacent
    else:
        return getAdjacentPoints(skeleton, v)


def getFurthestDist(skeleton, v):
    '''skeleton: polydata with lines and points
    v: point id'''
    furthestDist = 0
    cellIds = vtk.vtkIdList()
    skeleton.GetPointCells(v, cellIds) #get cells incident
    incident = [cellIds.GetId(c) for c in range(cellIds.GetNumberOfIds())]
    for c in incident:
        if (skeleton.GetPoint(skeleton.GetCell(c).GetPointId(0)) == skeleton.GetPoint(v)): #first point, find distance to next point
            dist = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(skeleton.GetPoint(skeleton.GetCell(c).GetPointId(1)), skeleton.GetPoint(skeleton.GetCell(c).GetPointId(0))))
            if dist > furthestDist:
                furthestDist = dist
        elif (skeleton.GetPoint(skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-1)) == skeleton.GetPoint(v)):
            dist = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(skeleton.GetPoint(skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-2)), skeleton.GetPoint(skeleton.GetCell(c).GetPointId(skeleton.GetCell(c).GetNumberOfPoints()-1))))
            if dist > furthestDist:
                furthestDist = dist

    return furthestDist

def areAdjacent(skeleton, v1, v2):
    cellIds1 = vtk.vtkIdList()
    cellIds2 = vtk.vtkIdList()
    skeleton.GetPointCells(v1, cellIds1) #get cells incident
    skeleton.GetPointCells(v2, cellIds2) #get cells incident
    v1cells = [cellIds1.GetId(i) for i in range(cellIds1.GetNumberOfIds())]
    commonCells = [cellIds2.GetId(i) for i in range(cellIds2.GetNumberOfIds()) if cellIds2.GetId(i) in v1cells]
    if len(commonCells)>0:
        return True
    return False

def multipleSpheres(imgshape, spheres, origin=(0,0,0), spacing=(1,1,1)):
    '''imgshape: Image shape to create mask.
    spheres: dictionary with tuples where the first element is sphere center and second is sphere radius.
    origin: if sphere coords are in real world, indicate origin to translate to voxels.
    spacing: if sphere coords are in real world, indicate voxel spacing to translate to voxels.'''
    origin=(0,0,0) #origin always 0, because skeleton starts at 0 and vmtk seems to be ignoring the origin value?
    mask = np.zeros(imgshape)
    for k in spheres.keys():
        center = divT(minus(spheres[k][0], origin),spacing)
        rad = divS(spheres[k][1], spacing)
        sphere(mask, rad[0], center)
    return mask

def flood_fill_hull(image): 
    '''image is a binary np matrix
    returns a binary image with 1 on the voxels belonging to the convex hull, and the convex hull'''   
    start_time = time.time()
    points = np.transpose(np.where(image))
    hull = scipy.spatial.ConvexHull(points)
    deln = scipy.spatial.Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape, np.int32)
    out_img[out_idx] = 1
    print("--- %s seconds ---" % (time.time() - start_time))
    return out_img, hull

def convexHull(imgshape, coords, origin=(0,0,0), spacing=(1,1,1)):
    '''image is a binary np matrix
    returns a binary image with true on the voxels that mark the center of the spiders and their adjacent points'''
    #print("convex hull delimiters")
    mask = np.zeros(imgshape, np.int32)
    for k in coords:
        for p in coords[k]:
            pointcoord = (divT(minus(p, origin),spacing))[::-1]
            p1 = min(pointcoord[0],mask.shape[0]-1) #check we're not out of bounds
            p2 = min(pointcoord[1],mask.shape[1]-1)
            p3 = min(pointcoord[2],mask.shape[2]-1)
            mask[p1,p2,p3] = 1
    hull, _ = flood_fill_hull(mask)
    return hull

def bbox_3D(img):
    '''img is a binary np matrix
    returns bounding box limits that contain all nonzero voxels in img'''
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax

def boundingBox(imgshape, coords, origin=(0,0,0), spacing=(1,1,1)):
    '''image is a binary np matrix
    returns a binary image with true on the voxels bounding box containing the center of the spiders and their adjacent points'''
    mask = np.zeros(imgshape, np.int32)
    for k in coords:
        for p in coords[k]:
            pointcoord = (divT(minus(p, origin),spacing))[::-1]
            p1 = min(pointcoord[0],mask.shape[0]-1) #check we're not out of bounds
            p2 = min(pointcoord[1],mask.shape[1]-1)
            p3 = min(pointcoord[2],mask.shape[2]-1)
            mask[p1,p2,p3] = 1
    min_x, max_x, min_y, max_y, min_z, max_z = bbox_3D(mask)
    #print(min_x, max_x, min_y, max_y, min_z, max_z)
    mask = np.zeros(imgshape, np.int32)
    mask[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1] = 1
    return mask

def findAVMSpiders(skeleton, method="spheres", next=0):
    '''skeleton: vtkpolydata with unique points (each location has its own id)
    method: how to report the spiders (spheres: max radius for each sphere center, hull: all adjacent points to get convex hull)'''

    #create dictionary with points that could potentially belong to the avm and count references
    vertexDict = {}
    maxdeg=0
    avm=0
    for i in range(skeleton.GetNumberOfPoints()):  
        cellIds = vtk.vtkIdList()
        skeleton.GetPointCells(i, cellIds) #get cells adjacent to our point
        degree = cellIds.GetNumberOfIds() #how many cells are incident to this point?
        if degree > 5 and isSpiderCenter(skeleton, i): #We consider this a potential avm point
            vertexDict[i] = degree  #I save this one
            if degree > maxdeg:
                avm=i #This one has the highest degree until now
                maxdeg=degree
       
    #get potential avm nodes adjacent to the main node (highest deg) and their point coords
    if len(vertexDict) > next: #Check that we have found a spider center and we have enough to skip (0 if we're not skipping)
        for i in range(next): #skip to next spider center with highest degree
            vertexDict.pop(avm)
            avm = max(vertexDict, key=vertexDict.get)

        avmPoints = {}
        avmPoints[avm] = [skeleton.GetPoint(avm)]#.append(avm)
        for k in vertexDict:
            if areAdjacent(skeleton, avm, k) and k not in avmPoints: #check we haven't added it yet bc we'd have twice the same --- Not necessary anymore
                #avmNodes.append(k)
                avmPoints[k] = [skeleton.GetPoint(k)]

        #find max radius for sphere centered on each node
        if method=="spheres":
            for k in avmPoints:
                avmPoints[k].append(getFurthestDist(skeleton, k)) #return central points and sphere radius
        elif method == "hull" or method == "boundingbox":
            #get all points connected to sphere centers which will allow to find convex hull or bounding box later
            for k in avmPoints:
                avmPoints[k].extend(getAdjacentPointsExtended(skeleton, k)) 
        return avmPoints
    return None #We didn't find a spider
    
def avmMask(imgshape, origin, spacing, skeleton, method="spheres", next=0):
    '''skeleton: vtkpolydata with unique points (each location has its own id)
    method: how to report the spiders (spheres: max radius for each sphere center, hull: all adjacent points to get convex hull)'''
    spiders = findAVMSpiders(skeleton, method, next)
    if spiders is not None: #we found a spider
        if method=="spheres":
            mask = multipleSpheres(imgshape, spiders, origin, spacing)
        elif method == "hull":
            mask = convexHull(imgshape, spiders, origin, spacing)
        elif method == "boundingbox":
            mask = boundingBox(imgshape, spiders, origin, spacing)
        else:
            print("Invalid method")
            return None
        return mask
    return None

def largestSphere(imgshape, origin, spacing, skeleton, next=0):
    '''skeleton: vtkpolydata with unique points (each location has its own id)
    method: how to report the spiders (spheres: max radius for each sphere center, hull: all adjacent points to get convex hull)'''
    spiders = findAVMSpiders(skeleton, "spheres", next)
    maxrad = 0
    if spiders is not None: #we found a spider
        for k in spiders.keys():
            rad = divS(spiders[k][1], spacing)
            if rad[0] > maxrad:
                maxrad = rad[0]
        return maxrad
    return None

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", help="path to segmentation or centerline", default="", required=True)
    parser.add_argument("-ofile", help="path to output mask", default="", required=True)
    parser.add_argument("-rad", help="sphere radius for morphological extraction", type=int, default=30, required=False)
    parser.add_argument("-method", help="nidus extraction method (morphological, spheres, convexhull, boundingbox)", default="", required=True)


    args = parser.parse_args()
    inputf = args.ifile
    outputf = args.ofile
    rad = args.rad
    method = args.method

    if method == "morphological":
        print("morphological spheres")
        img = readNIFTIasSITK(inputf)
        print(img.GetOrigin())
        print(img.GetSize())
        print(SITKToNumpy(img).shape)
        nidus = extractNidusSphere(img, rad)#spheres
        writeSITK(nidus, outputf)
    else:
        img = readVTPPolydata(inputf)
        infodict = readjson(inputf[:inputf.rindex(os.path.sep)+5]+".json") #get json
        numpyshape = infodict["shape"][::-1]
        numpyorigin = (0,0,0)
        numpyspa = infodict["spacing"] if "vmtk" in inputf else (1,1,1)

        mask = avmMask(numpyshape, numpyorigin, numpyspa, img, method)
        if mask is not None: #We only try to write something if we found a spider and such
            masksitk = numpyToSITK(mask.astype(np.int32))
            masksitk.SetOrigin(infodict["origin"])
            masksitk.SetSpacing(infodict["spacing"])
            writeSITK(masksitk, outputf)


if __name__ == "__main__":
    main()
