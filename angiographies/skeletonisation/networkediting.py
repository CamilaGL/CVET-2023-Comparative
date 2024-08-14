"""
Some vtk polydata editing tools to work with vmtknetworkextraction result

Running environment requirements: 

    numpy
    vtk

"""

import vtk
import os
import glob
import math
from itertools import chain
import argparse

from angiographies.utils.iovtk import writeVTKPolydataasVTP, readVTPPolydata


def uniquePoint(theCenterlines, pId):
    for p in chain(range(0,pId), range(pId+1,theCenterlines.GetNumberOfPoints())):
        if vtk.vtkMath.Distance2BetweenPoints(theCenterlines.GetPoint(pId), theCenterlines.GetPoint(p))==0.0: #if distance equals zero, same point
            return False #not unique
    return True #haven't found point with zero distance, thus -> unique
            
def isEndline(theCenterlines, clId):
    """Checks whether certain centerline is an endline"""
    #theCenterlines polydata with all centerline data
    #clId centerline Id to see topology
    #TO DO: check centerlines that don't have -1 but with topology number not used elsewhere
    topology = theCenterlines.GetCellData().GetArray(0).GetTuple(clId)
    if topology[0] == -1 or topology[1] == -1:
        return True
    else:
        np = theCenterlines.GetCell(clId).GetNumberOfPoints()
        if uniquePoint(theCenterlines, theCenterlines.GetCell(clId).GetPointId(0)) or uniquePoint(theCenterlines, theCenterlines.GetCell(clId).GetPointId(np-1)):
            return True
    return False

def findAdjacentCenterlinesPoints(theCenterlines, clId):
    #TO DO
    downstreamArray = []
    upstreamArray = []
    numPoints = theCenterlines.GetNumberOfPoints()
    #get first point, last point, set which way is up ¿is this necessary?
    # --------------------- ANALYSIS SHOULD BE DEEPER THAN JUST COMPARING Z
    lp = []
    p1 = theCenterlines.GetCell(clId).GetPointId(0)
    p2 = theCenterlines.GetCell(clId).GetPointId(theCenterlines.GetCell(clId).GetNumberOfPoints()-1)
    if (theCenterlines.GetPoint(p1)[2]<theCenterlines.GetPoint(p2)[2]):
        lp = [p1,p2]
    else:
        lp = [p2,p1]
    for i in range(clId,numPoints):
        if (theCenterlines.GetCellData().GetArray(0).GetTuple(i)[0] == theCenterlines.GetCellData().GetArray(0).GetTuple(clId)[1]) and (theCenterlines.GetCellData().GetArray(0).GetTuple(clId)[1] != -1):
            upstreamArray.append(i)
        elif (theCenterlines.GetCellData().GetArray(0).GetTuple(clId)[0] == theCenterlines.GetCellData().GetArray(0).GetTuple(i)[1]) and (theCenterlines.GetCellData().GetArray(0).GetTuple(clId)[0] != -1):
            downstreamArray.append(i)
        ifdistsquared = vtk.vtkMath.Distance2BetweenPoints(p1, p2)
    return downstreamArray, upstreamArray

def findAdjacentCenterlinesTopology(theCenterlines, clId):
    """Finds centerlines connected to a particular one"""
    downstreamArray = []
    upstreamArray = []
    numCells = theCenterlines.GetNumberOfCells()
    for i in chain(range(0,clId), range(clId+1,numCells)):
        if (theCenterlines.GetCellData().GetArray(0).GetTuple(i)[0] == theCenterlines.GetCellData().GetArray(0).GetTuple(clId)[1]) and (theCenterlines.GetCellData().GetArray(0).GetTuple(clId)[1] != -1):
            upstreamArray.append(i)
        elif (theCenterlines.GetCellData().GetArray(0).GetTuple(clId)[0] == theCenterlines.GetCellData().GetArray(0).GetTuple(i)[1]) and (theCenterlines.GetCellData().GetArray(0).GetTuple(clId)[0] != -1):
            downstreamArray.append(i)
    return downstreamArray, upstreamArray
        
def computeCenterlineLength(theCenterlines, centerlineId):
    #theCenterline is polydata w points and cells
    #centerline is the cell whose length we need
    centerline = theCenterlines.GetCell(centerlineId)
    npoints = centerline.GetNumberOfPoints()
    point0 = theCenterlines.GetPoint(centerline.GetPointId(0))
    absc = 0

    for i in range(1, npoints):
        point1 = theCenterlines.GetPoint(centerline.GetPointId(i))
        distsquared = vtk.vtkMath.Distance2BetweenPoints(point0, point1)
        dist = math.sqrt(distsquared)
        absc = absc + dist
        point0 = point1

    return absc

def computeCenterlineMeanRadius(theCenterlines, centerlineId):
    #theCenterline is polydata w points and cells
    #centerline is the cell whose mean rad we need
    centerline = theCenterlines.GetCell(centerlineId)
    radius = theCenterlines.GetPointData().GetArray('Radius')
    npoints = centerline.GetNumberOfPoints()
    point0 = theCenterlines.GetPoint(centerline.GetPointId(0))
    rad = 0.0

    for i in range(0, npoints):
        rad1 = radius.GetTuple(centerline.GetPointId(i))
        rad = rad + rad1[0]

    return (rad/float(npoints))

def getLowermostPoint(theCenterlines, pts):
    p=pts[0]
    for p1 in pts:
        if (theCenterlines.GetPoint(p1)[2]<theCenterlines.GetPoint(p)[2]):
            p=p1
    return p

def orderLinePoints2(theCenterlines, clId1, clId2):
    pointdict = []
    p1 = theCenterlines.GetCell(clId1).GetPointId(0)
    print("punto 1", theCenterlines.GetPoint(p1))
    p2 = theCenterlines.GetCell(clId1).GetPointId(theCenterlines.GetCell(clId1).GetNumberOfPoints()-1)
    p3 = theCenterlines.GetCell(clId2).GetPointId(0)
    p4 = theCenterlines.GetCell(clId2).GetPointId(theCenterlines.GetCell(clId2).GetNumberOfPoints()-1)
    if vtk.vtkMath.Distance2BetweenPoints(theCenterlines.GetPoint(p2),theCenterlines.GetPoint(p3))==0.0: #first end meets second start
        if (theCenterlines.GetPoint(p1)[2]<theCenterlines.GetPoint(p4)[2]): #start of first is lower than end of second
            for pl1 in range(0,theCenterlines.GetCell(clId1).GetNumberOfPoints()):
                pointdict.append(theCenterlines.GetCell(clId1).GetPointId(pl1))
            for pl2 in range(1,theCenterlines.GetCell(clId2).GetNumberOfPoints()):
                pointdict.append(theCenterlines.GetCell(clId2).GetPointId(pl2))
        else:                                                               #start of first is higher than end of second
            for pl2 in range(theCenterlines.GetCell(clId2).GetNumberOfPoints()-1, 0, -1):
                pointdict.append(theCenterlines.GetCell(clId2).GetPointId(pl2))
            for pl1 in range(theCenterlines.GetCell(clId1).GetNumberOfPoints()-1, -1, -1):
                pointdict.append(theCenterlines.GetCell(clId1).GetPointId(pl1))
    elif vtk.vtkMath.Distance2BetweenPoints(theCenterlines.GetPoint(p4),theCenterlines.GetPoint(p1))==0.0: #second start meets first end
        if (theCenterlines.GetPoint(p3)[2]<theCenterlines.GetPoint(p2)[2]): #start of second is lower than end of first
            for pl2 in range(0,theCenterlines.GetCell(clId2).GetNumberOfPoints()):
                pointdict.append(theCenterlines.GetCell(clId2).GetPointId(pl2))
            for pl1 in range(1,theCenterlines.GetCell(clId1).GetNumberOfPoints()):
                pointdict.append(theCenterlines.GetCell(clId1).GetPointId(pl1))
        else:                                                               #start of second is higher than end of first
            for pl1 in range(theCenterlines.GetCell(clId1).GetNumberOfPoints()-1, 0, -1):
                pointdict.append(theCenterlines.GetCell(clId1).GetPointId(pl1))
            for pl2 in range(theCenterlines.GetCell(clId2).GetNumberOfPoints()-1, -1, -1):
                pointdict.append(theCenterlines.GetCell(clId2).GetPointId(pl2))    
    elif vtk.vtkMath.Distance2BetweenPoints(theCenterlines.GetPoint(p1),theCenterlines.GetPoint(p3))==0.0: #starts meet
        if (theCenterlines.GetPoint(p2)[2]<theCenterlines.GetPoint(p4)[2]): #starts meet and end of first is lower
            for pl1 in range(theCenterlines.GetCell(clId1).GetNumberOfPoints()-1, -1, -1):
                pointdict.append(theCenterlines.GetCell(clId1).GetPointId(pl1))
            for pl2 in range(1,theCenterlines.GetCell(clId2).GetNumberOfPoints()):
                pointdict.append(theCenterlines.GetCell(clId2).GetPointId(pl2))
        else:                                                               #starts meet and end of second is lower
            for pl2 in range(theCenterlines.GetCell(clId2).GetNumberOfPoints()-1, -1, -1):
                pointdict.append(theCenterlines.GetCell(clId2).GetPointId(pl2))
            for pl1 in range(1,theCenterlines.GetCell(clId1).GetNumberOfPoints()):
                pointdict.append(theCenterlines.GetCell(clId1).GetPointId(pl1))
    else:                                                                                                    #ends meet
        if (theCenterlines.GetPoint(p1)[2]<theCenterlines.GetPoint(p3)[2]): #ends meet and start of first is lower
            for pl1 in range(0,theCenterlines.GetCell(clId1).GetNumberOfPoints()-1):
                pointdict.append(theCenterlines.GetCell(clId1).GetPointId(pl1))
            for pl2 in range(theCenterlines.GetCell(clId2).GetNumberOfPoints()-1, -1, -1):
                pointdict.append(theCenterlines.GetCell(clId2).GetPointId(pl2))
        else:                                                               #ends meet and start of second is lower
            for pl2 in range(0,theCenterlines.GetCell(clId2).GetNumberOfPoints()-1):
                pointdict.append(theCenterlines.GetCell(clId2).GetPointId(pl2))
            for pl1 in range(theCenterlines.GetCell(clId1).GetNumberOfPoints()-1, -1, -1):
                pointdict.append(theCenterlines.GetCell(clId1).GetPointId(pl1))
    return pointdict
           
def removeEndlines(theCenterlines):
    centerlinesCellData = theCenterlines.GetCellData()

    topology = theCenterlines.GetCellData().GetArray(0) #get topology array ---------- replace index for 'Topology'
    numCells = theCenterlines.GetNumberOfCells()
    radius = theCenterlines.GetPointData().GetArray(0)#.GetTuple(0))

    oneLength = computeCenterlineLength(theCenterlines, 1)

    deletedIds = []
    mergePoints = []
    for clId in range(0,theCenterlines.GetNumberOfCells()):
        p1 = theCenterlines.GetCell(clId).GetPointId(0)
        p2 = theCenterlines.GetCell(clId).GetPointId(theCenterlines.GetCell(clId).GetNumberOfPoints()-1)
        for point in range(0,theCenterlines.GetNumberOfPoints()): # TO DO: iterate centerlines and check first and last point?
            if point != p1 and point != p2:
                cellid = vtk.vtkIdList()
                theCenterlines.GetPointCells(point, cellid)
                nclId = cellid.GetId(0)
                if vtk.vtkMath.Distance2BetweenPoints(theCenterlines.GetPoint(p1), theCenterlines.GetPoint(point))==0.0:
                    l = computeCenterlineLength(theCenterlines, nclId) #length point to remove
                    r = computeCenterlineMeanRadius(theCenterlines, nclId) #mean radius point to remove
                    rad = radius.GetTuple(p1) #radius my last point
                    if (((1.5*rad[0] > l) or (l < 2 and r<0.04)) 
                        and isEndline(theCenterlines, nclId)): # length is shorter than radius or very short line, and is also endline
                        deletedIds.append(nclId)
                        mergePoints.append(point)
                elif vtk.vtkMath.Distance2BetweenPoints(theCenterlines.GetPoint(p2), theCenterlines.GetPoint(point))==0.0:
                    l = computeCenterlineLength(theCenterlines, nclId) #length point to remove
                    r = computeCenterlineMeanRadius(theCenterlines, nclId) #mean radius point to remove
                    rad = radius.GetTuple(p2) #radius my last point
                    if (((1.5*rad[0] > l) or (l < 2 and r<0.04)) 
                        and isEndline(theCenterlines, nclId)): # length is shorter than radius or very short line, and is also endline
                        deletedIds.append(nclId)
                        mergePoints.append(point)   
                        #check
    list(dict.fromkeys(deletedIds))
    # ------------ remove points too?
    for cl in list(dict.fromkeys(deletedIds)):
        for clp in range(0,theCenterlines.GetCell(cl).GetNumberOfPoints()):
            theCenterlines.DeletePoint(theCenterlines.GetCell(cl).GetPointId(clp))
        theCenterlines.DeleteCell(cl)
    theCenterlines.RemoveDeletedCells()

    print("no rompi esto")
    
    mergePoints = list(dict.fromkeys(mergePoints))
    #-------------- merge lines
    for mp in mergePoints:
        fcl = []
        for rcl in range(0,theCenterlines.GetNumberOfCells()):
            p1 = theCenterlines.GetCell(rcl).GetPointId(0)
            p2 = theCenterlines.GetCell(rcl).GetPointId(theCenterlines.GetCell(rcl).GetNumberOfPoints()-1)
            if vtk.vtkMath.Distance2BetweenPoints(theCenterlines.GetPoint(p1), theCenterlines.GetPoint(mp))==0.0 or vtk.vtkMath.Distance2BetweenPoints(theCenterlines.GetPoint(p2), theCenterlines.GetPoint(mp))==0.0:
                fcl.append(rcl)
        if len(fcl) == 2: #if it's just two, it can be one single line
            print('gonna merge',fcl)
            l1 = computeCenterlineLength(theCenterlines,fcl[0])
            l2 = computeCenterlineLength(theCenterlines,fcl[1])
            lId=0
            # A function that returns the 'absc' value:
            def getAbsc(e):
                return e['absc']
            
            pointdict = orderLinePoints2(theCenterlines, fcl[0], fcl[1])
            print(pointdict)
            newLine = vtk.vtkPolyLine()
            newLine.GetPointIds().SetNumberOfIds(len(pointdict))
            for i in range(0,len(pointdict)):
                newLine.GetPointIds().SetId(i,pointdict[i])#['id'])
            theCenterlines.DeleteCell(fcl[1])
            theCenterlines.DeleteCell(fcl[0])
            theCenterlines.RemoveDeletedCells()
            theCenterlines.GetPolys().InsertNextCell(newLine)
            theCenterlines.BuildCells()
            theCenterlines.BuildLinks()
            theCenterlines.GetPolys().Modified()

    # ------------ create new polydata
    newPoints = vtk.vtkPoints()
    newPolyLines = vtk.vtkCellArray()
    newPolyData = vtk.vtkPolyData()
    newRadius = vtk.vtkFloatArray()
    newRadius.SetName('Radius')
    npoints=0
    for nc in range(0,theCenterlines.GetNumberOfCells()):
        pId=0
        newPolyLine = vtk.vtkPolyLine()
        for np in range(0,theCenterlines.GetCell(nc).GetNumberOfPoints()):
            newPoints.InsertNextPoint(theCenterlines.GetPoint(theCenterlines.GetCell(nc).GetPointId(np)))
            newPolyLine.GetPointIds().SetNumberOfIds(theCenterlines.GetCell(nc).GetNumberOfPoints())
            newPolyLine.GetPointIds().SetId(pId, npoints)
            newRadius.InsertNextValue(radius.GetTuple(theCenterlines.GetCell(nc).GetPointId(np))[0])
            pId=pId+1
            npoints=npoints+1
        newPolyLines.InsertNextCell(newPolyLine)
    newPolyData.SetPoints(newPoints)
    newPolyData.SetLines(newPolyLines)
    newPolyData.GetPointData().AddArray(newRadius)
    return newPolyData

def cleanNetwork(base_path, patient):
    file_list = glob.glob(base_path + patient + '/retry/skeletons/'+patient+'*.vtp')
    for f in file_list:
        fileName = ((f.split(os.path.sep)[-1]).split('.')[-2]).split(patient)[-1]
        print(f, fileName)
        centerline = readVTPPolydata(f) # Read vtp as Polydata
        if centerline.GetNumberOfCells() > 1:
            centerline.BuildLinks()
            newCl = removeEndlines(centerline)
            writeVTKPolydataasVTP(newCl, base_path+patient+'/retry/skeletons/processed/'+patient+fileName+'-cleaned2-merged-sorted-1.vtp')

def toUniquePointID(theCenterlines):
    '''Delete repeated point locations and keep one id per point.
    theCenterlines: vtkPolydata'''
    radius = theCenterlines.GetPointData().GetArray(0)#.GetTuple(0))
    deleteablePoints = []
    vertextype = vtk.vtkFloatArray()
    vertextype.SetName('VertexType')
    for i in range(theCenterlines.GetNumberOfPoints()):
        vertextype.InsertNextValue(0)
    pointDict = {}
    for i in range(theCenterlines.GetNumberOfCells()):  
        for j in [0, theCenterlines.GetCell(i).GetNumberOfPoints()-1]:#only node points
            if theCenterlines.GetPoint(theCenterlines.GetCell(i).GetPointId(j)) in pointDict.keys():
                deleteablePoints.append(theCenterlines.GetCell(i).GetPointId(j)) #we'll need to delete this point later
                theCenterlines.ReplaceCellPoint(i, theCenterlines.GetCell(i).GetPointId(j), pointDict[theCenterlines.GetPoint(theCenterlines.GetCell(i).GetPointId(j))])
                thisrad = radius.GetValue(theCenterlines.GetCell(i).GetPointId(j))
                if thisrad > radius.GetValue(pointDict[theCenterlines.GetPoint(theCenterlines.GetCell(i).GetPointId(j))]): #keep largest radius
                    radius.SetValue(pointDict[theCenterlines.GetPoint(theCenterlines.GetCell(i).GetPointId(j))],thisrad)
            else:
                pointDict[theCenterlines.GetPoint(theCenterlines.GetCell(i).GetPointId(j))] = theCenterlines.GetCell(i).GetPointId(j) #we're keeping this id
                vertextype.SetValue(theCenterlines.GetCell(i).GetPointId(j), 1)
    

    theCenterlines.GetPointData().AddArray(vertextype)

    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(theCenterlines)
    cleaner.Update()
    newCenterlines = cleaner.GetOutput()

    return newCenterlines

def main():
    '''This script receives the path to a polydata vtp to merge lines into polylines.
    Assumes there's a unique pointId for each point and also a VertexType array that indicates if point is link or node (endpoint or bifurcation).'''

    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", help="path to binary skeleton to convert to graph and skeleton", default="", required=True)
    parser.add_argument("-ofile", help="path to output folder+case for skeleton", default="", required=True)
    parser.add_argument("--tounique", help="convert skeleton to polylines", action="store_true", required=False, default=False)

    args = parser.parse_args()
    inputf = args.ifile
    outputf = args.ofile
    tounique = args.tounique

    if tounique:
        input = readVTPPolydata(inputf)
        uniquePointIDs = toUniquePointID(input)
        writeVTKPolydataasVTP(uniquePointIDs, outputf)

if __name__ == "__main__":
    main()
