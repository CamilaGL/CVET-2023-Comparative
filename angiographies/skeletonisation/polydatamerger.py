"""
Merge polydata of lines into polylines

Running environment requirements: 

    numpy
    vtk

"""

import vtk
import argparse
from angiographies.utils.iovtk import writeVTKPolydataasVTP,readVTPPolydata


def tracePolyline(inputske, vertextype, start, next, linesstillleft, pointsstillleft):
    '''Given two points, trace the polyline we can create until meeting a bifurcation or an endpoint.

    inputske: vtkpolydata with all lines and points
    vertextype: vtkfloatarray with 0 and 1 indicating if a point is link or bifurcation/end point
    start: id of starting point for our polyline
    next: id of point to which we are going next, from the starting point
    linesstillleft: list with lineIds that we still haven't turned to polyline
    pointsstillleft: list of pointIds that we still have to use to create polylines'''
    idList = [start, next]
    upcoming = next
    while vertextype.GetValue(upcoming) == 0:
        cellIds = vtk.vtkIdList()
        inputske.GetPointCells(upcoming, cellIds) #get adjacent to out point
        nextCellId = [cellIds.GetId(cell) for cell in range(cellIds.GetNumberOfIds()) if cellIds.GetId(cell) in linesstillleft]
        adjcell = inputske.GetCell(nextCellId[0])
        linesstillleft.remove(nextCellId[0]) #remove this line from lines left to visit because each line belongs to only one polyline
        adjPoint = adjcell.GetPointId(1) if upcoming == adjcell.GetPointId(0) else adjcell.GetPointId(0)
        idList.append(adjPoint)
        upcoming = adjPoint
    for i in range(1,len(idList)-1):
        pointsstillleft.remove(idList[i]) #remove all middle points because they only belong to one polyline
    return idList


def skeToPolyline(inputske):
    '''Given a vtkpolydata with points and lines, turn all lines into polylines (first and last points are either a bifurcation or an endpoint)
    
    inputske: vtkpolydata'''
    
    inputske.BuildCells()
    inputske.BuildLinks()

    skepoly = vtk.vtkPolyData()
    edges = vtk.vtkCellArray()

    #Flood fill skeleton to create polylines

    notmarked = [*range(0,inputske.GetNumberOfPoints())] #get all non-visited points
    stillleft = [*range(0,inputske.GetNumberOfCells())] #get all non-visited cells
    stillworking = True
    vertextype = inputske.GetPointData().GetArray('VertexType')
    while stillworking:
        stillworking=False
        for n in notmarked[:]:
            if vertextype.GetValue(n) == 1: #i'm only looking at N-vertexes
                notmarked.remove(n)
                vertex = n
                cellIds = vtk.vtkIdList()
                inputske.GetPointCells(vertex, cellIds) #get adjacent to out point
                for cell in range(cellIds.GetNumberOfIds()):
                    adjcell = inputske.GetCell(cellIds.GetId(cell))
                    adjPoint = adjcell.GetPointId(1) if vertex == adjcell.GetPointId(0) else adjcell.GetPointId(0)
                    if cellIds.GetId(cell) in stillleft: #if it's not there, we've either already got it lined up on our queue or discarded it
                        stillleft.remove(cellIds.GetId(cell)) #removing this line because I'm creating a polyline now
                        thisPolyline = tracePolyline(inputske, vertextype, vertex, adjPoint, stillleft, notmarked) #get all point ids for this polyline
                        polyLine = vtk.vtkPolyLine()
                        polyLine.GetPointIds().SetNumberOfIds(len(thisPolyline))
                        for i in range(0, len(thisPolyline)):
                            polyLine.GetPointIds().SetId(i, thisPolyline[i])
                        edges.InsertNextCell(polyLine)
                stillworking=True
                break

    skepoly.SetPoints(inputske.GetPoints())
    skepoly.SetLines(edges)
    for i in range(inputske.GetPointData().GetNumberOfArrays()): #copy all arrays with vertex information
        skepoly.GetPointData().AddArray(inputske.GetPointData().GetAbstractArray(i))
    return skepoly



def main():
    '''This script receives the path to a polydata vtp to merge lines into polylines.
    Assumes there's a unique pointId for each point and also a VertexType array that indicates if point is link or node (endpoint or bifurcation).'''

    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", help="path to binary skeleton to convert to graph and skeleton", default="", required=True)
    parser.add_argument("-ofile", help="path to output folder+case for skeleton", default="", required=True)
    parser.add_argument("--topoly", help="convert skeleton to polylines", action="store_true", required=False, default=False)

    args = parser.parse_args()
    inputf = args.ifile
    outputf = args.ofile
    topoly = args.topoly

    if topoly:
        inputske = readVTPPolydata(inputf)
        polydata = skeToPolyline(inputske)
        writeVTKPolydataasVTP(polydata, outputf)


if __name__ == "__main__":
    main()