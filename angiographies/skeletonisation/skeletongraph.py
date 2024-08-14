"""
Based on the skeleton algorithm described by BABIN 2018 
Requires binary skeleton/thinning

Running environment requirements: 

    numpy
    sitk
    edt

"""

import numpy as np
import vtk
import argparse
from itertools import chain
import math
import time
from angiographies.utils.iositk import readNIFTIasSITK
from angiographies.utils.iovtk import writeVTKPolydataasVTP, readVTPPolydata
from angiographies.utils.formatconversionsitk import SITKToNumpy


def uniqueLine(graph, idp1, idp2):
    '''Check if these two point ids are already part of an edge.
    graph is vtkPolydata.
    idp1, idp2 are point ids'''
    for l in range(0,graph.GetNumberOfCells()):
        cp1 = graph.GetCell(l).GetPointId(0)
        cp2 = graph.GetCell(l).GetPointId(1)
        if (cp1==idp1 and cp2==idp2) or (cp1==idp2 and cp2==idp1):
            return l
    return None

def uniquePoint(vertexes, point):
    '''Check if this coordinates are already a vertex.
    vertexes is vtkPoints
    point is tuple with coordinates'''
    for p in range(0,vertexes.GetNumberOfPoints()):
        if vtk.vtkMath.Distance2BetweenPoints(point, vertexes.GetPoint(p))==0.0: #if distance equals zero, same point
            return p #not unique
    return None #haven't found point with zero distance, thus -> unique

def samePoint(graph, point1, point2):
    if vtk.vtkMath.Distance2BetweenPoints(graph.GetPoints().GetPoint(point2), graph.GetPoints().GetPoint(point1))==0.0: #if distance equals zero, same point
        return True #samepoint
    return False #haven't found point with zero distance, thus -> unique

def get26connlim(img, v):
    '''Return matrix with 26-connected neighbours (fewer, if central index is on the edge of the original matrix)'''
    #img as np, v as list of indexes
    list_max = [x + y for x, y in zip(v, [1,1,1])]
    list_min = [x - y for x, y in zip(v, [1,1,1])]
    neigh_min = [max((line_min, 0)) for line_min in list_min]
    neigh_max = [min((line_max, line_shape-1)) for line_max, line_shape in zip(list_max, img.shape)]
    return neigh_min, neigh_max

def get26conn(img, v):
    '''Return a submatrix with the 26 neighbourhood of v'''
    neigh_min, neigh_max = get26connlim(img, v)
    return img[neigh_min[0]:neigh_max[0]+1,neigh_min[1]:neigh_max[1]+1,neigh_min[2]:neigh_max[2]+1] #max +1 because it will not be included otherwise!

def classifyVoxel(img, v):
    '''Return the number of foreground neighbours to this voxel'''
    #count 26-connected neighbours in foreground
    #minus one because we don't care about the central voxel
    return np.count_nonzero(get26conn(img, v), axis=None)-1

def getSubgraphPoints(graph, edgelist):
    '''Given a graph (vtkpolydata) and a list of edges that form a connected subgraph, get all the unique points that belong to the subgraph'''
    #this method assumes that all points with the same coordinates also have the same Point ID (which doesn't happen with structure created with VMTK)
    points = []
    for e in edgelist:
        p1 = graph.GetCell(e).GetPointId(0)
        p2 = graph.GetCell(e).GetPointId(1)
        if p1 not in points:
            points.append(p1)
        if p2 not in points:
            points.append(p2)
    return points

def getMedianPoint(graph, pointList):
    '''Given a list of points, compute the median (the one with the min sum of distances to all the rest)'''
    #for each point, sum euclidean distance to all the rest
    #this distance is computed in voxel coordinates, not in real world coordinates. Thus, it only works with isotropic spacing.
    distancesSum = []
    for p in range(0,len(pointList)):
        point1 = graph.GetPoint(pointList[p])
        distanceSum = 0
        for p2 in chain(range(0,p),range(p+1, len(pointList))):
            point2 = graph.GetPoint(pointList[p2])
            distsquared = math.sqrt(vtk.vtkMath.Distance2BetweenPoints(point1, point2))
            distanceSum = distanceSum + distsquared
        distancesSum.append(distanceSum)

    #return the one with min sum
    return np.argmin(distancesSum)


def connectedEdges(graph, e1, e2):
    '''Check if two edges are connected by one vertex inside graph.
    graph: vtkPolydata
    e1, e2: edge ids'''
    #check e1v1 e2v1, e1v2 e2v1, etc
    
    if  samePoint(graph,graph.GetCell(e1).GetPointId(0),graph.GetCell(e2).GetPointId(0)) or \
        samePoint(graph,graph.GetCell(e1).GetPointId(0),graph.GetCell(e2).GetPointId(1)) or \
        samePoint(graph,graph.GetCell(e1).GetPointId(1),graph.GetCell(e2).GetPointId(0)) or \
        samePoint(graph,graph.GetCell(e1).GetPointId(1),graph.GetCell(e2).GetPointId(1)):
            return True
    return False

def belongsToSubgraph(graph, subgraph, edge):
    '''Check if one edge is connected to the subgraph by any of its vertexes.
    graph: vtkPolydata
    subgraph: list of edge ids belonging to graph
    edge: edge id'''
    for e in subgraph:
        if connectedEdges(graph, e, edge):
            return True
    return False

def graphToSkeleton(graph):
    '''Given a graph (vtkPolydata), merge all subgraphs as indicated in BABIN2018'''
    #This code is less time and computationally expensive, since it emulates the flood-fill logic and doesn't repeat an indecent amount of comparisons
    # ----------------------- Should i repeat this in case the modified edges (fused subgraphs) end up forming a new subgraph? (this should not happen)

    #turn N-edges sub-graphs into one N-vertex
    skeleton = vtk.vtkPolyData()
    #get all sets of connected N-edges
    notmarked = [*range(0,graph.GetNumberOfCells())] #get all foreground voxels
    marked = []
    subgraph = []
    deleteableEdges = []
    deleteableVertexes = []
    stillworking = True
    edgetype = graph.GetCellData().GetArray('EdgeType')
    while stillworking:
        stillworking=False
        for n in notmarked[:]:
            if edgetype.GetValue(n) == 1: #i'm only looking at N-edges
                subgraph = []
                marked.append(n)
                notmarked.remove(n)
                while len(marked)!=0:
                    edge = marked.pop()
                    subgraph.append(edge)
                    for p in range(0,2):
                        p1 = graph.GetCell(edge).GetPointId(p)
                        cellIds = vtk.vtkIdList()
                        graph.GetPointCells(p1, cellIds) #get adjacent to p1
                        for cell in range(cellIds.GetNumberOfIds()):
                            if cellIds.GetId(cell) in notmarked: #if it's not there, we've either already got it lined up on our queue or discarded it
                                if edgetype.GetValue(cellIds.GetId(cell)) == 1: #is N vertex
                                    #we don't have it already queued
                                    marked.append(cellIds.GetId(cell)) #add to visit next
                                    notmarked.remove(cellIds.GetId(cell)) #remove from pending
                                else:
                                    notmarked.remove(cellIds.GetId(cell)) #remove from pending, we don't care about this one
                #found all adjacent to this one, now merge with median
                if len(subgraph)>2: #not sure if this is necessary?
                    uniquePoints = getSubgraphPoints(graph,subgraph)
                    if len(uniquePoints) == 1: #maybe i have three N-edges but they meet at one single point --this shouldn't happen thoughhhhhhhh
                        continue
                    medianPointIndex = getMedianPoint(graph, uniquePoints) #index in this list of the medianPoint
                    medianPoint = uniquePoints[medianPointIndex] #get PointId of the medianvertex
                    for p in range(0, len(uniquePoints)): #overwriting medianPoint too, so that the edge changes its type to N-edge.
                        #if vertextype.GetValue(medianPoint) == 1: ------ this isnt needed because all vertexes in a N-edge are N-vertex
                        #find all lines with p as a vertex
                        #replace p for medianPoint
                        cellIds = vtk.vtkIdList()
                        graph.GetPointCells(uniquePoints[p], cellIds)
                        for cell in range(cellIds.GetNumberOfIds()):
                            if edgetype.GetValue(cellIds.GetId(cell)) == 0: #replace only if L-edge (connected to the subgraph but not in the subgraph)
                                if graph.GetCell(cellIds.GetId(cell)).GetPointId(0) == uniquePoints[p]:
                                    graph.ReplaceCellPoint(cellIds.GetId(cell), graph.GetCell(cellIds.GetId(cell)).GetPointId(0), medianPoint)
                                    graph.BuildCells()
                                    graph.BuildLinks()
                                    graph.GetLines().Modified()
                                    continue
                                if graph.GetCell(cellIds.GetId(cell)).GetPointId(1) == uniquePoints[p]:
                                    graph.ReplaceCellPoint(cellIds.GetId(cell), graph.GetCell(cellIds.GetId(cell)).GetPointId(1), medianPoint)
                                    graph.BuildCells()
                                    graph.BuildLinks()
                                    graph.GetLines().Modified()
                                    continue
                    del uniquePoints[medianPointIndex]
                    deleteableVertexes.extend(uniquePoints)
                    deleteableEdges.extend(subgraph)
                stillworking=True
                break
            else:
                notmarked.remove(n) #I don't care about this one, remove it.

    #   this is done at the end because the structure gets modified when deleting a cell
    #   delete all N-edges (actualSubGraphs)
    #   delete all vertexes but the median vertex (uniquePoints MINUS medianPoint)        
    for c in deleteableEdges:
        graph.DeleteCell(c)
    graph.DeleteLinks()
    graph.RemoveDeletedCells()
    graph.BuildCells()
    graph.BuildLinks()
    graph.GetLines().Modified()

    # question: Delete also those items from the type arrays? (doesn't seem to be required)

    #   -----------------------     optional:
    # create polylines with all L-edges between two N-vertexes

    return graph



def binarySkeletonToGraph(img, gray = False):
    '''Given a binary thinning/skeleton, create a graph where voxels are nodes and neighbour voxels are connected by edges,'''
    #binary img in numpy (remember this means the coordinates are zyx and not xyz, so all operations with numpy are in zyx and the ones with vtk are xyz)

    vertexes = vtk.vtkPoints()
    edges = vtk.vtkCellArray()
    graph = vtk.vtkPolyData()
    vertextype = vtk.vtkFloatArray()
    vertextype.SetName('VertexType')
    if not gray:
        edt = vtk.vtkFloatArray()
        edt.SetName('Radius') #save here squared euclidean distance -- THIS IS NOT RADIUS. Array name is set for ease of usage with vmtknetworkextraction
    edgetype = vtk.vtkFloatArray()
    edgetype.SetName('EdgeType')
    graph.SetPoints(vertexes)
    graph.SetLines(edges)
    npoints=0

    specified = np.nonzero(img) #get nonzero voxels

    pointIds = np.empty(img.shape, dtype=np.intc)
    pointIds[:,:,:] = -1
    cantpoints = len(list(zip(*(specified)))) #this is the number of points i'm going to maybe create
    print(cantpoints)
    lineIds = {}#np.empty((cantpoints, cantpoints), dtype=np.intc)


    for (z,y,x) in zip(*specified): #iterate over nonzero voxels 
        vxConn = classifyVoxel(img, [z,y,x]) #classify this voxel

        #consider if it would be better to keep a matrix with all vertexes already created instead of searching the graph?
        if vxConn==0: #if isolated, voxel is not considered a vertex
            continue
        
        #else, check if voxel is vertex already.
        #   if vertex, get id and classify
        #   create lines not already present
        id = pointIds[z,y,x]
        if id == -1: #we haven't assigned an id to this point yet
            #print("a ver si creo uno, no sé")
            id = vertexes.InsertNextPoint([x,y,z]) #create vertex with the correct coordinate order
            vertextype.InsertNextValue(0) #create as L-vertex
            if not gray:
                edt.InsertNextValue(math.sqrt(img[z,y,x]))
            pointIds[z,y,x] = id
            lineIds[id] = {} #initiate a new line dict for this point

        if vxConn == 1 or vxConn>2:
            vertextype.SetValue(id, 1) #mark as N-vertex

        neighbours = np.nonzero(get26conn(img, [z,y,x])) #get nonzero 26-connected voxels, but moved to the origin
        minbias, maxbias = get26connlim(img, [z,y,x]) #move 3x3x3 26-conn mat to the appropriate index

        for (k,j,i) in zip(*neighbours): #for all connected 
            id2 = pointIds[k+minbias[0],j+minbias[1],i+minbias[2]]
            if id2 == -1: #create vertex
                id2 = vertexes.InsertNextPoint([i+minbias[2],j+minbias[1],k+minbias[0]]) #create vertex
                vertextype.InsertNextValue(0) #create as L-vertex
                if not gray:
                    edt.InsertNextValue(math.sqrt(img[k+minbias[0],j+minbias[1],i+minbias[2]]))
                pointIds[k+minbias[0],j+minbias[1],i+minbias[2]] = id2 #saving it on the pointmatrix
                lineIds[id2] = {} #initiate a new line dict for this point
            
            if id != id2: #don't create line with same vertex!
                idline = -1 
                if id2 in lineIds[id]:
                    idline = lineIds[id][id2]
                if idline == -1: #create line
                    newEdge = vtk.vtkLine()
                    newEdge.GetPointIds().SetNumberOfIds(2)
                    newEdge.GetPointIds().SetId(0, id)
                    newEdge.GetPointIds().SetId(1, id2)
                    edgetype.InsertNextValue(1) #create as N-edge
                    idline = graph.GetLines().InsertNextCell(newEdge)
                    graph.BuildCells()
                    graph.BuildLinks()
                    graph.GetLines().Modified()
                    lineIds[id][id2] = idline
                    lineIds[id2][id] = idline
                    
                if vxConn == 2:
                    edgetype.SetValue(idline,0) #if edge is incident to at least one L-vertex, then it turns into L-edge
        
    
    graph.GetPointData().AddArray(vertextype)
    if not gray:
        graph.GetPointData().AddArray(edt)
    graph.GetCellData().AddArray(edgetype)
    return graph




def binSketoSke(inputImage, gray = False):
    '''Given a binary thinning/skeletonisation, create a skeleton consisting of vertexes and edges.
    inputImage: numpy matrix
    returns two vtkpolydata'''
    
    start_time = time.time()
    graph = binarySkeletonToGraph(inputImage, gray)
    print("--- binary skeleton to graph took %s seconds ---" % (time.time() - start_time))

    start_time = time.time()    
    skeleton = graphToSkeleton(graph)
    print("--- graph to skeleton took %s seconds ---" % (time.time() - start_time))

    return skeleton, graph



def graphtoSke(graph):
    '''Given a graph created from a binary thinning/skeletonisation, create a skeleton consisting of vertexes and edges.
    graph: vtkpolydata with graph created from thinning'''
    
    start_time = time.time()    
    skeleton = graphToSkeleton(graph)
    print("--- graph to skeleton took %s seconds ---" % (time.time() - start_time))

    return skeleton


def main():
    '''This script receives the path to a binary image with a skeletonisation (.nii.gz) and turns it into a connected skeleton
    with vertexes and edges, as described by BABIN2018. Optional is the output of the graph without the subgraphs merged.'''

    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", help="path to binary skeleton to convert to graph and skeleton", default="", required=True)
    parser.add_argument("-ifilegraph", help="path to graph to convert to skeleton", default=None, required=False)
    parser.add_argument("-ofile", help="path to output folder+case for skeleton", default="", required=True)
    parser.add_argument("-ofilegraph", help="path to output folder+case for graph", default=None, required=False)
    parser.add_argument("-ofileinfo", help="path to output folder+case for origin and spacing info", default=None, required=False)
    #TO DO: add flag to skeletonise w grayscale instead of checking for "gray" in inputf

    args = parser.parse_args()
    inputf = args.ifile
    outputf = args.ofile
    outputfgraph = args.ofilegraph
    inputfgraph = args.ifilegraph
    gray = True if "gray" in inputf else False

    if inputfgraph is None:
        img = readNIFTIasSITK(inputf)
        npimg = SITKToNumpy(img)
        ske, graph = binSketoSke(npimg, gray)
        if outputfgraph is not None:
            writeVTKPolydataasVTP(graph, outputfgraph)
        writeVTKPolydataasVTP(ske, outputf)
    else:
        graph = readVTPPolydata(inputfgraph)
        skeleton = graphtoSke(graph)
        writeVTKPolydataasVTP(skeleton, outputf)


if __name__ == "__main__":
    main()