"""
vmtknetworkextraction from binary segmentation

Running environment requirements: 

    numpy
    vmtk
    vtk < 9.1 (memory explodes otherwise due to internal representation changes)

"""

import vtk
from vmtk import vmtkscripts
import argparse
import time
from collections import OrderedDict
from angiographies.utils.iovtk import readNIFTIasVTK, writeVTKPolydataasVTP
from angiographies.skeletonisation.networkediting import toUniquePointID


def segmentationToClippedMesh(inputImage, dist):
    '''binary image meshed and clipped to create an opening
    inputImage: vtkImage binary
    dist: distance from bottom to clip'''
    spacing = inputImage.GetSpacing()
    origin = inputImage.GetOrigin()
    filtered = binaryToMesh(inputImage)
    clipped = clipMesh(filtered, spacing, origin, dist)
    return clipped

def binaryToMesh(inputImage):
    '''Create mesh from binary image
    inputImage: vtkImage binary'''
    spacing = inputImage.GetSpacing()
    sigma = (spacing[0] * 1.7, spacing[1] * 1.7, spacing[2] * 1.7)
    imagefilter = vtk.vtkImageGaussianSmooth()
    imagefilter.SetInputData(inputImage)
    imagefilter.SetRadiusFactors(1,1,1)
    imagefilter.SetStandardDeviations(sigma)

    mcubes = vtk.vtkMarchingCubes()
    mcubes.SetInputConnection(imagefilter.GetOutputPort())
    mcubes.ComputeScalarsOff()
    mcubes.ComputeGradientsOff()
    mcubes.ComputeNormalsOff()
    mcubes.SetValue(0, 0.25)

    smoothMesh = vtk.vtkSmoothPolyDataFilter()
    smoothMesh.SetInputConnection(mcubes.GetOutputPort())
    smoothMesh.SetNumberOfIterations(10)
    smoothMesh.SetRelaxationFactor(0.5)

    remesh = vtk.vtkLinearSubdivisionFilter()
    remesh.SetInputConnection(smoothMesh.GetOutputPort())
    remesh.SetNumberOfSubdivisions(2)

    remesh.Update()
    return remesh.GetOutput()


def marchingCubes(inputImage):
    '''inputImage: vtkImage binary'''
    mcubes = vtk.vtkMarchingCubes()
    mcubes.SetInputData(inputImage)
    mcubes.ComputeScalarsOff()
    mcubes.ComputeGradientsOff()
    mcubes.ComputeNormalsOff()
    mcubes.SetValue(0, 0.4)
    mcubes.Update()
    return mcubes.GetOutput()


def clipMesh(inputImage, spacing, origin, offset):
    '''Clip mesh on z plane according to offset
    inputImage: vtk polydata mesh
    spacing: image spacing
    origin: image origin
    offset: z offset to clip mesh'''
    
    
    originPoint = (origin[0], origin[1], origin[2]+offset*spacing[2])
    normal = (0.0,0.0,1.0)
    plane = vtk.vtkPlane()
    plane.SetNormal(normal)
    plane.SetOrigin(originPoint)

    clip = vtk.vtkClipPolyData()
    clip.SetValue(0)
    clip.GenerateClippedOutputOn()
    clip.SetInputData(inputImage)
    clip.SetClipFunction(plane)

    clip.Update()

    return clip.GetOutput()


def vmtkNetwork(input):
    '''input is a polydata mesh
    '''
    extractor = vmtkscripts.vmtkNetworkExtraction()
    extractor.Surface = input
    extractor.Execute()
    return extractor.Network

def getvmtkNetwork(inputImage):
    '''
    inputImage: binary segmentation as vtkImage'''
    start_time = time.time()

    #First create mesh and clip on z plane to create an opening
    clipped = segmentationToClippedMesh(inputImage,20.0)
    
    #Now create network
    network = vmtkNetwork(clipped)
    print("--- %s seconds ---" % (time.time() - start_time))
    return network

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-ifile", help="path to segmentation", default="", required=True)
    parser.add_argument("-ofile", help="path to output", default="", required=True)
    parser.add_argument("-ofileinfo", help="path to output json with origin+spacing", default=None, required=False)

    args = parser.parse_args()
    inputf = args.ifile
    outputf = args.ofile
    outputjson = args.ofileinfo

    img = readNIFTIasVTK(inputf)
    origin = img.GetOrigin()
    spacing = img.GetSpacing()
    network = getvmtkNetwork(img)


    uniquePointIDs = toUniquePointID(network)
    writeVTKPolydataasVTP(uniquePointIDs, outputf)


if __name__ == "__main__":
    main()
