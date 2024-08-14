import vtk
try:
    from vtkmodules.util import numpy_support
except ImportError:
    from vtk.util import numpy_support

def vtkToNumpy(inputImage):
    dimensions = inputImage.GetDimensions() # Get Dimensions
    data = inputImage.GetPointData() # get VTK-formatted data
    data = numpy_support.vtk_to_numpy(data.GetArray(0)) # convert VTKdata to numpy data
    dataReshaped = data.reshape((dimensions[2],dimensions[1],dimensions[0])) #reshape data from 1D -> 3D
    return dataReshaped

def NumpyToVTK(inputImage):
    dimensions = inputImage.shape
    dataRavelled = inputImage.ravel()
    dataArray = numpy_support.numpy_to_vtk(dataRavelled, True)
    vtkImage = vtk.vtkImageData()
    vtkImage.GetPointData().SetScalars(dataArray)
    vtkImage.SetDimensions([dimensions[2], dimensions[1], dimensions[0]])
    return vtkImage

