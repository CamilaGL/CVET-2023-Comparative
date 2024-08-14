import SimpleITK as sitk

def numpyToSITK(inputImage):
   return sitk.GetImageFromArray(inputImage)

def SITKToNumpy(inputImage):
   return sitk.GetArrayFromImage(inputImage)
