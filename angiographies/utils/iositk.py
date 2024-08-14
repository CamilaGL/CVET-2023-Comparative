import SimpleITK as sitk

def readDICOMasSITK(inputDirectory):
    #TO DO
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(inputDirectory)
    reader.SetFileNames(dicom_names)
    reader.ReadImageInformation()

    # Get the sorted file names, opens all files in the directory and reads the meta-information
    # without reading the bulk pixel data
    series_ID = reader.GetMetaData('0020|000e')
    sorted_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(inputDirectory, series_ID)

    # Read the bulk pixel data
    img = sitk.ReadImage(sorted_file_names)

def readNIFTIasSITK(filename):
    return sitk.ReadImage(filename)

def writeSITK(imageSITK, outputFile):
    sitk.WriteImage(imageSITK, outputFile)
