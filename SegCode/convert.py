"""
Python VTK version >= 5.0
This script is designed to convert segmentation volume into vtkPolyData surface.
"""
import vtk
import os
import glob
from tqdm import tqdm


def volumesToSurface(volume_path, save_surface_path):
    for file in tqdm(glob.glob(os.path.join(volume_path, "*.mha"))):
        reader = vtk.vtkMetaImageReader()
        reader.SetFileName(file)
        reader.Update()

        # matching cubes: convert volumes into surface
        marchingCubes = vtk.vtkMarchingCubes()
        marchingCubes.SetInputData(reader.GetOutput())
        marchingCubes.ComputeNormalsOn()
        marchingCubes.ComputeGradientsOn()
        marchingCubes.SetValue(0, 127)  # set the threshold to 127
        marchingCubes.Update()

        # # Get the connectivity of each cube by vtkPolyDataConnectivityFilter()
        # connectFilter = vtk.vtkPolyDataConnectivityFilter()
        # connectFilter.SetInputData(marchingCubes.GetOutput())
        # connectFilter.SetExtractionModeToAllRegions()
        # connectFilter.Update()
        #
        # # remove small region surfaces
        # # connectFilter.SetExtractionModeToSpecifiedRegions()
        # # connectFilter.InitializeSpecifiedRegionList()
        # # regions = connectFilter.GetNumberOfExtractedRegions()
        # # for i in range(regions-10):
        # #     connectFilter.AddSpecifiedRegion(i)
        # #     if connectFilter.GetRegionSizes() < 50:
        # #         connectFilter.DeleteSpecifiedRegion(i)
        # # connectFilter.Update()
        #
        # Smooth the surface
        smoothFilter = vtk.vtkSmoothPolyDataFilter()
        smoothFilter.SetInputData(marchingCubes.GetOutput())
        smoothFilter.SetNumberOfIterations(15)
        smoothFilter.SetRelaxationFactor(0.1) ### 0.1
        smoothFilter.FeatureEdgeSmoothingOff()
        smoothFilter.BoundarySmoothingOn()
        smoothFilter.Update()

        # Update normals
        normalsGenerator = vtk.vtkPolyDataNormals()
        normalsGenerator.SetInputData(smoothFilter.GetOutput())
        normalsGenerator.ComputePointNormalsOn()
        normalsGenerator.ComputeCellNormalsOn()
        normalsGenerator.Update()

        # Save the surface
        if not os.path.exists(save_surface_path):
            os.makedirs(save_surface_path)

        surface_basename = os.path.basename(file)[:-4] + ".vtk"
        surface_save_name = os.path.join(save_surface_path, surface_basename)
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(surface_save_name)
        writer.SetInputData(normalsGenerator.GetOutput())
        # writer.SetInputData(marchingCubes.GetOutput())
        writer.Write()


if __name__ == '__main__':
    volume_path = "/home/imed/segmentation/MICCAI_code"
    save_suface_path = "/home/imed/segmentation/MICCAI_code"
    volumesToSurface(volume_path, save_suface_path)
