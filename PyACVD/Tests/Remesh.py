""" Generates a non uniform sphere and resamples it """
import vtk
from os.path import dirname, join, realpath
from PyACVD import VTK_Plotting, Clustering

def Bunny():
    """ Remesh a non-uniform mesh of the Stanford Bunny """
    # get location of this file
    pth = dirname(realpath(__file__))
    filename = join(pth, 'StanfordBunny.stl')
    
    # Import STL of the Stanford Bunny
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    mesh = reader.GetOutput()
    
    # Plot original mesh
    VTK_Plotting.PlotPoly(mesh)
    
    # Cluster
    cobj = Clustering.Cluster(mesh)
    cobj.GenClusters(10000, subratio=20, verbose=True) 
    #cobj.PlotClusters()
    
    # Plot new mesh
    cobj.GenMesh()
    cobj.PlotRemesh()
    

def Sphere():
    """ Generate a non-uniform sphere and remesh it"""
    # Create Sphere
    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(0.0, 0.0, 0.0)
    sphere.SetRadius(5.0)
    sphere.SetThetaResolution(50)
    sphere.SetPhiResolution(500)
    sphere.Update()
    
    # Extract mesh
    mesh = sphere.GetOutput()
    
    # Plot mesh
    VTK_Plotting.PlotPoly(mesh)
    
    
    cobj = Clustering.Cluster(mesh)
    cobj.GenClusters(2000, subratio=10, verbose=True) 
    cobj.PlotClusters()
    cobj.GenMesh()
    cobj.PlotRemesh()