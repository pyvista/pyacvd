import vtk
from vtk.util import numpy_support as VN
import numpy as np
import colorsys
    
def PlotCurvature(mesh, curvtype):
    """ Plots curvature of a mesh """
    
    # Curvatures Filter
    curvefilter = vtk.vtkCurvatures()
    
    curvefilter.SetInput(mesh)
    
    if curvtype == 'Mean':
        curvefilter.SetCurvatureTypeToMean()
    elif curvtype == 'Gaussian': 
        curvefilter.SetCurvatureTypeToGaussian()
    elif curvtype == 'Maximum':
        curvefilter.SetCurvatureTypeToMaximum()
    else:
        curvefilter.SetCurvatureTypeToMinimum()

    # Get curves
    curvefilter.Update()
    
    # Mapper
    mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        mapper.SetInputData(curvefilter.GetOutput())
    else:
        mapper.SetInput(curvefilter.GetOutput())
    
    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()    
        
    ###############################
    # Display
    ###############################
    
    # Render
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    ren.AddActor(actor)
        
    iren.Initialize()
    renWin.Render()
    iren.Start()


def PlotGrids(grids):
    """ Plots CFD structured grids """

    N = len(grids)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    
    actors = []
    for i in range(len(grids)):
        
        # Create mapper
        mapper = vtk.vtkDataSetMapper()
        if vtk.vtkVersion().GetVTKMajorVersion() >5:
            mapper.SetInput(grids[i])
        else:
            mapper.SetInputData(grids[i])
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().EdgeVisibilityOn()
        actor.GetProperty().SetColor(RGB_tuples[i])
        actor.GetProperty().LightingOff()
        actors.append(actor)
        
    # Add FEM Actor to renderer window
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.8, 0.8, 0.8)
#    ren.SetBackground(1, 1, 1)

    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    # Add actor
    for actor in actors:
        ren.AddActor(actor)
    
    # Add axes
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOrientationMarker( axes )
    widget.SetInteractor( iren )
    widget.SetViewport( 0.0, 0.0, 0.4, 0.4 )
    widget.SetEnabled( 1 )
    widget.InteractiveOn()
        
    # Render
    iren.Initialize()
    renWin.Render()
    iren.Start()

def PlotGrids_wFEM(grids):
    """ Plots CFD structured grids with a single FEM """

    N = len(grids)
    HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
    RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)
    
    actors = []
    for i in range(len(grids)):
        
        # Create mapper
        mapper = vtk.vtkDataSetMapper()
        if vtk.vtkVersion().GetVTKMajorVersion() >5:
            mapper.SetInput(grids[i])
        else:
            mapper.SetInputData(grids[i])
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().EdgeVisibilityOn()
        if i != 0:
            actor.GetProperty().SetRepresentationToWireframe()
            actor.GetProperty().SetColor(RGB_tuples[i])
        actor.GetProperty().LightingOff()
        actors.append(actor)
        
    # Add FEM Actor to renderer window
    ren = vtk.vtkRenderer()
#    ren.SetBackground(0.3, 0.3, 0.3)
    ren.SetBackground(0.8, 0.8, 0.8)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    

    
    # Add actor
    for actor in actors:
        ren.AddActor(actor)
    
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
#    widget.SetOutlineColor( 0.9300, 0.5700, 0.1300 )
    widget.SetOrientationMarker( axes )
    widget.SetInteractor( iren )
    widget.SetViewport( 0.0, 0.0, 0.4, 0.4 )
    widget.SetEnabled( 1 )
    widget.InteractiveOn()
    
    # Render
    iren.Initialize()
    renWin.Render()
    iren.Start()

     

def PlotPoly(mesh, representation='surface'):
    """ Plots vtk unstructured grid or poly object """
    
    # Create mapper
    mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        mapper.SetInputData(mesh)
    else:
        mapper.SetInput(mesh)
    
    # Create Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
            
    if representation == 'wireframe':
        actor.GetProperty().SetRepresentationToWireframe()
    elif representation == 'points':
        actor.GetProperty().SetRepresentationToPoints()
        actor.GetProperty().SetPointSize(5)
    else:
        actor.GetProperty().SetRepresentationToSurface()
    
        
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(1, 1, 1)
    actor.GetProperty().LightingOff()
    
    # Add FEM Actor to renderer window
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.3, 0.3, 0.3)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    # Add actor
    ren.AddActor(actor)
    
    # Render
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
    
def AccuracyPlotter(meshA, meshB, rmax=[]):
    """ Plot mesh accuracy between A and B """
    distfilt = vtk.vtkDistancePolyDataFilter()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        distfilt.SetInputData(0, meshA)
        distfilt.SetInputData(1, meshB)
    else:
        distfilt.SetInput(0, meshA)
        distfilt.SetInput(1, meshB)
    distfilt.ComputeSecondDistanceOn()
    distfilt.Update()
    distvtk = distfilt.GetOutput()
    
    # Make absolute value
    dist = VN.vtk_to_numpy(distvtk.GetPointData().GetScalars())
    dist = np.abs(dist)
    vtkfloat = VN.numpy_to_vtk(np.ascontiguousarray(dist), deep=True)
    distvtk.GetPointData().SetScalars(vtkfloat)
    
    # if range has not been specified
    if not rmax:
        dist = VN.vtk_to_numpy(distvtk.GetPointData().GetScalars())
        rmax =  dist.max()

    # Create lookup table
    look = vtk.vtkLookupTable()
    look.SetHueRange(0.33, 0)
    look.SetTableRange (0, rmax)
    look.SetSaturationRange (1, 1)
    look.SetValueRange (1, 1)
    look.Build()

    # Create mapper
    mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        mapper.SetInputData(distvtk)
    else:
        mapper.SetInput(distvtk)
    mapper.SetScalarRange(0, rmax)
    mapper.SetLookupTable(look)

    # Create Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(1, 1, 1)
    actor.GetProperty().LightingOff()

    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(look)
    scalarBar.SetTitle('Accuracy')
    scalarBar.SetNumberOfLabels(5)    
    
    # Add FEM Actor to renderer window
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.3, 0.3, 0.3)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    # Add actor
    ren.AddActor(actor)
    ren.AddActor(scalarBar)

    # Render
    iren.Initialize()
    renWin.Render()
    renWin.SetWindowName('FEM to STL Accuracy')
    iren.Start()
    
    
    
def PlotQual(mesh, qual, rng):
    """ Plot score """
    import numpy as np
    # Add score to mesh
    
    vtkfloat = VN.numpy_to_vtk(np.ascontiguousarray(qual), deep=True)
    vtkfloat.SetName('Score')
    mesh.GetPointData().AddArray(vtkfloat)
    mesh.GetPointData().SetActiveScalars('Score')
    
    mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        mapper.SetInputData(mesh)
    else:
        mapper.SetInput(mesh)
    mapper.SetScalarRange(rng[0], rng[1])
    mapper.SetScalarModeToUsePointData()

    # Create Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    actor.GetProperty().SetRepresentationToSurface()
        
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(1, 1, 1)
    actor.GetProperty().LightingOff()
    
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(mapper.GetLookupTable())
    scalarBar.SetTitle('Quality')
    scalarBar.SetNumberOfLabels(5)    
    
    # Add FEM Actor to renderer window
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.3, 0.3, 0.3)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    # Add actor
    ren.AddActor(actor)
    ren.AddActor(scalarBar)
    
    # Render
    iren.Initialize()
    renWin.Render()
    iren.Start()

    
def PlotQualFace(mesh, qual, rng, scbar=False):
    """ Plot score """
    
    # Add score to mesh
    vtkfloat = VN.numpy_to_vtk(np.ascontiguousarray(qual), deep=True)
    vtkfloat.SetName('Score')
    mesh.GetCellData().AddArray(vtkfloat)
    mesh.GetCellData().SetActiveScalars('Score')
    
    mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        mapper.SetInputData(mesh)
    else:
        mapper.SetInput(mesh)
    mapper.SetScalarRange(rng[0], rng[1])
    mapper.SetScalarModeToUseCellData()

    # Create Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    actor.GetProperty().SetRepresentationToSurface()
        
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(1, 1, 1)
    actor.GetProperty().LightingOff()
    

    
    # Add FEM Actor to renderer window
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.3, 0.3, 0.3)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    # Add actor
    ren.AddActor(actor)
    
    if scbar:
        scalarBar = vtk.vtkScalarBarActor()
        scalarBar.SetLookupTable(mapper.GetLookupTable())
#        scalarBar.SetTitle('Quality')
        scalarBar.SetNumberOfLabels(5)    
        ren.AddActor(scalarBar)
    
    # Render
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
    
    
    
def PlotEdges(mesh, angle):
    featureEdges = vtk.vtkFeatureEdges()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        featureEdges.SetInputData(mesh)
    else:
        featureEdges.SetInput(mesh)
    featureEdges.FeatureEdgesOn()
    featureEdges.BoundaryEdgesOff()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.SetFeatureAngle(angle)
    
    
    edgeMapper = vtk.vtkPolyDataMapper();
    edgeMapper.SetInputConnection(featureEdges.GetOutputPort());
    
    edgeActor = vtk.vtkActor();
    edgeActor.GetProperty().SetLineWidth(5);
    edgeActor.SetMapper(edgeMapper)
    
    
    mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        mapper.SetInputData(mesh)
    else:
        mapper.SetInput(mesh)

    
    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()    
        
    ###############################
    # Display
    ###############################
    
    # Render
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    ren.AddActor(actor)
    ren.AddActor(edgeActor)
    
        
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
    
    
def PlotBoundaries(mesh):
    featureEdges = vtk.vtkFeatureEdges()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        featureEdges.SetInputData(mesh)
    else:
        featureEdges.SetInput(mesh)
    featureEdges.FeatureEdgesOff()
    featureEdges.BoundaryEdgesOn()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.ManifoldEdgesOff()
    
    
    edgeMapper = vtk.vtkPolyDataMapper();
    edgeMapper.SetInputConnection(featureEdges.GetOutputPort());
    
    edgeActor = vtk.vtkActor();
    edgeActor.GetProperty().SetLineWidth(5);
    edgeActor.SetMapper(edgeMapper)
    
    
    mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        mapper.SetInputData(mesh)
    else:
        mapper.SetInput(mesh)    
    # Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().LightingOff()    
        
    ###############################
    # Display
    ###############################
    
    # Render
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    ren.AddActor(actor)
    ren.AddActor(edgeActor)
    
        
    iren.Initialize()
    renWin.Render()
    iren.Start()
    
    
def Plot_uGridQual(grid):
    """ Plots quality of a unstructured grid while ignoring pyramids and wedges """
    
    # Create quality filter
    qual_filter = vtk.vtkMeshQuality()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        qual_filter.SetInput(grid)
    else:
        qual_filter.SetInputData(grid)    
        
    qual_filter.SetHexQualityMeasureToScaledJacobian()
    qual_filter.SetTetQualityMeasureToScaledJacobian()
    qual_filter.SaveCellQualityOn
    qual_filter.Update()
    qual_out = qual_filter.GetOutput()
    
    # Get quality as array
    qual = VN.vtk_to_numpy(qual_out.GetCellData().GetScalars())
    
    # If unstructured then replace quality of pyramids and wedges wtih nans
    if str(grid)[:5] =='vtkUnstructuredGrid':
        # Get cell types
        celltypes = VN.vtk_to_numpy(qual_out.GetCellTypesArray())
        
        # Set quality of wedges and pyramids to 1
        wedge_pyr_celltypes = [vtk.vtkWedge().GetCellType(), vtk.vtkPyramid().GetCellType()]
        qual[np.in1d(celltypes, wedge_pyr_celltypes)] = 1
        
        # Reinsert quality array back into uGrid
        vtkfloat = VN.numpy_to_vtk(np.ascontiguousarray(qual), deep=True)
        qual_out.GetCellData().SetScalars(vtkfloat)
        
    # otherwise, negate the grid quality
    else:
        qual = -qual
        vtkfloat = VN.numpy_to_vtk(np.ascontiguousarray(qual), deep=True)
        qual_out.GetCellData().SetScalars(vtkfloat)        
    
    ################################# Plotting #################################
    # Create mapper
    mapper = vtk.vtkDataSetMapper()
    if vtk.vtkVersion().GetVTKMajorVersion() >5:
        mapper.SetInput(qual_out)
    else:
        mapper.SetInputData(qual_out)
    mapper.SetScalarRange(np.nanmin(qual), np.nanmax(qual))
    
    # Create Actor
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetRepresentationToSurface()
    actor.GetProperty().EdgeVisibilityOn()
    actor.GetProperty().SetColor(1, 1, 1)
    actor.GetProperty().LightingOff()
    
    # Add FEM Actor to renderer window
    ren = vtk.vtkRenderer()
    ren.SetBackground(0.3, 0.3, 0.3)
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)
    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Allow user to interact
    istyle = vtk.vtkInteractorStyleTrackballCamera()
    iren.SetInteractorStyle(istyle)
    
    # Add surface to display
    ren.AddActor(actor)
    
    # Add scalar bar
    scalarBar = vtk.vtkScalarBarActor()
    scalarBar.SetLookupTable(mapper.GetLookupTable())
    scalarBar.SetTitle('Quality')
    scalarBar.SetNumberOfLabels(5)    
    ren.AddActor(scalarBar)
    
    # Render
    iren.Initialize()
    renWin.Render()
    renWin.SetWindowName('FEM Element Quality')
    iren.Start()
    