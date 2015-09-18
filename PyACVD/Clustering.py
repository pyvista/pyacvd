# -*- coding: utf-8 -*-
"""ACVD Clustering module

This module takes a vtk surface mesh (vtkPolyData) surface and returns a
uniformly meshed surface also as a vtkPolyData.  It is based on research by:
S. Valette, and J. M. Chassery in
Approximated Centroidal Voronoi Diagrams for Uniform Polygonal Mesh Coarsening
http://www.creatis.insa-lyon.fr/site/en/acvd

Much of this code was translated from the C++ source code available on the
above website.  Cython was used as much of the remeshing process is of an
iterative nature.  This is currently a work in progress.


Example:
    from PyACVD import Clustering
    
    # Load mesh from file.  One method would be to load a *.stl file using:
    
    filename = 'file.stl'
    stlReader = vtk.vtkSTLReader() 
    stlReader.SetFileName(filename) 
    stlReader.Update()
    mesh = stlReader.GetOutput()
    
    # Create clustering object
    cobj = Clustering.Cluster(target)

    # Generate clusters
    cobj.GenClusters(10000)
    
    # Generate uniform mesh
    cobj.GenMesh()

    # Get mesh
    remesh = cobj.ReturnNewMesh()
    
    
    # The clustered original mesh and new mesh can be viewed with:
    cobj.PlotClusters()   # must run cobj.GenClusters first
    cobj.PlotRemesh()     # must run cobj.GenMesh first


Restrictions: 
    vtkPolyData mesh should not contain duplicate points (i.e. adjcent faces
    should share identical points).  If not already done so, clean the mesh
    using "vtk.vtkCleanPolyData()"
    
    The number of resulting points is limited by the available memory of the
    host computer.  If approaching the upper limit of your available memory,
    reduce the "subratio" option when generating the mesh.  As it will be
    pointed out below, the coarser the mesh, the less accurate the solution.
    
    The input mesh should be composed of one surface.  Unexpected behavior
    may result from a multiple input meshes, though some testing has shown
    that it is stable.
    
    Holes in the input mesh may not be filled by the module and will result in
    a non-manifold output.


Options:
    See individual modules for available options


Known bugs:
    - Cluster sizes are highly dependent on initial cluster placement.
    - Clusters one face (or point) large will generate highly non-uniform
      meshes.
    
"""

from __future__ import print_function
import vtk
import VTK_Plotting
from vtk.util import numpy_support as VN
import numpy as np
from scipy import sparse
import sys
from math import log, ceil

# Cython module
import Clustering_Cython


class Cluster(object):
    """ Surface clustering routine that can also be used to uniformly remesh a surface mesh """    
    
    def __init__(self, mesh, mode='point'):
        """ Initializes cluster object 
        
        Option "mode" allows user to select a face or point based clustering
        approach.  The point based approach tends to be faster and more accurate.
        
        """
    
        # Check mode input
        if mode not in ['point', 'face']:
            raise Exception("Improper mode input.  Please input 'point' or 'face'")
            
    
        # Load mesh to self
        self.origmesh = mesh
        
        # Store clustering mode
        self.mode = mode
        
        
    def PrepareMesh(self, nclus, subratio, verbose):
        """ Initialize mesh parameters, to include subdividing mesh if necessary"""
        
        # Check if the mesh needs to be subdivided
        if self.mode == 'face':
            n = self.origmesh.GetPolys().GetNumberOfCells()
        else:
            n = self.origmesh.GetPoints().GetNumberOfPoints()
        
        # Required number of points/faces
        nreq = nclus*subratio
        
        if nreq > n:
            
            # Determine the number of subdivisions given that each subdivision generates 4 triangles
            # per single triangle input or two points per triangle
            if self.mode == 'face':
                nsub = int(ceil(log(float(nreq)/n, 4))) # solved given nreq = n*4**nsub
            else:
                nsub = int(ceil(log(float(nreq)/n, 3))) # solved given nreq = n*2**nsub
            
            if verbose:
                print('Subdividing mesh with {:d} subdivision(s)'.format(nsub)); sys.stdout.flush()
            
            # Perform subdivision
            self.mesh = SubdivideMesh(self.origmesh, nsub)
        else:
            self.mesh = self.origmesh
        
        # Extract vertices and faces of mesh
        f = VN.vtk_to_numpy(self.mesh.GetPolys().GetData()).reshape((-1, 4)).astype(np.int32)
        v = VN.vtk_to_numpy(self.mesh.GetPoints().GetData()).astype(np.float)     
        
        # Store faces without padding
        self.fc = f[:, 1:]
        
        
        if self.mode == 'face':
            # Get face areas and centroids
            self.area = GetTriangleArea(v, f)
            self.cent = v.take(self.fc, axis=0).mean(1)
            
        else:
            # Compute the mean of area of the triangles adjcent to a point
            self.area = GetPointWeight(v, f)
            
            # Centers are simply the surface points
            self.cent = v
            
        # Develop face/edge associations or get the points at either edge of unique edges
        self.edgeID = GetEdgeID(self.fc, self.mode)
        self.nclus = nclus
        
        
    def GenClusters(self, nclus, max_iter=10000, subratio=10, verbose=True):
        """ Function to optimally distrubte surface clusters on mesh
        
        Options:
            "max_iter" can be reduced from the detault 10000 to exit trouble
            meshes early
        
            "subratio" is the ratio between the number of items (faces or
            points) on the loaded mesh and the number of requested points
            on the resulting mesh.  The default is 10.  A higher number
            results in a more equal distribution of points on the mesh,
            while a lower number results in faster remeshing.
            
            "verbose" can be disabled to avoid printing progress to screen
        
        """
        
        # ensure subsampling ratio is greater than one
        if subratio < 1:
            subratio = 1
        
        # Prepare mesh and mesh data
        self.PrepareMesh(nclus, subratio, verbose)        

        # Get item neighbors
        nitems = self.area.shape[0]
        neighbors, nneigh = Clustering_Cython.GetNeighborsInterface(self.edgeID, nitems)
        
        # Compute initial clusters
        if verbose:
            print('Computing initial {:d} clusters... '.format(nclus), end='\r')
            clusters = Clustering_Cython.InitClustersInterface(neighbors, nneigh, self.area,
                                                                   nclus)
            print('Done!'.format(nclus), end='\r')
            
        # Optimize cluster positions
        wcent = self.cent*self.area.reshape((-1, 1))
        self.clusters = Clustering_Cython.ClusterOpt(clusters, nclus, self.area, wcent,
                                                      self.edgeID, max_iter, verbose)
                 
        maxtry=5
        i = 0
        # Attempt to eliminate disconnected clusters when assume manifold is set
        while i < maxtry:
            i += 1
            disconclus, cmod = self.GetDisconnected()

            ndis = disconclus.shape[0]
            if ndis == 0:
                break
            if verbose:
                print('{:d} disconnected clusters.  Repairing and restarting optimization...'.format(ndis))
            self.clusters = Clustering_Cython.NullDisconnected(self.clusters, cmod, disconclus,
                                                              self.edgeID)
                                                              
            self.clusters = Clustering_Cython.ClusterOpt(clusters, nclus, self.area, wcent,
                                                               self.edgeID, max_iter, verbose)
        if i == maxtry:
            if verbose:
                print('Exiting repair process early.  Disconnected clusters may still remain')

        self.clusters = np.asarray(self.clusters)

    def GetDisconnected(self):
        
        self.clusters = np.asarray(self.clusters)
        
        # Copy originial clusters and shift by one
        cmod = self.clusters+ 1
        
        # Get the index of a single face for each cluster
        _, idx = np.unique(cmod, return_index=True)
                
        # Make this cluster index negative and grow these negative faces into adjcent clusters
        cmod[idx] = -cmod[idx]
        cmod = Clustering_Cython.GrowNegative(cmod, self.edgeID)
        
        # Determine if there are any disconnected clusters
        remclus, ncount = np.unique(cmod, return_counts=True)
        disconclus = np.ascontiguousarray(remclus[np.logical_and(remclus > 0, 
                                                                 ncount > 1)]).astype(np.int32)
        
        return disconclus, cmod


    def GenMesh(self):
        """ Creates surface mesh from mesh clusters """
        self.remesh, self.storedcenters = CreateMesh(self.fc, self.cent, self.area, self.clusters,
                                             self.edgeID, self.mode)


    def PlotClusters(self):
        """ Plots clusters """
        PlotClusters(self.mesh, self.clusters, self.mode)
        
        
    def PlotRemesh(self):
        """ Plots remeshed surface """
        try:
            self.remesh
        except:
            raise Exception('Mesh not generated yet')
             
        VTK_Plotting.PlotPoly(self.remesh, 'surface')


    def ReturnMesh(self):
        """ Returns simpolified mesh """
        try:
            return self.remesh
        except:
            raise Exception('Mesh not generated yet')              



######################################## PRIVATE FUNCTIONS #########################################
def VertFacetoPoly(new_pt, new_fc):
    """ Creates a vtk polydata object given points and triangular faces """
    
    # Convert points to vtkfloat object
    points = np.vstack(new_pt)
    vtkArray = VN.numpy_to_vtk(np.ascontiguousarray(points), deep=True)#, deep=True) 
    points = vtk.vtkPoints()
    points.SetData(vtkArray) 
    
    # Convert faces to vtk cells object
    ints = np.ones(len(new_fc), 'int')*3
    cells = np.hstack((ints.reshape(-1, 1), np.vstack(new_fc)))
    cells = np.ascontiguousarray(np.hstack(cells).astype('int64'))
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(cells.shape[0], VN.numpy_to_vtkIdTypeArray(cells, deep=True))

    # Create polydata object
    pdata = vtk.vtkPolyData()
    pdata.SetPoints(points)
    pdata.SetPolys(vtkcells)
    
    # Remove duplicate verticies
    clean = vtk.vtkCleanPolyData()
    clean.ConvertPolysToLinesOff()
    clean.ConvertLinesToPointsOff()
    clean.ConvertStripsToPolysOff()
    if vtk.vtkVersion().GetVTKMajorVersion() > 5:
        clean.SetInputData(pdata)
    else:
        clean.SetInput(pdata) 
    clean.Update()

    return clean.GetOutput()


def PlotClusters(mesh, clusters, mode):
    """ Plots clusters """
    
    # Setup color
    nclus = np.unique(clusters).shape[0]
    color = clusters.astype('float')
    for i in range(nclus):
        color[color == i] = np.random.random(1)
        
    # Set color range depending if null clusters exist
    if np.any(color == -1):
        color[color == -1] = -0.25
        rng = [-0.25, 1]
    else:
        rng = [0, 1]
        
    # Plot
    if mode == 'face':
        VTK_Plotting.PlotQualFace(mesh, color, rng)
        
    else: # if point
        VTK_Plotting.PlotQual(mesh, color, rng)


def GetEdgeID(fc, mode):
    """ Return the faces or points associated with each unique edge
    
    This returns a [n x 2] matrix regardless if the faces are duplicate (as in the case of
    boundary edges)
    
    This function assumes you are working with a surface mesh.
    
    """
    # Find boundary faces have at least one edge that is unique
    
    # Get all edges from a cluster
    edgeID = [[0, 1],
              [1, 2],
              [0, 2]]
    
    # Create a list of all edges
    a = fc[:, edgeID].reshape((-1, 2))
    a.sort(1)
    
    # View array as rows
    b = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx, edges, ncount = np.unique(b, return_index=True, return_inverse=True, return_counts=True)
    edges = edges.reshape((-1, 3))
    

    # Return different ids depending on the mode.  If face, return the two faces attached to an edge
    # if point, return the two points attached to each edge.  Both cases will only include unique
    # edges
    if mode=='face': 
    
        # As numpy only returns the first instance of each "find" get the second instance by
        # reversing the order of the row view (b)
        b = b[::-1]
        _, idx2 = np.unique(b, return_index=True)
        
        # Create an array of face numbers
        l = np.arange(fc.shape[0])
        f = np.vstack((l, l, l)).ravel('F')
        f2 = f[::-1] # reverse ordering for idx2
        
        # Get the faces beloning to each edge
        edgeID = np.vstack((f[idx], f2[idx2])).T
        
    else: # if point
        # return edges
        edgeID = a[idx, :]
    
    return np.ascontiguousarray(edgeID).astype(np.int32)
    

def ClusterCentroid(cent, area, clusters):
    """ Computes an area normalized centroid for each cluster """

    # Check if null cluster exists
    null = False
    if np.any(clusters == -1):
        clusters = clusters.copy()
        clusters[clusters == -1] = clusters.max() + 1
        null = True
    
    wval = cent*area.reshape(-1, 1)
    
    cweight = np.bincount(clusters, weights=area)
    
    cval = np.vstack((np.bincount(clusters, weights=wval[:, 0]),
                      np.bincount(clusters, weights=wval[:, 1]),
                      np.bincount(clusters, weights=wval[:, 2])))/cweight
    
    if null:
        cval[:, -1] = np.inf
        
    return cval.T
    
    
def CreateMesh(fc, cent, area, clusters, edgeID, mode):
    """ Generates a mesh given cluster data """
    
    # Compute centroids
    ccent = ClusterCentroid(cent, area, clusters)
    new_pt = []
    new_fc = []
    
    
    if mode == 'face':
        # Create sparse matrix storing the number of adjcent clusters a point has
        a = fc.ravel()
        b = np.vstack((clusters, clusters, clusters)).T.ravel()
        c = np.ones(len(a), dtype='bool')
    else:
        # Create sparse matrix storing the number of adjcent clusters a face has
        rng = np.arange(fc.shape[0]).reshape((-1, 1))
        a = np.hstack((rng, rng, rng)).ravel()
        b = clusters[fc].ravel()
        c = np.ones(len(a), dtype='bool')
    
    boolmatrix = sparse.csr_matrix((c, (a, b)), dtype='bool')
    
    # Find all points/faces with three neighboring clusters.  Each of the three cluster neighbors
    # becomes a point on a triangle
    nadjclus = boolmatrix.sum(1)
    adj = np.array(nadjclus == 3).nonzero()[0]
    idx = boolmatrix[adj, :].nonzero()[1]
    
    # Append these points and faces
    new_pt.append(ccent.take(idx, 0))
    new_fc.append(np.arange(len(idx)).reshape((-1, 3)))
    trinum = len(idx)
    
    # Compute the cluster adjcency matrix
    clusadj = GetClusterAdjcency(clusters, edgeID)
    
    if mode == 'face':
        # for the remaining points with more than three clusters, iterate through the clusters to
        # determine the ideal face indexing
        
        for clevel in range(4, nadjclus.max() + 1):
            f_adj = np.array(nadjclus == clevel).nonzero()[0]
            
            # Get adjcent clusters
            adj = boolmatrix[f_adj, :].nonzero()[1].reshape((-1, clevel))
                
            # Shift adjcency matrix until in order
            for j in range(clevel - 1):
                
                # Shift adjcency matrix until in order
                not_adj = True
                for k in range(clevel - j - 1):
                    # Find clusters in adjcency matrix that are not adjcent
                    not_adj = np.logical_not(np.array(clusadj[adj[:, j], adj[:, j + 1]]).ravel())
    
                    if np.any(not_adj):            
                        # Reorder adjcency matrix
                        adj[not_adj, j + 1:] = np.roll(adj[not_adj, j + 1:], -1, axis=1)
                    else:
                        break
        
            # Directly connect adjcent clusters and create faces
            faces = []
            for i in range(2, clevel):
                faces.append(np.array([0, i - 1, i]))
            faces = np.vstack(faces)
                
            # New faces
            n = adj[:, faces].ravel()
            new_pt.append(ccent[n, :])
            new_fc.append(np.arange(trinum, trinum + len(n)).reshape((-1, 3)))
            trinum += len(n)
                    
    # Convert to numpy arrays
    new_pt = np.vstack(new_pt)
    new_fc = np.vstack(new_fc)
    
    # Create vtk surface     
    surf = VertFacetoPoly(new_pt, new_fc)
    
    return surf, ccent


def GetClusterAdjcency(clusters, facedge):
    """ Creates sparse cluster adjcent matrix """
    
    # Get boundary clusters for adjcent cluster computation
    edgeclus = clusters[facedge]
    bmask = edgeclus[:, 0] != edgeclus[:, 1]
    bclus = edgeclus[bmask]
    
    a = np.hstack((bclus[:, 0], bclus[:, 1]))
    b = np.hstack((bclus[:, 1], bclus[:, 0]))
    c = np.ones(len(a), dtype='bool')
    
    return sparse.csr_matrix((c, (a, b)), dtype='bool')
    

def Gen_uGrid(new_pt, new_fc):
    """ Generates a vtk unstructured grid given points and triangular faces"""    
    ints = np.ones(len(new_fc), 'int')*3
    cells = np.hstack((ints.reshape(-1, 1), np.vstack(new_fc)))
              
    # Generate vtk mesh
    
    # Convert points to vtkfloat object
    points = np.vstack(new_pt)
    vtkArray = VN.numpy_to_vtk(np.ascontiguousarray(points), deep=True)#, deep=True) 
    points = vtk.vtkPoints()
    points.SetData(vtkArray) 
                
    # Convert to vtk arrays
    tritype = vtk.vtkTriangle().GetCellType()*np.ones(len(new_fc), 'int')
    cell_type = np.ascontiguousarray(tritype).astype('uint8')
    cell_type = VN.numpy_to_vtk(cell_type, deep=True)
    
    offset = np.cumsum(np.hstack(ints + 1))
    offset = np.ascontiguousarray(np.delete(np.insert(offset, 0, 0), -1)).astype('int64')  # shift
    offset = VN.numpy_to_vtkIdTypeArray(offset, deep=True)
    
    cells = np.ascontiguousarray(np.hstack(cells).astype('int64'))
    vtkcells = vtk.vtkCellArray()
    vtkcells.SetCells(cells.shape[0], VN.numpy_to_vtkIdTypeArray(cells, deep=True))
    
    
    # Create unstructured grid
    uGrid = vtk.vtkUnstructuredGrid()
    uGrid.SetPoints(points)
    uGrid.SetCells(cell_type, offset, vtkcells)
    
    return uGrid
    
    
def GetPointWeight(v, f):
    """
    Given a vertices and faces of a mesh compute the "weight" of each point based on the areas of
    adjcent triangles
    """
        
    # Compute face area
    farea = GetTriangleArea(v, f)
    
    # Make indices for the points to reference the face area
    rng = np.arange(farea.shape[0]).reshape((-1, 1))
    ids = np.hstack((rng, rng, rng)).ravel()
    
    # Make raveled index to access each point
    fravel = f[:, 1:].ravel()
    
    # Use bincount to get the number of times a face touches a point and the area of that face
    pcount = np.bincount(fravel)/2
    pcount[pcount == 0] = 1
    parea = np.bincount(fravel, weights=farea[ids])/pcount    
    
    return parea


def GetTriangleArea(v, f):
    """Computes area of triangles given a vertex and face array """
    
    # Extract triangle points from mesh
    p1 = v.take(f[:, 1], 0)
    p2 = v.take(f[:, 2], 0)
    p3 = v.take(f[:, 3], 0)

    # Get the lengths of each triangle side
    s1 = (p1 - p2)
    s1 = np.sqrt(np.sum(s1*s1, 1))

    s2 = (p2 - p3)
    s2 = np.sqrt(np.sum(s2*s2, 1))

    s3 = (p1 - p3)
    s3 = np.sqrt(np.sum(s3*s3, 1))    

    # Compute perimiter
    p = (s1 + s2 + s3)/2
    
    # Compute and return area
    return np.sqrt(p*(p - s1)*(p - s2)*(p - s3))
    
    
def SubdivideMesh(mesh, nsub):
    """ Subdivides a VTK mesh """
    if nsub > 3:
        subdivide = vtk.vtkLinearSubdivisionFilter()
    else:
        subdivide = vtk.vtkLoopSubdivisionFilter() # slower, but appears to be smoother
    subdivide.SetNumberOfSubdivisions(nsub)
    if vtk.vtkVersion().GetVTKMajorVersion() > 5:
        subdivide.SetInputData(mesh)
    else:
        subdivide.SetInput(mesh)
    subdivide.Update()
    return subdivide.GetOutput()
