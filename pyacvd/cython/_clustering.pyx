# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.math cimport sqrt

import numpy as np
cimport numpy as np

import time
import ctypes

from cython.parallel import prange

from libcpp.vector cimport vector
from libcpp cimport bool as bool_t
from libc.stdint cimport int64_t
ctypedef unsigned char uint8


def weight_by_neighbors(double [:, ::1] points, int [:, ::1] neigh, 
                        int [::1] nneigh):
    """ Computes weights given neighbors """
    cdef int i, j, k, nbr
    cdef int neighbors = nneigh.size
    cdef int npoints = points.shape[0]
    cdef double [::1] weights = np.empty(npoints)
    cdef double [:, ::1] wvertex = np.empty((npoints, 3))
    cdef double [::1] pdist = np.empty(3)
    cdef double pt_weight
    cdef double px, py, pz, pdist_x, pdist_y, pdist_z

    for i in range(neighbors):
        if nneigh[i]:
            pt_weight = 0
            px = points[i, 0]
            py = points[i, 1]
            pz = points[i, 2]
            for j in range(nneigh[i]):
                nbr = neigh[i, j]
                pdist_x = px - points[nbr, 0]
                pdist_y = py - points[nbr, 1]
                pdist_z = pz - points[nbr, 2]
                pt_weight += sqrt(pdist_x*pdist_x + pdist_y*pdist_y + pdist_z*pdist_z)

            weights[i] = pt_weight
            wvertex[i, 0] = px*pt_weight
            wvertex[i, 1] = py*pt_weight
            wvertex[i, 2] = pz*pt_weight
        else:
            weights[i] = 0
            wvertex[i, 0] = 0
            wvertex[i, 1] = 0
            wvertex[i, 2] = 0

    if neighbors < npoints:
        for i in range(neighbors, npoints):
            weights[i] = 0
            wvertex[i, 0] = 0
            wvertex[i, 1] = 0
            wvertex[i, 2] = 0

    return np.array(weights), np.array(wvertex)


def fast_cluster(int [:, ::1] neighbors, int [::1] nneigh, 
                 int nclus, double [::1] area, 
                 double [:, ::1] cent, int [:, ::1] edges):
    """ Python interface function for cluster optimization """

    # Initialize clusters
    cdef int npoints = nneigh.shape[0]
    cdef int [::1] clusters = np.empty(npoints, ctypes.c_int)
    clusters [:] = -1
    
    cdef int [:, ::1] items = np.empty((2, npoints), ctypes.c_int)
    init_clusters(clusters, neighbors, nneigh, area, nclus, items)
        
    # Eliminat null clusters by growing existing null clusters
    grow_null(edges, clusters)
    
    # Assign any remaining clusters to 0 (just in case null clusters fails)
    for i in range(npoints):
        if clusters[i] == -1:
            clusters[i] = 0

    nclus = renumber_clusters(clusters, npoints, nclus)
    return np.asarray(clusters), nclus


def cluster(int [:, ::1] neighbors, int [::1] nneigh, 
            int nclus, double [::1] area, 
            double [:, ::1] cent, int [:, ::1] edges, int maxiter, 
            debug=False, int iso_try=10):
    """ Python interface function for cluster optimization """
    cdef int i

    # Initialize clusters
    cdef int npoints = nneigh.shape[0]
    cdef int [::1] clusters = np.empty(npoints, ctypes.c_int)
    clusters [:] = -1
    
    cdef int [:, ::1] items = np.empty((2, npoints), ctypes.c_int)
    if debug:
        tstart = time.time()
        print('Initializing clusters')
    init_clusters(clusters, neighbors, nneigh, area, nclus, items)

    if debug:
        print('Clusters initialized')
        print(time.time() - tstart)
        
    # Eliminat null clusters by growing existing null clusters
    if debug:
        tstart = time.time()
        print('Growing null clusters')
    grow_null(edges, clusters)
    if debug:
        print('Null grown')
        print(time.time() - tstart)
    
    # Assign any remaining clusters to 0 (just in case null clusters fails)
    for i in range(npoints):
        if clusters[i] == -1:
            clusters[i] = 0
            
    # Arrays for cluster centers, masses, and energies
    cdef double [:, ::1] sgamma = np.zeros((nclus, 3))
    cdef double [::1] srho = np.zeros(nclus)
    cdef double [::1] energy = np.empty(nclus)
        
    # Compute initial masses of clusters
    for i in range(npoints):
        srho[clusters[i]] += area[i]
        sgamma[clusters[i], 0] += cent[i, 0]
        sgamma[clusters[i], 1] += cent[i, 1]
        sgamma[clusters[i], 2] += cent[i, 2]
    
    for i in range(nclus):
        energy[i] = (sgamma[i, 0]**2 + \
                     sgamma[i, 1]**2 + \
                     sgamma[i, 2]**2)/srho[i]
    
    if debug:
        print('Energy initialized')
    
    # Count number of clusters
    cdef int [::1] cluscount = np.bincount(clusters).astype(ctypes.c_int)
    
    # Initialize modified array
    cdef uint8 [::1] mod1 = np.empty(nclus, ctypes.c_uint8)
    cdef uint8 [::1] mod2 = np.empty(nclus, ctypes.c_uint8)

    # Optimize clusters
    if debug:
        print('Minimizing energy')
        tstart = time.time()

    minimize_energy(edges, clusters, area, sgamma, cent, srho, cluscount,
                    maxiter, energy, mod1, mod2)

    if debug:
        print('Energy Minimized')
        print(time.time() - tstart)

                   
    # Identify isolated clusters here
    ndisc = null_disconnected(nclus, nneigh, neighbors, clusters)
    cdef int niter = 0
    while ndisc and niter < iso_try:

        if debug:
            print('Isolated cluster iteration {:d}'.format(niter))

        grow_null(edges, clusters)
        if debug:
            print('\tNull Grown')

        # Re optimize clusters
        minimize_energy(edges, clusters, area, sgamma, cent, srho, cluscount,
                       maxiter, energy, mod1, mod2)
        if debug:
            print('\tEnergy Minimized')

        # Check again for disconnected clusters
        for i in range(npoints):
            if clusters[i] == -1:
                clusters[i] = 0
        ndisc = null_disconnected(nclus, nneigh, neighbors, clusters)
        if debug:
            print('\tStill {:d} disconnected clusters'.format(ndisc))

        niter += 1
        
        if ndisc:
            grow_null(edges, clusters)
        
            # Check again for disconnected clusters
            for i in range(npoints):
                if clusters[i] == -1:
                    clusters[i] = 0

    # renumber clusters 0 to n
    nclus = renumber_clusters(clusters, npoints, nclus)
    return np.asarray(clusters), ndisc > 0, nclus


cdef int renumber_clusters(int [::1] clusters, int npoints, int nclus):
    """ renumbers clusters ensuring consecutive indexing """
    cdef uint8 [::1] assigned = np.zeros(nclus, ctypes.c_uint8)
    cdef int [::1] ref_arr = np.empty(nclus, ctypes.c_int)
    cdef int cnum
    cdef int c = 0
    for i in range(npoints):
        cnum = clusters[i]
        if assigned[cnum] == 0:
            assigned[cnum] = 1
            ref_arr[cnum] = c
            c += 1
        clusters[i] = ref_arr[cnum]

    return c


cdef void init_clusters(int [::1] clusters, int [:, ::1] neighbors,
                        int [::1] nneigh, double [::1] area, int nclus,
                        int [:, ::1] items) nogil:
    """ Initialize clusters"""
                               
    cdef double tarea, new_area, carea
    cdef int item
    cdef int i, j, k, checkitem, c, c_prev
    cdef double ctarea
    cdef int lstind = 0
    cdef int npoints = area.shape[0]
    cdef int i_items_new, i_items_old    
    
    # Total mesh size
    cdef double area_remain = 0
    for i in range(npoints):
        area_remain += area[i]
    
    # Assign clsuters
    ctarea = area_remain/nclus
    for i in range(nclus):
        # Get target area and reset current area
        tarea = area_remain - ctarea*(nclus - i - 1)
        carea = 0.0
        
        # Get starting index (the first free face in list)
        i_items_new = 0
        for j in range(lstind, npoints):
            if clusters[j] == -1:
                carea += area[j]
                items[i_items_new, 0] = j
                clusters[j] = i
                lstind = j
                break
        
        if j == npoints:
            break
        
        # While there are new items to be added
        c = 1
        while c:
            
            # reset items
            c_prev = c
            c = 0
            # swtich indices
            if i_items_new == 0:
                i_items_old = 0
                i_items_new = 1
            else:
                i_items_old = 1
                i_items_new = 0            
            
            
            # progressively add neighbors
            for j in range(c_prev):
                checkitem = items[i_items_old, j]
                for k in range(nneigh[checkitem]):
                    item = neighbors[checkitem, k]
                    
                    # check if the face is free
                    if clusters[item] == -1:
                        # if allowable, add to cluster
                        if area[item] + carea < tarea:
                            carea += area[item]
                            clusters[item] = i
                            items[i_items_new, c] = item
                            c += 1

        area_remain -= carea


def edge_id(int [:, ::1] neigh, int [::1] nneigh):
    """
    Convert neighbor connection array to unique edge array

    Parameters
    ----------
    neigh : np.ndarray
        Array containing the neighbors for each point at each row.
        Array is square and -1 indicates not used.

    nneigh : np.ndarray
        Array containing the number of valid connections for each point.

    Returns
    -------
    edges : np.ndarray
        Unique edges.
    """
    cdef int npoints = neigh.shape[0]
    cdef int maxnbr = neigh.shape[1]    
    cdef int i, j, k, ind

    # copy neighbor array
    cdef int [:, ::1] temp_neighbor = np.empty((npoints, maxnbr), dtype=ctypes.c_int)
    for i in range(npoints):
        for j in range(maxnbr):
            temp_neighbor[i, j] = neigh[i, j]

    # Compute maximum possible number of edges
    cdef int maxedge = 0
    for i in range(npoints):
        maxedge += nneigh[i]

    # Generate edgess
    cdef int [:, ::1] temp_arr = np.empty((maxedge, 2), ctypes.c_int)

    cdef int c = 0
    for i in range(npoints):
        for j in range(nneigh[i]):
            if temp_neighbor[i, j] == -1:
                continue
            else:
                ind = neigh[i, j]
                temp_arr[c, 0] = i
                temp_arr[c, 1] = ind
                c += 1

            # remove this index in temporary neighbor array
            for k in range(nneigh[ind]):
                if temp_neighbor[ind, k] == i:
                    temp_neighbor[ind, k] = -1

    return np.asarray(temp_arr)[:c]


cdef void grow_null(int [:, ::1] edges, int [::1] clusters) nogil:
    """ Grow clusters to include null faces """
    cdef int i
    cdef int face_a, face_b, clusA, clusB, nchange     

    nchange = 1
    while nchange > 0:
        nchange = 0
        # Determine edges that share two clusters
        for i in range(edges.shape[0]):
            # Get the two clusters sharing an edge
            face_a = edges[i, 0]
            face_b = edges[i, 1]
            clusA = clusters[face_a]
            clusB = clusters[face_b]        
    
            # Check and immedtialy flip a cluster edge if one is part
            # of the null cluster
            if clusA == -1 and clusB != -1:
                clusters[face_a] = clusB
                nchange += 1
            elif clusB == -1 and clusA != -1:
                clusters[face_b] = clusA
                nchange += 1


def py_grow_null(int [:, ::1] edges, int [::1] clusters):
    """ Grow clusters to include null faces """
    cdef int face_a, face_b, clusA, clusB, nchange, i
    nchange = 1
    while nchange > 0:
        nchange = 0
        # Determine edges that share two clusters
        for i in range(edges.shape[0]):
            # Get the two clusters sharing an edge
            face_a = edges[i, 0]
            face_b = edges[i, 1]
            clusA = clusters[face_a]
            clusB = clusters[face_b]        

            # Check and immedtialy flip a cluster edge if one is part
            # of the null cluster
            if clusA == -1 and clusB != -1:
                clusters[face_a] = clusB
                nchange += 1
            elif clusB == -1 and clusA != -1:
                clusters[face_b] = clusA
                nchange += 1

        
cdef int null_disconnected(int nclus,  int [::1] nneigh,  int [:, ::1] neigh,
                           int [::1] clusters):
    """ Removes isolated clusters """
    cdef int npoints = nneigh.shape[0]
    cdef uint8 [::1] ccheck = np.zeros(nclus, ctypes.c_uint8)
    cdef uint8 [::1] visited = np.zeros(npoints, ctypes.c_uint8)
    cdef uint8 [::1] visited_cluster = np.zeros(nclus, ctypes.c_uint8)
    cdef int [:, ::1] front = np.empty((2, npoints), np.int32)
    cdef int nclus_checked = 0
    cdef int lst_check = 0
    cdef int ind, index, ifound, cur_clus, c, i_front_old, i_front_new, j
    cdef int c_prev
    cdef int i = 0

    while nclus_checked < nclus:

        # seedpoint is first point available that has not been checked
        for i in range(lst_check, npoints):
            # if point and cluster have not been visited
            if not visited[i] and not visited_cluster[clusters[i]]:
                ifound = i
                lst_check = i
                nclus_checked += 1
                break
        
        # restart if reached the end of points
        if i == npoints - 1:
            break
        
        # store cluster data and check that this has been visited
        cur_clus = clusters[ifound]
        visited[ifound] = 1
        visited_cluster[cur_clus] = 1
        
        # perform front expansion
        i_front_new = 0
        front[i_front_new, 0] = ifound
        c = 1 # dummy init to start while loop
        while c > 0:
        
            # reset front
            c_prev = c
            c = 0
            # swtich indices
            if i_front_new == 0:
                i_front_old = 0
                i_front_new = 1
            else:
                i_front_old = 1
                i_front_new = 0

            for j in range(c_prev):
                ind = front[i_front_old, j]
                for i in range(nneigh[ind]):
                    index = neigh[ind, i]
                    if clusters[index] == cur_clus and not visited[index]:
                        front[i_front_new, c] = index
                        c += 1
                        visited[index] = 1
                        

    # Finally, null any points that have not been visited
    cdef ndisc = 0
    for i in range(npoints):
        if not visited[i]:
            clusters[i] = -1
            ndisc += 1
            
    return ndisc
            
        
def minimize_energy(int [:, ::1] edges, int [::1] clusters, double [::1] area,
                    double [:, ::1] sgamma, double [:, ::1] cent, double [::1] srho,
                    int [::1] cluscount, int maxiter, double [::1] energy,
                    uint8 [::1] mod1, uint8 [::1] mod2):
    """ Minimize cluster energy"""
    cdef int face_a, face_b, clusA, clusB
    cdef double areaface_a, centA0, centA1, centA2
    cdef double areaface_b, centB0, centB1, centB2
    cdef double eA, eB, eorig, eAwB, eBnB, eAnA, eBwA
    cdef int nchange = 1   
    cdef int niter = 0
    cdef int nclus = mod1.shape[0]
    cdef int nedge = edges.shape[0]
    cdef int i

    cdef int [1] nchange_arr

    # start all as modified
    for i in range(nclus):
        mod2[i] = 1

    tlast = 0
    while nchange > 0 and niter < maxiter:

        # Reset modification arrays
        for i in range(nclus):
            mod1[i] = mod2[i]
            mod2[i] = 0

        nchange = 0
        for i in range(nedge):
            # Get the two clusters sharing an edge
            face_a = edges[i, 0]
            face_b = edges[i, 1]
            clusA = clusters[face_a]
            clusB = clusters[face_b]

            # If edge shares two different clusters and at least one
            # has been modified since last iteration
            if clusA != clusB and (mod1[clusA] == 1 or mod1[clusB] == 1):
                # Verify that face can be removed from cluster
                if cluscount[clusA] > 1 and cluscount[clusB] > 1:

                    areaface_a = area[face_a]
                    centA0 = cent[face_a, 0]
                    centA1 = cent[face_a, 1]
                    centA2 = cent[face_a, 2]  

                    areaface_b = area[face_b]
                    centB0 = cent[face_b, 0]
                    centB1 = cent[face_b, 1]
                    centB2 = cent[face_b, 2] 

                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]

                    # Energy with both items assigned to cluster A
                    eAwB = ((sgamma[clusA, 0] + centB0)**2 + \
                            (sgamma[clusA, 1] + centB1)**2 + \
                            (sgamma[clusA, 2] + centB2)**2)/(srho[clusA] + areaface_b)

                    eBnB = ((sgamma[clusB, 0] - centB0)**2 + \
                            (sgamma[clusB, 1] - centB1)**2 + \
                            (sgamma[clusB, 2] - centB2)**2)/(srho[clusB] - areaface_b)

                    eA = eAwB + eBnB
    
                    # Energy with both items assigned to clusterB
                    eAnA = ((sgamma[clusA, 0] - centA0)**2 + \
                            (sgamma[clusA, 1] - centA1)**2 + \
                            (sgamma[clusA, 2] - centA2)**2)/(srho[clusA] - areaface_a)

                    eBwA = ((sgamma[clusB, 0] + centA0)**2 + \
                            (sgamma[clusB, 1] + centA1)**2 + \
                            (sgamma[clusB, 2] + centA2)**2)/(srho[clusB] + areaface_a)

                    eB = eAnA + eBwA

                    # select the largest case (most negative)
                    if eA > eorig and eA > eB:
                        mod2[clusA] = 1
                        mod2[clusB] = 1

                        nchange += 1
                        # reassign
                        clusters[face_b] = clusA
                        cluscount[clusB] -= 1
                        cluscount[clusA] += 1
                        
                        # Update cluster A mass and centroid
                        srho[clusA] += areaface_b
                        sgamma[clusA, 0] += centB0
                        sgamma[clusA, 1] += centB1
                        sgamma[clusA, 2] += centB2
                        
                        srho[clusB] -= areaface_b
                        sgamma[clusB, 0] -= centB0
                        sgamma[clusB, 1] -= centB1
                        sgamma[clusB, 2] -= centB2
                        
                        # Update cluster energy
                        energy[clusA] = eAwB
                        energy[clusB] = eBnB

                    # if the energy contribution of both to B is less than the original and to cluster A
                    elif eB > eorig and eB > eA:
                        
                        # Show clusters as modifies
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        nchange += 1
                        
                        # reassign
                        clusters[face_a] = clusB
                        cluscount[clusA] -= 1
                        cluscount[clusB] += 1
                        
                        # Add item A to cluster A
                        srho[clusB] += areaface_a
                        sgamma[clusB, 0] += centA0
                        sgamma[clusB, 1] += centA1
                        sgamma[clusB, 2] += centA2
                        
                        # Remove item A from cluster A
                        srho[clusA] -= areaface_a
                        sgamma[clusA, 0] -= centA0
                        sgamma[clusA, 1] -= centA1
                        sgamma[clusA, 2] -= centA2
                        
                        # Update cluster energy
                        energy[clusA] = eAnA
                        energy[clusB] = eBwA
                                                
                elif cluscount[clusA] > 1:

                    areaface_a = area[face_a]
                    centA0 = cent[face_a, 0]
                    centA1 = cent[face_a, 1]
                    centA2 = cent[face_a, 2]
                    
                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]
                    
                    # Energy with both items assigned to clusterB
                    eAnA = ((sgamma[clusA, 0] - centA0)**2 + \
                            (sgamma[clusA, 1] - centA1)**2 + \
                            (sgamma[clusA, 2] - centA2)**2)/(srho[clusA] - areaface_a)
                           
                    eBwA = ((sgamma[clusB, 0] + centA0)**2 + \
                            (sgamma[clusB, 1] + centA1)**2 + \
                            (sgamma[clusB, 2] + centA2)**2)/(srho[clusB] + areaface_a)
                             
                    eB = eAnA + eBwA
                    
                    # Compare energy contributions
                    if eB > eorig:
                        
                        # Flag clusters as modified
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        nchange += 1
                        
                        # reassign
                        clusters[face_a] = clusB
                        cluscount[clusA] -= 1
                        cluscount[clusB] += 1
                        
                        # Add item A to cluster A
                        srho[clusB] += areaface_a
                        sgamma[clusB, 0] += centA0
                        sgamma[clusB, 1] += centA1
                        sgamma[clusB, 2] += centA2
                        
                        # Remove item A from cluster A
                        srho[clusA] -= areaface_a
                        sgamma[clusA, 0] -= centA0
                        sgamma[clusA, 1] -= centA1
                        sgamma[clusA, 2] -= centA2
                        
                        # Update cluster energy
                        energy[clusA] = eAnA
                        energy[clusB] = eBwA
                        
                        
                elif cluscount[clusB] > 1:                
                    
                    areaface_b = area[face_b]
                    centB0 = cent[face_b, 0]
                    centB1 = cent[face_b, 1]
                    centB2 = cent[face_b, 2]  
                    
                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]
                    
                    # Energy with both items assigned to cluster A
                    eAwB = ((sgamma[clusA, 0] + centB0)**2 + \
                            (sgamma[clusA, 1] + centB1)**2 + \
                            (sgamma[clusA, 2] + centB2)**2)/(srho[clusA] + areaface_b)
                           
                    eBnB = ((sgamma[clusB, 0] - centB0)**2 + \
                            (sgamma[clusB, 1] - centB1)**2 + \
                            (sgamma[clusB, 2] - centB2)**2)/(srho[clusB] - areaface_b)
                            
                    eA = eAwB + eBnB
    
                    # If moving face B reduces cluster energy
                    if eA > eorig:
                        
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        
                        nchange+=1
                        # reassign
                        clusters[face_b] = clusA
                        cluscount[clusB] -= 1
                        cluscount[clusA] += 1
                        
                        # Update cluster A mass and centroid
                        srho[clusA] += areaface_b
                        sgamma[clusA, 0] += centB0
                        sgamma[clusA, 1] += centB1
                        sgamma[clusA, 2] += centB2
                        
                        srho[clusB] -= areaface_b
                        sgamma[clusB, 0] -= centB0
                        sgamma[clusB, 1] -= centB1
                        sgamma[clusB, 2] -= centB2
                        
                        # Update cluster energy
                        energy[clusA] = eAwB
                        energy[clusB] = eBnB
                        
        niter += 1


def weighted_points_double(double [:, ::1] v, int [:, ::1] f,
                           double [::1] additional_weights, return_weighted=True):
    """
    Returns point weight based on area weight and weighted points.
    Points are weighted by adjcent area faces.

    Parameters
    ----------
    v : np.ndarray, np.double
        Point array

    f : np.ndarray, int
        n x 4 face array.  First column is padding and is ignored.

    Returns
    -------
    pweight : np.ndarray, np.double
        Point weight array

    wvertex : np.ndarray, np.double
        Vertices mutlipled by their corresponding weights.
    """
    if f.shape[1] != 4:
        raise Exception('Must be an unclipped vtk face array')

    cdef int nfaces = f.shape[0]
    cdef int npoints = v.shape[0]
    cdef double [::1] pweight = np.zeros(npoints)
    cdef double [::1] farea = np.empty(nfaces)
    cdef double [:, ::1] wvertex = np.empty((npoints, 3))

    cdef double v0_0, v0_1, v0_2
    cdef double v1_0, v1_1, v1_2
    cdef double v2_0, v2_1, v2_2
    cdef double e0_0, e0_1, e0_2
    cdef double e1_0, e1_1, e1_2
    cdef double c0, c1, c2
    cdef double farea_l
    cdef int i

    cdef int point0, point1, point2

    for i in prange(nfaces, nogil=True):
        point0 = f[i, 1]
        point1 = f[i, 2]
        point2 = f[i, 3]

        v0_0 = v[point0, 0]
        v0_1 = v[point0, 1]
        v0_2 = v[point0, 2]

        v1_0 = v[point1, 0]
        v1_1 = v[point1, 1]
        v1_2 = v[point1, 2]

        v2_0 = v[point2, 0]
        v2_1 = v[point2, 1]
        v2_2 = v[point2, 2]

        # Edges
        e0_0 = v1_0 - v0_0
        e0_1 = v1_1 - v0_1
        e0_2 = v1_2 - v0_2
        
        e1_0 = v2_0 - v0_0
        e1_1 = v2_1 - v0_1
        e1_2 = v2_2 - v0_2

        c0 = e0_1*e1_2 - e0_2*e1_1
        c1 = e0_2*e1_0 - e0_0*e1_2
        c2 = e0_0*e1_1 - e0_1*e1_0

        # triangle area
        farea[i] = 0.5*sqrt(c0**2 + c1**2 + c2**2)

    for i in range(nfaces):
        point0 = f[i, 1]
        point1 = f[i, 2]
        point2 = f[i, 3]
        farea_l = farea[i]

        # Store the area of the faces adjcent to each point
        pweight[point0] += farea_l
        pweight[point1] += farea_l
        pweight[point2] += farea_l
        
    # Compute weighted vertex
    cdef double wgt
    if return_weighted:
        if additional_weights.shape[0] == npoints:
            for i in prange(npoints, nogil=True):
                wgt = additional_weights[i]*pweight[i]
                wvertex[i, 0] = wgt*v[i, 0]
                wvertex[i, 1] = wgt*v[i, 1]
                wvertex[i, 2] = wgt*v[i, 2]

        else:
            for i in prange(npoints, nogil=True):
                wgt = pweight[i]
                wvertex[i, 0] = wgt*v[i, 0]
                wvertex[i, 1] = wgt*v[i, 1]
                wvertex[i, 2] = wgt*v[i, 2]

        return np.asarray(pweight), np.asarray(wvertex)

    else: 
        return np.asarray(pweight)


def weighted_points_float(float [:, ::1] v, int [:, ::1] f,
                          double [::1] additional_weights, return_weighted=True):
    """
    Returns point weight based on area weight and weighted points.
    Points are weighted by adjcent area faces.

    Parameters
    ----------
    v : np.ndarray, np.float
        Point array

    f : np.ndarray, int
        n x 4 face array.  First column is padding and is ignored.

    Returns
    -------
    pweight : np.ndarray, np.double
        Point weight array

    wvertex : np.ndarray, np.double
        Vertices mutlipled by their corresponding weights.
    """
    if f.shape[1] != 4:
        raise Exception('Must be an unclipped vtk face array')

    cdef int nfaces = f.shape[0]
    cdef int npoints = v.shape[0]
    cdef double [::1] pweight = np.zeros(npoints)
    cdef double [::1] farea = np.empty(nfaces)
    cdef double [:, ::1] wvertex = np.empty((npoints, 3))

    cdef double v0_0, v0_1, v0_2
    cdef double v1_0, v1_1, v1_2
    cdef double v2_0, v2_1, v2_2
    cdef double e0_0, e0_1, e0_2
    cdef double e1_0, e1_1, e1_2
    cdef double c0, c1, c2
    cdef double farea_l
    cdef int i

    cdef int point0, point1, point2

    for i in prange(nfaces, nogil=True):
        point0 = f[i, 1]
        point1 = f[i, 2]
        point2 = f[i, 3]

        v0_0 = v[point0, 0]
        v0_1 = v[point0, 1]
        v0_2 = v[point0, 2]

        v1_0 = v[point1, 0]
        v1_1 = v[point1, 1]
        v1_2 = v[point1, 2]

        v2_0 = v[point2, 0]
        v2_1 = v[point2, 1]
        v2_2 = v[point2, 2]

        # Edges
        e0_0 = v1_0 - v0_0
        e0_1 = v1_1 - v0_1
        e0_2 = v1_2 - v0_2
        
        e1_0 = v2_0 - v0_0
        e1_1 = v2_1 - v0_1
        e1_2 = v2_2 - v0_2

        c0 = e0_1*e1_2 - e0_2*e1_1
        c1 = e0_2*e1_0 - e0_0*e1_2
        c2 = e0_0*e1_1 - e0_1*e1_0

        # triangle area
        farea[i] = 0.5*sqrt(c0**2 + c1**2 + c2**2)

    for i in range(nfaces):
        point0 = f[i, 1]
        point1 = f[i, 2]
        point2 = f[i, 3]
        farea_l = farea[i]

        # Store the area of the faces adjcent to each point
        pweight[point0] += farea_l
        pweight[point1] += farea_l
        pweight[point2] += farea_l

    # Compute weighted vertex
    cdef double wgt
    if return_weighted:
        if additional_weights.shape[0] == npoints:
            for i in prange(npoints, nogil=True):
                wgt = additional_weights[i]*pweight[i]
                wvertex[i, 0] = wgt*v[i, 0]
                wvertex[i, 1] = wgt*v[i, 1]
                wvertex[i, 2] = wgt*v[i, 2]

        else:
            for i in prange(npoints, nogil=True):
                wgt = pweight[i]
                wvertex[i, 0] = wgt*v[i, 0]
                wvertex[i, 1] = wgt*v[i, 1]
                wvertex[i, 2] = wgt*v[i, 2]

        return np.asarray(pweight), np.asarray(wvertex)

    else: 
        return np.asarray(pweight)


def partial_cluster(int [:, ::1] neigharr, int [::1] c_ind):
    """ This will need some checking """
    cdef int npoints = neigharr.shape[0]
    cdef int nnode = neigharr.shape[1]
    cdef int n_nocheck = c_ind.shape[0]
    cdef int i, j, node, newnode, idx

    cdef vector[int] clus, edgeclus, testnodes, newtestnodes

    # Make a index bool array
    cdef int [::1] ncflag = np.zeros(npoints, ctypes.c_int)
    cdef int [::1] ncflag_unmod = ncflag.copy()
    cdef int [::1] edgept = ncflag.copy()

    for i in range(n_nocheck):
        idx = c_ind[i]
        ncflag[idx] = 1
        ncflag_unmod[idx] = 1

    # Cluster no check nodes and store all edges with movements
    cdef vector[vector[int]] nocheck_clusters
    cdef vector[vector[int]] edgeclus_list
    while np.any(ncflag):
        # Start new cluster with the first no check node
        for i in range(npoints):
            if ncflag[i]:
                break

        # Remove this point and push it back into the cluster
        ncflag[i] = 0
        testnodes.clear()
        testnodes.push_back(i)

        clus.clear()
        clus.push_back(i)

        edgeclus.clear()
        while testnodes.size():
            newtestnodes.clear()
            for i in testnodes:
                for j in xrange(nnode):
                    newnode = neigharr[i, j]
                    if newnode == -1:
                        break

                    elif ncflag[newnode]:
                        clus.push_back(newnode)
                        newtestnodes.push_back(newnode)
                        ncflag[newnode] = 0

                    elif not ncflag_unmod[newnode] and not edgept[newnode]:
                        edgeclus.push_back(newnode)
                        edgept[newnode] = 1
                        
            testnodes = newtestnodes

        # Append to cluster lists
        edgeclus_list.push_back(edgeclus)
        nocheck_clusters.push_back(clus)

    return edgeclus_list, nocheck_clusters


def neighbors_from_faces(int npoints, int [:, ::1] f):
    """
    Assemble neighbor array based on faces

    Parameters
    ----------
    points : int
        Number of points

    f : int [:, ::1]
        Face array.

    Returns
    -------
    neigh : int np.ndarray [:, ::1]
        Indices of each neighboring node for each node.

    nneigh : int np.ndarray [::1]
        Number of neighbors for each node.
    """

    # Find the maximum number of edges, with overflow buffer
    cdef int nconmax = max_con_face(npoints, f) + 10
    # NON-CODE BREAKING BUG: under-estimates number of connections

    cdef int nfaces = f.shape[0]
    cdef int i, j, k, pA, pB, pC
    cdef int [:, ::1] neigharr = np.empty((npoints, nconmax), ctypes.c_int)
    neigharr[:] = -1
    cdef int [::1] ncon = np.empty(npoints, ctypes.c_int)
    ncon[:] = 0

    for i in range(nfaces):

        # for each edge
        for j in range(1, 4):

            # always the current point
            pA = f[i, j]

            # wrap around last edge
            if j < 4 - 1:
                pB = f[i, j + 1]
            else:
                pB = f[i, 1]

            for k in range(nconmax):
                if neigharr[pA, k] == pB:
                    break
                elif neigharr[pA, k] == -1:
                    neigharr[pA, k] = pB
                    ncon[pA] += 1

                    # Mirror node will have the same number of connections
                    neigharr[pB, ncon[pB]] = pA
                    ncon[pB] += 1
                    break

    py_neigharr = np.asarray(neigharr)
    py_ncon = np.asarray(ncon)
    py_neigharr = np.ascontiguousarray(py_neigharr[:, :py_ncon.max()])
    return py_neigharr, py_ncon


cdef int max_con_face(int npoints, int [:, ::1] f):
    """Get maximum number of connections given edges """

    cdef int nfaces = f.shape[0]
    # cdef int nface_pts = f.shape[1]
    cdef int i
    cdef int mxval = 0
    cdef int [::1] ncon = np.zeros(npoints, ctypes.c_int)

    for i in range(nfaces):
        for j in range(1, 4):
            ncon[f[i, j]] += 1
        
    for i in range(npoints):
        if ncon[i] > mxval:
            mxval = ncon[i]

    return mxval


