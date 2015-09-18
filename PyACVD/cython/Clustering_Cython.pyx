# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython nonecheck=False
"""
Cython module to support ACVD.py

"""


# Python import
from __future__ import division
from __future__ import print_function
import numpy as np
cimport numpy as np
import sys

# C++ import
from cython.view cimport array as cvarray
from libcpp.vector cimport vector
from libcpp cimport bool as bool_t
from libc.math cimport sqrt


cdef int [::1] GrowNull(int [:, :] edgeID, int [::1] clusters, int nclus) nogil:
    """ Grow clusters to include null faces """

    cdef int faceA, faceB, clusA, clusB, nrem, nchange     
    
    # the number of null equals the number of starting clusters minus the total number of faces
    nrem = clusters.shape[0] - nclus
    
    while nrem > 0:
        nchange = 0
        # Determine edges that share two clusters
        for i in range(edgeID.shape[0]):
            # Get the two clusters sharing an edge
            faceA = edgeID[i, 0]
            faceB = edgeID[i, 1]
            clusA = clusters[faceA]
            clusB = clusters[faceB]        
    
            # Check and immedtialy flip a cluster edge if one is part of the null cluster
            if clusA == -1 and clusB != -1:
                clusters[faceA] = clusB
                nrem -= 1
                nchange += 1
            elif clusB == -1 and clusA != -1:
                clusters[faceB] = clusA
                nrem -= 1
                nchange += 1

        if nchange == 0:
            break
        
    return clusters


def GrowNegative(int [::1] cmod, int [:, :] edgeID):
    """ Grows negative clusters """
    
    cdef int itemA, itemB, clusA, clusB, nmod
    
    nmod = 1
    while nmod:
        nmod = 0
        for i in range(edgeID.shape[0]):
            # Test if an edge is equal to its negative
            itemA = edgeID[i, 0]
            itemB = edgeID[i, 1]
            clusA = cmod[itemA]
            clusB = cmod[itemB]
            
            # if a cluster is equal to its negative (along the growing negative front)
            if clusA + clusB == 0:
                nmod += 1
                # Determine which is negative
                if clusA < clusB:
                    cmod[itemB] = clusA
                else:
                    cmod[itemA] = clusB
                    
    return np.asarray(cmod)
            


def ClusterOpt(int [::1] clusters, nclusin, double [::1] area, double [:, ::1] cent,
               int [:, ::1] edgeID, int maxiter, bool_t verbose):
    """ Python interface function for cluster optimization """

    # Number of clusters
    cdef int nclus = nclusin
    
    # Number of faces
    cdef int nface = cent.shape[0]
    
    # Begin by eliminating null clusters by growing existing null clusters
    clusters = GrowNull(edgeID, clusters, nclus)
    
    # Arrays for cluster centers, masses, and energies
    cdef double [:, ::1] sgamma = np.zeros((nclus, 3), dtype=np.float, order='C')
    cdef double [::1] srho = np.zeros(nclus, dtype=np.float, order='C')
    cdef double [::1] energy = np.zeros(nclus, dtype=np.float, order='C')
        
    # Compute initial masses of clusters
    for i in range(nface):
        srho[clusters[i]] += area[i]
        sgamma[clusters[i], 0] += cent[i, 0]
        sgamma[clusters[i], 1] += cent[i, 1]
        sgamma[clusters[i], 2] += cent[i, 2]
    
    for i in range(nclus):
        energy[i] = (sgamma[i, 0]**2 + \
                     sgamma[i, 1]**2 + \
                     sgamma[i, 2]**2)/srho[i]
                                    
    # Count number of clusters
    cdef int [::1] cluscount = np.bincount(clusters).astype(np.int32)
    
    # Initialize modified array
    cdef int [::1] mod1 = np.ones(nclus, dtype=np.int32, order='C')
    cdef int [::1] mod2 = np.ones(nclus, dtype=np.int32, order='C')
    
    clusters = MinimizeEnergy(edgeID, clusters, area, sgamma, cent, srho, cluscount, maxiter,
                               energy, mod1, mod2, verbose)    
    return clusters
        
        
cdef int [::1] MinimizeEnergy(int [:, ::1] edgeID, int [::1] clusters, double [::1] area,
                              double [:, ::1] sgamma, double [:, ::1] cent, double [::1] srho,
                              int [::1] cluscount, int maxiter, double [::1] energy,
                              int [::1] mod1, int [::1] mod2, bool_t verbose):
    """ Minimize cluster energy"""

    cdef int faceA, faceB, clusA, clusB
    cdef double areafaceA, centA0, centA1, centA2
    cdef double areafaceB, centB0, centB1, centB2
    cdef double eA, eB, eorig, eAwB, eBnB, eAnA, eBwA
    cdef int nchange = 1   
    cdef int niter = 0
    cdef int nclus = mod1.shape[0]
    cdef int nedge = edgeID.shape[0]
    
    while nchange > 0 and niter < maxiter:
        
        # Reset modification arrays
        for i in range(nclus):
            mod1[i] = mod2[i]
            mod2[i] = 0
        
        nchange = 0
        for i in range(nedge):
            # Get the two clusters sharing an edge
            faceA = edgeID[i, 0]
            faceB = edgeID[i, 1]
            clusA = clusters[faceA]
            clusB = clusters[faceB]        
        
            # If edge shares two different clusters and at least one has been modified since last iteration
            if clusA != clusB and (mod1[clusA] == 1 or mod1[clusB] == 1):
                # Verify that face can be removed from cluster
                if cluscount[clusA] > 1 and cluscount[clusB] > 1:
    
                    areafaceA = area[faceA]
                    centA0 = cent[faceA, 0]
                    centA1 = cent[faceA, 1]
                    centA2 = cent[faceA, 2]  
                    
                    areafaceB = area[faceB]
                    centB0 = cent[faceB, 0]
                    centB1 = cent[faceB, 1]
                    centB2 = cent[faceB, 2] 
                    
                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]
                    
                    # Energy with both items assigned to cluster A
                    eAwB = ((sgamma[clusA, 0] + centB0)**2 + \
                            (sgamma[clusA, 1] + centB1)**2 + \
                            (sgamma[clusA, 2] + centB2)**2)/(srho[clusA] + areafaceB)
                           
                    eBnB = ((sgamma[clusB, 0] - centB0)**2 + \
                            (sgamma[clusB, 1] - centB1)**2 + \
                            (sgamma[clusB, 2] - centB2)**2)/(srho[clusB] - areafaceB)
                            
                    eA = eAwB + eBnB
    
                    # Energy with both items assigned to clusterB
                    eAnA = ((sgamma[clusA, 0] - centA0)**2 + \
                            (sgamma[clusA, 1] - centA1)**2 + \
                            (sgamma[clusA, 2] - centA2)**2)/(srho[clusA] - areafaceA)
                           
                    eBwA = ((sgamma[clusB, 0] + centA0)**2 + \
                            (sgamma[clusB, 1] + centA1)**2 + \
                            (sgamma[clusB, 2] + centA2)**2)/(srho[clusB] + areafaceA)
                             
                    eB = eAnA + eBwA
                    
                    # select the largest case (most negative)
                    if eA > eorig and eA > eB:
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        
                        nchange+=1
                        # reassign
                        clusters[faceB] = clusA
                        cluscount[clusB] -= 1
                        cluscount[clusA] += 1
                        
                        # Update cluster A mass and centroid
                        srho[clusA] += areafaceB
                        sgamma[clusA, 0] += centB0
                        sgamma[clusA, 1] += centB1
                        sgamma[clusA, 2] += centB2
                        
                        srho[clusB] -= areafaceB
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
                        clusters[faceA] = clusB
                        cluscount[clusA] -= 1
                        cluscount[clusB] += 1
                        
                        # Add item A to cluster A
                        srho[clusB] += areafaceA
                        sgamma[clusB, 0] += centA0
                        sgamma[clusB, 1] += centA1
                        sgamma[clusB, 2] += centA2
                        
                        # Remove item A from cluster A
                        srho[clusA] -= areafaceA
                        sgamma[clusA, 0] -= centA0
                        sgamma[clusA, 1] -= centA1
                        sgamma[clusA, 2] -= centA2
                        
                        # Update cluster energy
                        energy[clusA] = eAnA
                        energy[clusB] = eBwA
                                                
                elif cluscount[clusA] > 1:

                    areafaceA = area[faceA]
                    centA0 = cent[faceA, 0]
                    centA1 = cent[faceA, 1]
                    centA2 = cent[faceA, 2]
                    
                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]
                    
                    # Energy with both items assigned to clusterB
                    eAnA = ((sgamma[clusA, 0] - centA0)**2 + \
                            (sgamma[clusA, 1] - centA1)**2 + \
                            (sgamma[clusA, 2] - centA2)**2)/(srho[clusA] - areafaceA)
                           
                    eBwA = ((sgamma[clusB, 0] + centA0)**2 + \
                            (sgamma[clusB, 1] + centA1)**2 + \
                            (sgamma[clusB, 2] + centA2)**2)/(srho[clusB] + areafaceA)
                             
                    eB = eAnA + eBwA
                    
                    # Compare energy contributions
                    if eB > eorig:
                        
                        # Flag clusters as modified
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        nchange += 1
                        
                        # reassign
                        clusters[faceA] = clusB
                        cluscount[clusA] -= 1
                        cluscount[clusB] += 1
                        
                        # Add item A to cluster A
                        srho[clusB] += areafaceA
                        sgamma[clusB, 0] += centA0
                        sgamma[clusB, 1] += centA1
                        sgamma[clusB, 2] += centA2
                        
                        # Remove item A from cluster A
                        srho[clusA] -= areafaceA
                        sgamma[clusA, 0] -= centA0
                        sgamma[clusA, 1] -= centA1
                        sgamma[clusA, 2] -= centA2
                        
                        # Update cluster energy
                        energy[clusA] = eAnA
                        energy[clusB] = eBwA
                        
                        
                elif cluscount[clusB] > 1:                
                    
                    areafaceB = area[faceB]
                    centB0 = cent[faceB, 0]
                    centB1 = cent[faceB, 1]
                    centB2 = cent[faceB, 2]  
                    
                    # Current energy
                    eorig =  energy[clusA] + energy[clusB]
                    
                    # Energy with both items assigned to cluster A
                    eAwB = ((sgamma[clusA, 0] + centB0)**2 + \
                            (sgamma[clusA, 1] + centB1)**2 + \
                            (sgamma[clusA, 2] + centB2)**2)/(srho[clusA] + areafaceB)
                           
                    eBnB = ((sgamma[clusB, 0] - centB0)**2 + \
                            (sgamma[clusB, 1] - centB1)**2 + \
                            (sgamma[clusB, 2] - centB2)**2)/(srho[clusB] - areafaceB)
                            
                    eA = eAwB + eBnB
    
                    # If moving face B reduces cluster energy
                    if eA > eorig:
                        
                        mod2[clusA] = 1
                        mod2[clusB] = 1
                        
                        nchange+=1
                        # reassign
                        clusters[faceB] = clusA
                        cluscount[clusB] -= 1
                        cluscount[clusA] += 1
                        
                        # Update cluster A mass and centroid
                        srho[clusA] += areafaceB
                        sgamma[clusA, 0] += centB0
                        sgamma[clusA, 1] += centB1
                        sgamma[clusA, 2] += centB2
                        
                        srho[clusB] -= areafaceB
                        sgamma[clusB, 0] -= centB0
                        sgamma[clusB, 1] -= centB1
                        sgamma[clusB, 2] -= centB2
                        
                        # Update cluster energy
                        energy[clusA] = eAwB
                        energy[clusB] = eBnB
                        
        niter += 1
        if verbose:
            sys.stdout.write('\rLoop {:4d} with {:6d} modifications'.format(niter, nchange))
            
    if verbose:
        sys.stdout.write('\n')
            
    return clusters      


def NullDisconnected(int [::1] clusters, int [::1] cmod, int [::1] disconclus, int[:, ::] edgeID):
    """ Assign disconnected clusters to the null cluster (-1) """

    cdef vector[int] null_list
    cdef vector[vector[int]] itemlist
    cdef vector[vector[int]] itemlist_temp
    cdef vector[int] tempvec
    cdef int ndiscon = disconclus.shape[0]
    cdef int nmod2
    cdef int nedge = edgeID.shape[0]
    cdef int itemA, itemB, clusA, clusB
    cdef int items = cmod.shape[0]
    cdef int clus, additem

    # Initialize lists
    for i in range(ndiscon):
        itemlist.push_back(tempvec)
        itemlist_temp.push_back(tempvec)

    # reset cmod array to positive
    for i in range(cmod.shape[0]):
        if cmod[i] < 0:
            cmod[i] = -cmod[i]    
    
    cdef int nmod = 1
    while nmod: # while any positive clusters exist
        nmod = 0

        # Clear temporary lists
        for i in range(ndiscon):
            itemlist_temp[i].clear()
            
        # Find first occurances of each disconnected cluster
        for i in range(ndiscon):
            clus = disconclus[i]
            for j in range(items):
                if cmod[j] == clus:
                    nmod += 1
                    cmod[j] = -clus
                    itemlist_temp[i].push_back(j)
                    break
                    
        # Grow negative items into positive ones
        nmod2 = 1
        while nmod2 and nmod:
            nmod2 = 0
            for i in range(nedge):
                # Test if an edge is equal to its negative
                itemA = edgeID[i, 0]
                itemB = edgeID[i, 1]
                clusA = cmod[itemA]
                clusB = cmod[itemB]
                
                # if a cluster is equal to its negative (along the growing negative front)
                if clusA + clusB == 0:
                    nmod2 += 1
                    # Determine which is negative
                    if clusA < clusB:
                        clusID = clusB
                        cmod[itemB] = clusA
                        additem = itemB
                    else:
                        clusID = clusA
                        cmod[itemA] = clusB
                        additem = itemA
        
                        # determine which list to add this disconnected cluster to
                        for j in range(ndiscon):
                            if clusID == disconclus[j]:
                                itemlist_temp[j].push_back(additem)
                                break
                            
        # Determine which items to reset to the null cluster
        for i in range(ndiscon):
            # if new cluster is larger than the saved
            if itemlist_temp[i].size() > itemlist[i].size():
                for item in itemlist[i]:
                    null_list.push_back(item)
                itemlist[i] = itemlist_temp[i]
            # otherwise, if the new cluster is smaller than the cached one
            else:
                for item in itemlist_temp[i]:
                    null_list.push_back(item)

    # Set cluster items to null
    for i in null_list:
        clusters[i] = -1
        
    return clusters


def GetNeighborsInterface(int [:, ::1] edgeID, int nitems):
    """ Python interface function to return an array of item neighbor connectivity """

    cdef vector[vector[int]] fneigh

    cdef int [::1] nneigh = np.zeros(nitems, dtype=np.int32, order='C')
    nneigh = CountNeighbors(edgeID, nneigh)

    cdef int nmax=0
    for i in range(nitems):
        if nneigh[i] > nmax:
            nmax = nneigh[i]
    
    cdef int [:, ::1] neigharr = np.ones((nitems, nmax), dtype=np.int32, order='C')*-1
    
#    nneigh[:] = 0
    neigharr = GetNeighbors(edgeID, neigharr, nneigh)
    
    
    return np.asarray(neigharr), np.asarray(nneigh)
    

cdef int [:, ::1] GetNeighbors(int [:, ::1] edgeID, int [:, ::1] neigharr,
                               int [::1] nneigh) nogil:
   
    # Reset number of neighbors
    for i in range(nneigh.shape[0]):
        nneigh[i] = 0
   
    # Populate neighbor array with neighbors
    for i in range(edgeID.shape[0]):
        neigharr[edgeID[i, 0], nneigh[edgeID[i, 0]]] = edgeID[i, 1]
        neigharr[edgeID[i, 1], nneigh[edgeID[i, 1]]] = edgeID[i, 0]
        nneigh[edgeID[i, 0]] += 1
        nneigh[edgeID[i, 1]] += 1
        

    return neigharr    
    
    
cdef int [::1] CountNeighbors(int [:, ::1] edgeID, int [::1] nneigh) nogil:
   
    for i in range(edgeID.shape[0]):
        nneigh[edgeID[i, 0]] += 1
        nneigh[edgeID[i, 1]] += 1
        
    return nneigh
    

def InitClustersInterface(int [:, ::1] neighbors, int [::1] nneigh, double [::1] area, n):
    """ Python interface function to initialize clusters """

    # number of items
    cdef int nitems = area.shape[0]

    # Total mesh size
    cdef double area_remain = 0
    for i in range(nitems):
        area_remain += area[i]
        
    # Create clusters array
    cdef int [::1] clusters = np.ones(nitems, dtype=np.int32, order='C')*-1
    cdef int nclus = n
    
    clusters = InitClusters(neighbors, nneigh, area, clusters, nitems, nclus, area_remain)    
        
    return clusters
    

cdef int[::1] InitClusters(int [:, ::1] neighbors, int [::1] nneigh, double [::1] area, 
                           int [::1] clusters, int nitems, int nclus, float area_remain):
    """ Cython module to initialize clusters """
                               
    # Initialize vectors
    cdef vector[int] check_items
    cdef vector[int] new_items
    cdef double tarea, new_area, carea
    cdef int item
    cdef int i, k
    cdef double ctarea
    cdef double under_allowance = 0
    
    # Assign clsuters
    ctarea = area_remain/nclus
    for i in range(nclus):
        # Get target area and reset current area
        tarea = area_remain - ctarea*(nclus - i - 1)
        carea = 0
        
        # Get starting index (the first free face in list)
        for j in range(nitems):
            if clusters[j] == -1:
                carea += area[j]
                new_items.push_back(j)
                clusters[j] = i
                break            
        
        # While there are new items to be added
        while new_items.size():
            
            # check all new items and reset list
            check_items = new_items
            new_items.clear()
            
            # progressively add neighbors
            for checkitem in check_items:
                for k in range(nneigh[checkitem]):
                    item = neighbors[checkitem, k]
                    
                    # check if the face is free
                    if clusters[item] == -1:
                        # if allowable, add to cluster
                        if area[item] + carea < tarea:
                            carea += area[item]
                            clusters[item] = i
                            new_items.push_back(item)

        under_allowance += carea - tarea
        area_remain -= carea
        
    return clusters
