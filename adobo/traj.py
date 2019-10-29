# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for trajectory analysis.
"""

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

def _ss_dist(X, w1, w2):
    """Computes the distance between two clusters."""
    mu1 = np.average(X, axis=0, weights=w1)
    mu2 = np.average(X, axis=0, weights=w2)
    diff = mu1-mu2
    s1 = np.cov(X.transpose(), aweights=w1)
    s2 = np.cov(X.transpose(), aweights=w2)
    return np.dot(diff, np.dot(np.linalg.solve(s1+s2,np.identity(2)), diff))

def slingshot(obj, name=(), verbose=False):
    """Trajectory analysis on the cluster level following the strategy in the R package
    slingshot
    
    Notes
    -----
    Slingshot's approach takes cells in a low dimensional space (UMAP is used below) and
    a clustering to generate a graph where vertices are clusters.
    
    Only slingthot's 'getLineages' method is used at the moment.
    
    References
    ----------
    Street et al. (2018) BMC Genomics. Slingshot: cell lineage and pseudotime inference
        for single-cell transcriptomics

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    name : `tuple`
        A tuple of normalization to use. If it has the length zero, then all available
        normalizations will be used.
    verbose : `bool`, optional
        Be verbose or not. Default: False

    Returns
    -------
    Nothing modifies the passed object.
    """
    targets = {}
    if len(name) == 0 or name == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    for i, k in enumerate(targets):
        if verbose:
            print('Running slingshot trajectory on %s' % k)
        item = targets[k]
        X = item['dr']['umap']['embedding']
        cl = np.array(item['clusters']['leiden']['membership'])
        clusters = np.unique(cl)
        nclus = len(clusters)
        # cluster weights matrix
        l = np.array([(cl == clID).astype(int) for clID in clusters]).transpose()
        # calculate the center for eveyr cluster in the embedding
        centers = []
        for clID in clusters:
            w = l[:, clID]
            centers.append(np.average(X, axis=0, weights=w))
        centers = np.array(centers)
        min_clus_size = min(l.sum(axis=0))
        # generate cluster distance matrix
        D = []
        for clID1 in clusters:
            r = []
            for clID2 in clusters:
                w1 = l[:, clID1]
                w2 = l[:, clID2]
                r.append(_ss_dist(X, w1, w2))
            D.append(r)
        D = np.array(D)
        # artifical cluster
        omega = D.max() + 1
        D = np.vstack((D, [omega]*D.shape[0]))
        q = [omega]*D.shape[1]
        q.append(0)
        D = np.column_stack([D, np.array(q)])
        # define a minimum spanning tree
        # computed using Kruskal's algorithm (ape::mst appears to be using Prim's
        # algorithm)
        # https://en.wikipedia.org/wiki/Prim%27s_algorithm
        # https://en.wikipedia.org/wiki/Kruskal%27s_algorithm
        mst = minimum_spanning_tree(D).todense()
        mst = (mst>0).astype(int) # cast to binary
        D = D[:-1, :-1]
        # define lineages
        subtrees = mst.copy()
        subtrees_update = mst.copy()
        np.fill_diagonal(subtrees, 1)
        
        while sum(subtrees_update) > 0:

        while(sum(subtrees.update) > 0){
            subtrees.new <- apply(subtrees,2,function(col){
                rowSums(subtrees[,as.logical(col), drop=FALSE]) > 0
            })
            subtrees.update <- subtrees.new - subtrees
            subtrees <- subtrees.new
        }
