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

from collections import Counter
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
import igraph as ig

def _ss_dist(X, w1, w2):
    """Computes the distance between two clusters."""
    mu1 = np.average(X, axis=0, weights=w1)
    mu2 = np.average(X, axis=0, weights=w2)
    diff = mu1-mu2
    s1 = np.cov(X.transpose(), aweights=w1)
    s2 = np.cov(X.transpose(), aweights=w2)
    return np.dot(diff, np.dot(np.linalg.solve(s1+s2,np.identity(2)), diff))

def slingshot(obj, name=(), min_cluster_size=10, verbose=False):
    """Trajectory analysis on the cluster level following the strategy in the R
    package slingshot
    
    Notes
    -----
    Slingshot's approach takes cells in a low dimensional space (UMAP is used
    below) and a clustering to generate a graph where vertices are clusters.
    
    Only slingthot's 'getLineages' method is used at the moment.
    
    References
    ----------
    .. [1] Street et al. (2018) BMC Genomics. Slingshot: cell lineage and pseudotime
           inference for single-cell transcriptomics
    .. [2] https://bioconductor.org/packages/release/bioc/html/slingshot.html

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    name : `tuple`
        A tuple of normalization to use. If it has the length zero, then all
        available normalizations will be used.
    min_cluster_size : `int`
        Minimum number of cells per cluster to include the cluster. Default: 10
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
        for clust_alg in item['clusters']:
            X = item['dr']['umap']['embedding'].copy()
            if verbose:
                print(clust_alg)
            cl = np.array(item['clusters'][clust_alg]['membership'])
            if min_cluster_size > 0:
                z = pd.Series(dict(Counter(cl)))
                remove = z[z < min_cluster_size].index.values
                keep = np.logical_not(pd.Series(cl).isin(remove)).values
                X = X.loc[keep, :]
                cl = cl[keep]
            clusters = np.unique(cl)
            nclus = len(clusters)
            # cluster weights matrix
            l = np.array([(cl == clID).astype(int) for clID in clusters]).transpose()
            # calculate the center for every cluster in the embedding
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
            mst = minimum_spanning_tree(D).toarray()
            mst = (mst>0).astype(int) # cast to binary
            forest = mst[:-1, :-1]
            # identify subtrees
            subtrees = forest.copy()
            subtrees_update = forest.copy()
            np.fill_diagonal(subtrees, 1)
            while subtrees_update.sum() > 0:
                subtrees_new = []
                for w in np.arange(0, subtrees.shape[1]):
                    subtrees_new.append(subtrees[:, subtrees[:, w]>0].sum(axis=1)>0)
                subtrees_new = np.array(subtrees_new).transpose()
                subtrees_update = subtrees_new - subtrees
                subtrees = subtrees_new.astype(int)
            subtrees = pd.DataFrame(np.unique(subtrees, axis=0))
            # identify lineages (paths through subtrees)
            forest = pd.DataFrame(forest)
            lineages = []
            for r in np.arange(0, subtrees.shape[1]):
                st = subtrees.iloc[r,]
                if st.sum() == 1:
                    continue
                tree_graph = forest.loc[st>0, st>0]
                degree = tree_graph.sum(axis=1)
                # create a graph object from the adjacency matrix
                g = ig.Graph.Adjacency(tree_graph.to_numpy().tolist(),
                                       mode='UNDIRECTED')
                leaves = np.arange(0,len(degree))[degree==1]
                avg_lineage_length = []
                for leave in leaves:
                    end = leaves[leaves!=leave]
                    paths = g.get_all_shortest_paths(v=leave, to=end, mode='out')
                    avg_lineage_length.append(np.mean([len(q) for q in paths]))
                if len(avg_lineage_length) == 0:
                    continue
                st = leaves[avg_lineage_length.index(max(avg_lineage_length))]
                ends = leaves[leaves != st]
                paths = g.get_all_shortest_paths(v=st, to=ends, mode='out')
                for p in paths:
                    lineages.append(tree_graph.columns[p].values)
            # sort by number of clusters included
            lineages.sort(key=len, reverse=True)
            if verbose:
                print('found %s lineages' % len(lineages))
            res = {'lineages' : lineages, 'adjacency' : forest}
            obj.norm_data[k]['slingshot'][clust_alg] = res
            obj.set_assay('slingshot')
