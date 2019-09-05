# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
This module contains functions to cluster data.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import leidenalg as la
import igraph as ig
import sys

from ._log import warning

def _knn(obj, k=10, target='irlb'):
    """
    Nearest Neighbour Search. Finds the k number of near neighbours for each cell.
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    k : `int`
        Number of nearest neighbors. Default: 10
    target : `{'irlb', 'pca'}`
        The dimensionality reduction result to run the NN search on. Default: irlb
        
    Returns
    -------
    Nothing. Modifies the passed object.
    """
    X = obj.dr[target]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs.fit(X)
    indices = nbrs.kneighbors(X)[1]
    obj.nn_idx = indices+1

def _snn(obj, k=10, prune_snn=0.067, verbose=False):
    """
    Computes a Shared Nearest Neighbor (SNN) graph
    
    Notes
    -----
    Link weights are number of shared nearest neighbors. The sum of SNN similarities over
    all KNNs is retrieved with linear algebra.
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    k : `int`
        Number of nearest neighbors. Default: 10
    prune_snn : `float`        
        Threshold for pruning the SNN graph, i.e. the edges  with lower value (Jaccard
        index) than this will be removed. Set to 0 to disable pruning. Increasing this
        value will result in fewer edges in the graph. Default: 0.067
    verbose : `bool`
        Be verbose or not.
    
    References
    ----------
    http://mlwiki.org/index.php/SNN_Clustering
        
    Returns
    -------
    Nothing. Modifies the passed object.
    """
    k_param = k
    # create sparse matrix from tuples
    melted = pd.DataFrame(obj.nn_idx).melt(id_vars=[0])[[0, 'value']]

    rows = np.array(melted[melted.columns[0]])
    cols = np.array(melted[melted.columns[1]])
    d = [1]*len(rows)

    ll = list(range(1, obj.nn_idx.shape[0]+1))
    rows = np.array(list(melted[melted.columns[0]].values) + ll)
    cols = np.array(list(melted[melted.columns[1]]) + ll)

    d = [1]*len(rows)
    knn_sparse = coo_matrix((d, (rows-1, cols-1)),
                            shape=(obj.nn_idx.shape[0], obj.nn_idx.shape[0]))
    snn_sparse = knn_sparse*knn_sparse.transpose()
    cx = coo_matrix(snn_sparse)
    node1 = []
    node2 = []
    pruned_count = 0
    for i, j, v in zip(cx.row, cx.col, cx.data):
        item = (i, j, v)
        strength = v/(k+(k-v))
        if strength > prune_snn:
            node1.append(i)
            node2.append(j)
        else:
            pruned_count += 1
    perc_pruned = (pruned_count/len(cx.row))*100
    if verbose:
        print('%.2f%% (n=%s) of links pruned' % (perc_pruned,
                                                 '{:,}'.format(pruned_count)))
    if perc_pruned > 80:
        warning('more than 80% of the edges were pruned')
    df = pd.DataFrame({'source_node' : node1, 'target_node' : node2})
    obj.snn_graph = df

def _leiden(obj, res=0.8, seed=42):
    """
    Runs the Leiden algorithm
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    res : `float`
        Resolution parameter, change to modify cluster resolution. Default: 0.8
    seed : `int`
        For reproducibility.
    
    References
    ----------
    [0] https://github.com/vtraag/leidenalg
    [1] Traag et al. (2018) https://arxiv.org/abs/1810.08473
    
    Returns
    -------
    Nothing. Modifies the passed object.
    """
    # construct the graph object
    nn = set(obj.snn_graph[obj.snn_graph.columns[0]])
    g = ig.Graph()
    g.add_vertices(len(nn))
    g.vs['name'] = list(range(1, len(nn)+1))

    ll = []
    for i in obj.snn_graph.itertuples(index=False):
        ll.append(tuple(i))
    g.add_edges(ll)
    #if self.params == 'ModularityVertexPartition':
    #    part = leidenalg.ModularityVertexPartition
    #else:
    part = la.RBERVertexPartition
    cl = la.find_partition(g, part, n_iterations=10, resolution_parameter=res, seed=seed)
    obj.clusters.append({ 'algo' : 'leiden', 'cl' : cl.membership })

def _igraph(obj, clust_alg):
    """
    Runs clustering functions within igraph
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    clust_alg : `{'walktrap', 'spinglass'}`
    
    References
    ----------
    Pons & Latapy (2006) Computing Communities in Large NetworksUsing Random Walks,
        Journal of Graph Algorithms and Applications
    
    Returns
    -------
    Nothing. Modifies the passed object.
    """
    nn = set(obj.snn_graph[obj.snn_graph.columns[0]])
    g = ig.Graph()
    g.add_vertices(len(nn))
    g.vs['name'] = list(range(1, len(nn)+1))

    ll = []
    for i in obj.snn_graph.itertuples(index=False):
        ll.append(tuple(i))
    g.add_edges(ll)

    if clust_alg == 'walktrap':
        z = ig.Graph.community_walktrap(g)
        cl = z.as_clustering(z.optimal_count).membership
    elif clust_alg == 'spinglass':
        z = ig.Graph.community_spinglass(g)
        cl = z.membership
    else:
        raise Exception('Unsupported community detection algorithm specified.')
    
    obj.clusters.append({ 'algo' : clust_alg, 'cl' : cl })

def generate(obj, k=10, graph='snn', clust_alg='leiden', prune_snn=0.067,
             res=0.8, seed=42, verbose=False):
    """
    A wrapper function for generating single cell clusters from a shared nearest neighbor
    graph with the Leiden algorithm
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    k : `int`
        Number of nearest neighbors. Default: 10
    graph : `{'snn'}`
        Type of graph to generate. Only shared nearest neighbor (snn) supported at the
        moment.
    clust_alg : `{'leiden', 'walktrap', 'spinglass'}`
        Clustering algorithm to be used.
    prune_snn : `float`
        Threshold for pruning the SNN graph, i.e. the edges  with lower value (Jaccard
        index) than this will be removed. Set to 0 to disable pruning. Increasing this
        value will result in fewer edges in the graph. Default: 0.067
    res : `float`
        Resolution parameter for the Leiden algorithm _only_; change to modify cluster
        resolution. Default: 0.8
    seed : `int`
        For reproducibility.
    verbose : `bool`
        Be verbose or not.
    
    References
    ----------
    Yang et al. (2016) A Comparative Analysis of Community Detection Algorithms on
        Artificial Networks. Scientific Reports
    
    Returns
    -------
    Nothing. Modifies the passed object.
    """
    m = ('leiden', 'walktrap', 'spinglass')
    if not clust_alg in m:
        raise Exception('Supported community detection algorithms are: %s' % ', '.join(m))
    _knn(obj, k)
    _snn(obj, k, verbose)
    if clust_alg == 'leiden':
        _leiden(obj, res, seed)
    else:
        _igraph(obj, clust_alg)
    obj.set_assay(sys._getframe().f_code.co_name)
