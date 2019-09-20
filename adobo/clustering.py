# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzén <p.oscar.franzen@gmail.com>
"""
Summary
-------
This module contains functions to cluster data.
"""
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import coo_matrix
import leidenalg as la
import igraph as ig
import sys

from ._log import warning

def knn(comp, k=10, distance='euclidean'):
    """
    Nearest Neighbour Search. Finds the k number of near neighbours for each cell.
    
    Parameters
    ----------
    comp : :py:class:`pandas.DataFrame`
        A pandas data frame containing PCA components.
    k : `int`
        Number of nearest neighbors. Default: 10
    target : `{'irlb', 'svd'}`
        The dimensionality reduction result to run the NN search on. Default: irlb
    distance : `str`
        Distance metric to use. See here for valid choices: https://tinyurl.com/y4bckf7w
        
    Returns
    -------
    numpy.ndarray
        Array containing indices.
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric=distance)
    nbrs.fit(comp)
    indices = nbrs.kneighbors(comp)[1]
    nn_idx = indices+1
    return nn_idx

def snn(nn_idx, k=10, prune_snn=0.067, verbose=False):
    """
    Computes a Shared Nearest Neighbor (SNN) graph
    
    Notes
    -----
    Link weights are number of shared nearest neighbors. The sum of SNN similarities over
    all KNNs is retrieved with linear algebra.
    
    Parameters
    ----------
    nn_idx : :py:class:`numpy.ndarray`
        Numpy array generated using knn()
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
    # create sparse matrix from tuples
    melted = pd.DataFrame(nn_idx).melt(id_vars=[0])[[0, 'value']]

    rows = np.array(melted[melted.columns[0]])
    cols = np.array(melted[melted.columns[1]])
    d = [1]*len(rows)

    ll = list(range(1, nn_idx.shape[0]+1))
    rows = np.array(list(melted[melted.columns[0]].values) + ll)
    cols = np.array(list(melted[melted.columns[1]]) + ll)

    d = [1]*len(rows)
    knn_sparse = coo_matrix((d, (rows-1, cols-1)),
                            shape=(nn_idx.shape[0], nn_idx.shape[0]))
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
    if verbose and perc_pruned > 80:
        warning('More than 80% of the edges were pruned')
    df = pd.DataFrame({'source_node' : node1, 'target_node' : node2})
    snn_graph = df
    return snn_graph

def leiden(snn_graph, res=0.8, seed=42):
    """
    Runs the Leiden algorithm
    
    Parameters
    ----------
    snn_graph : :py:class:`pandas.DataFrame`
        Source and target nodes.
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
    nn = set(snn_graph[snn_graph.columns[0]])
    g = ig.Graph()
    g.add_vertices(len(nn))
    g.vs['name'] = list(range(1, len(nn)+1))
    ll = []
    for i in snn_graph.itertuples(index=False):
        ll.append(tuple(i))
    g.add_edges(ll)
    #if self.params == 'ModularityVertexPartition':
    #    part = leidenalg.ModularityVertexPartition
    #else:
    part = la.RBERVertexPartition
    cl = la.find_partition(g, part, n_iterations=10, resolution_parameter=res, seed=seed)
    return cl.membership

def igraph(snn_graph, clust_alg):
    """
    Runs clustering functions within igraph
    
    Parameters
    ----------
    snn_graph : :py:class:`pandas.DataFrame`
        Source and target nodes.
    clust_alg : `{'walktrap', 'spinglass', 'multilevel', 'infomap', 'label_prop', 'leading_eigenvector'}`
        Specifies the community detection algorithm.
    
    References
    ----------
    Pons & Latapy (2006) Computing Communities in Large NetworksUsing Random Walks,
        Journal of Graph Algorithms and Applications
    Reichardt & Bornholdt (2006) Statistical mechanics of community detection,
        Physical Review E
    
    Returns
    -------
    Nothing. Modifies the passed object.
    """
    nn = set(snn_graph[snn_graph.columns[0]])
    g = ig.Graph()
    g.add_vertices(len(nn))
    g.vs['name'] = list(range(1, len(nn)+1))

    ll = []
    for i in snn_graph.itertuples(index=False):
        ll.append(tuple(i))
    g.add_edges(ll)

    if clust_alg == 'walktrap':
        z = ig.Graph.community_walktrap(g)
        cl = z.as_clustering(z.optimal_count).membership
    elif clust_alg == 'spinglass':
        z = ig.Graph.community_spinglass(g)
        cl = z.membership
    elif clust_alg == 'multilevel':
        z = ig.Graph.community_multilevel(g)
        cl = z.membership
    elif clust_alg == 'infomap':
        z = ig.Graph.community_infomap(g)
        cl = z.membership
    elif clust_alg == 'label_prop':
        z = ig.Graph.community_label_propagation(g)
        cl = z.membership
    elif clust_alg == 'leading_eigenvector':
        z = ig.Graph.community_leading_eigenvector(g)
        cl = z.membership
    else:
        raise Exception('Unsupported community detection algorithm specified.')
    return cl

def generate(obj, k=10, name=None, distance='euclidean', graph='snn', clust_alg='leiden',
             prune_snn=0.067, res=0.8, seed=42, verbose=False):
    """
    A wrapper function for generating single cell clusters from a shared nearest neighbor
    graph with the Leiden algorithm
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    k : `int`
        Number of nearest neighbors. Default: 10
    name : `str`
        The name of the normalization to operate on. If this is empty or None then the
        function will be applied on all normalizations available.
    distance : `str`
        Distance metric to use. See here for valid choices: https://tinyurl.com/y4bckf7w
    target : `{'irlb', 'svd'}`
        The dimensionality reduction result to run on. Default: irlb
    graph : `{'snn'}`
        Type of graph to generate. Only shared nearest neighbor (snn) supported at the
        moment.
    clust_alg : `{'leiden', 'walktrap', 'spinglass', 'multilevel', 'infomap',
                  'label_prop', 'leading_eigenvector'}`
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
    `dict`
        A dict containing cluster sizes (number of cells), only retx is set to True.
    """
    m = ('leiden', 'walktrap', 'spinglass', 'multilevel', 'infomap', 'label_prop',
         'leading_eigenvector')
    if not clust_alg in m:
        raise Exception('Supported community detection algorithms are: %s' % ', '.join(m))
    if not obj.norm_data:
        raise Exception('Run normalization first before running umap. See here: \
https://oscar-franzen.github.io/adobo/adobo.html#adobo.normalize.norm')
    targets = {}
    if name is None or name == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    print(targets.keys())
    for l in targets:
        item = targets[l]
        if verbose:
            print('Running clustering on the %s normalization' % l)
        comp = item['dr']['pca']['comp']
        nn_idx = knn(comp, k, distance)
        snn_graph = snn(nn_idx, k, prune_snn, verbose)
        if clust_alg == 'leiden':
            cl = leiden(snn_graph, res, seed)
        else:
            cl = igraph(snn_graph, clust_alg)
        obj.norm_data[l]['clusters'][clust_alg] = {'membership' : cl}
        obj.set_assay('clustering')
        if verbose:
            print(dict(Counter(cl)))
