# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzén <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for dimensional reduction.
"""
import sys
from random import sample
import pandas as pd
import numpy as np
import scipy.linalg
from sklearn.preprocessing import scale as sklearn_scale
import sklearn.manifold
import umap as um
import igraph as ig
from fa2 import ForceAtlas2

from . import irlbpy
from ._log import warning

def force_graph(obj, name=(), iterations=1000, edgeWeightInfluence=1.0,
                jitterTolerance=1.0, barnesHutOptimize=True, scalingRatio=2.0,
                gravity=1.0, strongGravityMode=False, verbose=False):
    """Generates a force-directed graph

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    name : `str`
        The name of the normalization to operate on. If this is empty or None
        then the function will be applied on the last normalization that was applied.
    iterations : `int`
        Number of iterations. Default: 1000
    edgeWeightInfluence : `float`
        How much influence to edge weights. 0 is no influence and 1 is normal.
        Default: 1.0
    jitterTolerance : `float`
        Amount swing. Lower gives less speed and more precision. Default: 1.0
    barnesHutOptimize : `bool`
        Run Barnes Hut optimization. Default: True
    scalingRatio : `float`
        Amount of repulsion, higher values make a more sparse graph. Default: 2.0
    gravity : `float`
        Attracts nodes to the center. Prevents islands from drifting away. Default: 1.0
    strongGravityMode : `bool`
        A stronger gravity view. Default: False
    verbose : `bool`
        Be verbose or not.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Force-directed_graph_drawing

    Returns
    -------
    None
    """
    targets = {}
    if name is None or len(name) == 0:
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    forceatlas2 = ForceAtlas2(outboundAttractionDistribution=True,
                              edgeWeightInfluence=edgeWeightInfluence,
                              jitterTolerance=jitterTolerance,
                              barnesHutOptimize=barnesHutOptimize, barnesHutTheta=1.2,
                              scalingRatio=scalingRatio,
                              strongGravityMode=strongGravityMode, gravity=gravity,
                              verbose=verbose)
    for l in targets:
        item = targets[l]
        if verbose:
            print('Generating force-directed graph for the %s normalization' % l)
        if not 'graph' in item:
            raise Exception('Graph has not been generated. Run \
`adobo.clustering.generate(...)` first.')
        snn_graph = item['graph']
        nn = set(snn_graph[snn_graph.columns[0]])
        g = ig.Graph()
        g.add_vertices(len(nn))
        g.vs['name'] = list(range(1, len(nn)+1))
        ll = []
        for i in snn_graph.itertuples(index=False):
            ll.append(tuple(i))
        g.add_edges(ll)
        layout = forceatlas2.forceatlas2_igraph_layout(g, pos=None, iterations=iterations)
        npa = np.array(layout)
        obj.norm_data[l]['dr']['force_graph'] = {'coords' : pd.DataFrame(npa)}
    obj.set_assay(sys._getframe().f_code.co_name)

def irlb(data_norm, scale=True, ncomp=75, var_weigh=True, seed=None):
    """Truncated SVD by implicitly restarted Lanczos bidiagonalization

    Notes
    -----
    The augmented implicitly restarted Lanczos bidiagonalization algorithm (IRLBA) finds
    a few approximate largest singular values and corresponding singular vectors using a
    method of Baglama and Reichel.
    
    Cells should be rows and genes as columns.

    Parameters
    ----------
    data_norm : :py:class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data.
    scale : `bool`
        Scales input data prior to PCA. Default: True
    ncomp : `int`
        Number of components to return. Default: 75
    var_weigh : `bool`
        Weigh by the variance of each component. Default: True
    seed : `int`
        For reproducibility. Default: None

    References
    ----------
    .. [1] Baglama et al (2005) Augmented Implicitly Restarted Lanczos Bidiagonalization
           Methods SIAM Journal on Scientific Computing
    .. [2] https://github.com/bwlewis/irlbpy

    Returns
    -------
    `pd.DataFrame`
        A py:class:`pandas.DataFrame` containing the components (columns).
    `pd.DataFrame`
        A py:class:`pandas.DataFrame` containing the contributions of every gene (rows).
    """
    inp = data_norm
    idx = inp.index
    cols = inp.columns
    inp = inp.transpose()
    if scale:
        inp = sklearn_scale(inp,  # cells as rows and genes as columns
                            axis=0,           # over genes, i.e. features (columns)
                            with_mean=True,   # subtracting the column means
                            with_std=True)    # scale the data to unit variance
        inp = pd.DataFrame(inp, columns=idx, index=cols)
    # cells should be rows and genes as columns
    lanc = irlbpy.lanczos(inp, nval=ncomp, maxit=1000, seed=seed)
    if var_weigh:
        # weighing by variance
        comp = np.dot(lanc.U, np.diag(lanc.s))
    else:
        comp = lanc.U
    comp = pd.DataFrame(comp)
    # gene loadings
    contr = pd.DataFrame(lanc.V, index=inp.columns)
    return comp, contr

def svd(data_norm, scale=True, ncomp=75, only_sdev=False):
    """Principal component analysis via singular value decomposition

    Parameters
    ----------
    data_norm : :class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data. Preferrably this
        should be a subset of the normalized gene expression matrix containing highly
        variable genes.
    scale : `bool`
        Scales input data prior to PCA. Default: True
    ncomp : `int`
        Number of components to return. Default: 75
    only_sdev : `bool`
        Only return the standard deviation of the components. Default: False

    References
    ----------
    .. [1] https://tinyurl.com/yyt6df5x

    Returns
    -------
    `pd.DataFrame`
        A py:class:`pandas.DataFrame` containing the components (columns). Only if
        only_sdev=False.
    `pd.DataFrame`
        A py:class:`pandas.DataFrame` containing the contributions of every gene (rows).
        Only if only_sdev=False.
    `pd.DataFrame`
        A py:class:`pandas.DataFrame` containing standard deviations of components. Only
        if only_sdev is set to True.
    """
    inp = data_norm
    idx = inp.index
    cols = inp.columns
    inp = inp.transpose()
    if scale:
        inp = sklearn_scale(inp,  # cells as rows and genes as columns
                            axis=0,           # over genes, i.e. features (columns)
                            with_mean=True,   # subtracting the column means
                            with_std=True)    # scale the data to unit variance
        inp = pd.DataFrame(inp, columns=idx, index=cols)
    nfeatures = inp.shape[0]
    compute_uv = not only_sdev
    if only_sdev:
        s = scipy.linalg.svd(inp, compute_uv=compute_uv)
        sdev = s/np.sqrt(nfeatures-1)
        return sdev
    # cells should be rows and genes as columns
    U, s, Vh = scipy.linalg.svd(inp, compute_uv=compute_uv)
    Vh = Vh.transpose()
    retx = inp.dot(Vh)
    retx = retx.iloc[:, 0:ncomp]
    comp = retx
    # gene loadings
    contr = pd.DataFrame(Vh[:, 0:ncomp], index=inp.columns)
    return comp, contr

def pca(obj, method='irlb', name=None, ncomp=75, allgenes=False, scale=True,
        var_weigh=True, verbose=False, seed=42):
    """Principal Component Analysis

    Notes
    -----
    A wrapper function around the individual normalization functions, which can also be
    called directly. Scaling of the data is achieved by setting scale=True (default),
    which will center (subtract the column mean) and scale columns (divide by their
    standard deviation).

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : `{'irlb', 'svd'}`
        Method to use for PCA. This does not matter much. Default: irlb
    name : `str`
        The name of the normalization to operate on. If this is empty or None then the
        function will be applied on all normalizations available.
    ncomp : `int`
        Number of components to return. Default: 75
    allgenes : `bool`
        Use all genes instead of only HVG. Default: False
    scale : `bool`
        Scales input data prior to PCA. Default: True
    var_weigh : `bool`
        Weigh by the variance of each component. Default: True
    verbose : `bool`
        Be noisy or not. Default: False
    seed : `int`
        For reproducibility (only irlb). Default: 42

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Principal_component_analysis

    Returns
    -------
    Nothing. Modifies the passed object. Results are stored in two dictonaries in the
    passed object: `dr` (containing the components) and `dr_gene_contr` (containing
    gene contributions to each component)
    """
    if not obj.norm_data:
        raise Exception('Run normalization first before running pca. See here: \
https://oscar-franzen.github.io/adobo/adobo.html#adobo.normalize.norm')
    targets = {}
    if name is None or name == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    # remove previous cluster analysis, b/c this changes after running hvg
    obj.remove_analysis(('clusters', 'dr'))
    for k in targets:
        item = targets[k]
        data = item['data']
        if not allgenes:
            if not 'hvg' in item:
                raise Exception('Run adobo.dr.find_hvg() first.')
            hvg = item['hvg']['genes']
            data = data[data.index.isin(hvg)]
        elif verbose:
            print('Using all genes')
        if verbose:
            v = (method, k, data.shape[0], data.shape[1])
            print('Running PCA (method=%s) on the %s normalization (dimensions \
%s genes x %s cells)' % v)
        if method == 'irlb':
            comp, contr = irlb(data, scale, ncomp, var_weigh, seed)
        elif method == 'svd':
            comp, contr = svd(data, scale, ncomp)
        else:
            raise Exception('Unkown PCA method spefified. Valid choices are: irlb and svd')
        comp.index = data.index
        obj.norm_data[k]['dr']['pca'] = {'comp' : comp,
                                         'contr' : contr,
                                         'method' : method}
        if verbose:
            print('saving %s components' % ncomp)
        obj.set_assay(sys._getframe().f_code.co_name, method)

def tsne(obj, run_on_PCA=True, name=None, perplexity=30, n_iter=2000, seed=None,
         verbose=False, **args):
    """Projects data to a two dimensional space using the tSNE algorithm.

    Notes
    -----
    It is recommended to perform this function on data in PCA space. This function calls
    :py:func:`sklearn.manifold.TSNE`, and any additional parameters will be passed to it.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    run_on_PCA : `bool`
        To run tSNE on PCA components or not. If False then runs on the entire normalized
        gene expression matrix. Default: True
    name : `str`
        The name of the normalization to operate on. If this is empty or None then the
        function will be applied on all normalizations available.
    perplexity : `float`
        From [1]: The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets usually require
        a larger perplexity. Consider selecting a value between 5 and 50. Different
        values can result in significanlty different results. Default: 30
    n_iter : `int`
        Number of iterations. Default: 2000
    seed : `int`
        For reproducibility. Default: None
    verbose : `bool`
        Be verbose. Default: False

    References
    ----------
    .. [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
           Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    .. [2] https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if not obj.norm_data:
        raise Exception('Run normalization first before running tsne. See here: \
https://oscar-franzen.github.io/adobo/adobo.html#adobo.normalize.norm')
    targets = {}
    if name is None or name == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    if verbose and not run_on_PCA:
        warning('Running tSNE on the entire gene expression matrix is not recommended.')
    for k in targets:
        item = targets[k]
        if not run_on_PCA:
            X = item['data']
        else:
            if len(item['dr']) == 0:
                raise Exception('Run dimensionality reduction first, for exampe \
adobo.dr.pca()')
            X = item['dr']['pca']['comp']
        if verbose:
            print('Running tSNE (perplexity %s) on the %s normalization' % (perplexity, k))
        tsne = sklearn.manifold.TSNE(n_components=2,
                                     n_iter=n_iter,
                                     perplexity=perplexity,
                                     random_state=seed,
                                     verbose=verbose,
                                     **args)
        emb = tsne.fit_transform(X)
        emb = pd.DataFrame(emb)
        obj.norm_data[k]['dr']['tsne'] = {'embedding' : emb,
                                          'perplexity' : perplexity,
                                          'n_iter' : n_iter}
    obj.set_assay(sys._getframe().f_code.co_name)

def umap(obj, run_on_PCA=True, name=None, n_neighbors=15, distance='euclidean',
         n_epochs=None, learning_rate=1.0, min_dist=0.1, spread=1.0, seed=None,
         verbose=False, **args):
    """Projects data to a low-dimensional space using the Uniform Manifold Approximation
    and Projection (UMAP) algorithm

    Notes
    -----
    UMAP is a non-linear data reduction algorithm.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    run_on_PCA : `bool`
        To run tSNE on PCA components or not. If False then runs on the entire normalized
        gene expression matrix. Default: True
    name : `str`
        The name of the normalization to operate on. If this is empty or None then the
        function will be applied on all normalizations available.
    n_neighbors : `int`
        The size of local neighborhood (in terms of number of neighboring sample points)
        used for manifold approximation. Larger values result in more global views of the
        manifold, while smaller values result in more local data being preserved.
        In general values should be in the range 2 to 100. Default: 15
    distance : `str`
        The metric to use to compute distances in high dimensional space.
        Default: 'euclidean'
    n_epochs : `int`
        The number of training epochs to be used in optimizing the low dimensional
        embedding. Larger values result in more accurate embeddings. If None is specified
        a value will be selected based on the size of the input dataset (200 for large
        datasets, 500 for small). Default: None
    learning_rate : `float`
        The initial learning rate for the embedding optimization. Default: 1.0
    min_dist : `float`
        The effective minimum distance between embedded points. Default: 0.1
    spread : `float`
        The effective scale of embedded points. Default: 1.0
    seed : `int`
        For reproducibility. Default: None
    verbose : `bool`
        Be verbose. Default: False

    References
    ----------
    .. [1] McInnes L, Healy J, Melville J (2018) UMAP: Uniform Manifold Approximation and
           Projection for Dimension Reduction, https://arxiv.org/abs/1802.03426
    .. [2] https://github.com/lmcinnes/umap
    .. [3] https://umap-learn.readthedocs.io/en/latest/

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if not obj.norm_data:
        raise Exception('Run normalization first before running umap. See here: \
https://oscar-franzen.github.io/adobo/adobo.html#adobo.normalize.norm')
    targets = {}
    if name is None or name == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    if verbose and not run_on_PCA:
        warning('Running UMAP on the entire gene expression matrix is not recommended.')
    for k in targets:
        item = targets[k]
        if not run_on_PCA:
            X = item['data']
        else:
            try:
                X = item['dr']['pca']['comp']
            except KeyError:
                raise Exception('Run `adobo.dr.pca(...)` first.')
        if verbose:
            print('Running UMAP on the %s normalization' % k)
        reducer = um.UMAP(random_state=seed, verbose=verbose, n_neighbors=n_neighbors,
                          metric=distance, n_epochs=n_epochs, learning_rate=learning_rate,
                          min_dist=min_dist, spread=spread, **args)
        emb = reducer.fit_transform(X)
        emb = pd.DataFrame(emb, index=X.index)
        obj.norm_data[k]['dr']['umap'] = {'embedding' : emb}
    obj.set_assay(sys._getframe().f_code.co_name)

def jackstraw(obj, normalization=None, permutations=100, ncomp=15, subset_frac_genes=0.05):
    """Determine the number of relevant PCA components.
    
    Notes
    -----
    Permutes a subset of the data matrix and compares PCA scores with the original.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    normalization : `str`
        The name of the normalization to operate on. If this is empty or None then the
        function will be applied on all normalizations available.
    permutations : `int`
        Number of permutations to run. Default: 100
    ncomp : `int`
        Number of principal components to calculate significance for. Default: 15
    subset_frac_genes : `float`
        Proportion genes to use. Default: 0.10
    verbose : `bool`
        Be verbose. Default: False

    References
    ----------
    .. [1] Chung & Storey (2015) Statistical significance of variables driving
            systematic variation in high-dimensional data, Bioinformatics
            https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4325543/

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if normalization == None or normalization == '':
        norm = list(obj.norm_data.keys())[-1]
    else:
        norm = normalization
    item = obj.norm_data[norm]
    try:
        l = item['dr']['pca']['contr']
    except KeyError:
        raise Exception('Run `adobo.dr.pca(...)` first.')
    if ncomp > X.shape[1]:
        raise Exception('"ncomp" is higher than the number of available components \
computed by adobo.dr.pca(...)')
    X = item['data']
    try:
        hvg = item['hvg']['genes']
    except KeyError:
        raise Exception('Run adobo.dr.find_hvg() first.')
    X = X[X.index.isin(hvg)]
    idx = X.index
    cols = X.columns
    X = sklearn_scale(X.transpose(),  # cells as rows and genes as columns
                      axis=0,         # over genes, i.e. features (columns)
                      with_mean=True, # subtracting the column means
                      with_std=True)  # scale the data to unit variance
    X = pd.DataFrame(X.transpose(), index=idx, columns=cols)
    for perm in np.arange(0,permutations):
        shuffled_genes = sample(list(idx.values), round(len(idx)*subset_frac_genes))
        X_cp = X.copy()
        data_perm = X_cp.loc[shuffled_genes, :]
        # shuffle rows
        data_perm = pd.DataFrame(np.random.permutation(data_perm),
                                 index=data_perm.index, columns=data_perm.columns)
        X_cp.loc[shuffled_genes, :] = data_perm
        comp, contr = irlb(X_cp, ncomp=ncomp)
        q = contr.loc[shuffled_genes, :]
