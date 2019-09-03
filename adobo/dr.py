# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for dimensional reduction.
"""
import sys
import pandas as pd
import numpy as np
import scipy.linalg
from sklearn.preprocessing import scale as sklearn_scale
import sklearn.manifold
import umap as um

from . import irlbpy

def irlb(data_norm, ncomp=75, seed=42):
    """Truncated SVD by implicitly restarted Lanczos bidiagonalization
    
    Notes
    -----
    The augmented implicitly restarted Lanczos bidiagonalization algorithm (IRLBA) finds
    a few approximate largest singular values and corresponding singular vectors using a
    method of Baglama and Reichel.
    
    Parameters
    ----------
    data_norm : :class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data.
    ncomp : `int`
        Number of components to return, optional.
    seed : `int`
        For reproducibility, optional.
    
    References
    ----------
    Baglama et al (2005) Augmented Implicitly Restarted Lanczos Bidiagonalization Methods
    SIAM Journal on Scientific Computing
    
    https://github.com/bwlewis/irlbpy
    
    Returns
    -------
    `pd.DataFrame`
        A py:class:`pandas.DataFrame` containing the components (columns).
    """
    inp = data_norm
    lanc = irlbpy.lanczos(inp, nval=ncomp, maxit=1000, seed=seed)
    # weighing by var
    comp = np.dot(lanc.V, np.diag(lanc.s))
    comp = pd.DataFrame(comp)
    return comp

def svd(data_norm, ncomp=75):
    """PCA via SVD
    
    Parameters
    ----------
    data_norm : :class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data.
    ncomp : `int`
        Number of components to return, optional.
    
    References
    ----------
    https://stats.stackexchange.com/questions/79043/why-pca-of-data-by-means-of-svd-of-the-data
    
    Returns
    -------
    `pd.DataFrame`
        A py:class:`pandas.DataFrame` containing the components (columns).
    """
    inp = data_norm
    inp = inp.transpose()
    s = scipy.linalg.svd(inp)
    v = s[2].transpose()
    d = s[1]
    s_d = d/np.sqrt(inp.shape[0]-1)
    retx = inp.dot(v)
    retx = retx[:, 0:ncomp]
    comp = retx
    comp = pd.DataFrame(comp)
    return comp

def pca(obj, method='irlb', ncomp=75, allgenes=False, scale=True, verbose=False, seed=42):
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
    ncomp : `int`
        Number of components to return. Default: 75
    allgenes : `bool`
        Use all genes instead of only HVG. Default: False
    scale : `bool`
        Scales input data prior to PCA. Default: True
    verbose : `bool`
        Be noisy or not. Default: False
    seed : `int`
        For reproducibility (only irlb). Default: 42

    References
    ----------
    https://en.wikipedia.org/wiki/Principal_component_analysis

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    data = obj.norm
    hvg = obj.hvg
    if len(hvg)>0 and not allgenes:
        data = data[data.index.isin(hvg)]
        if verbose:
            print('Only using HVG genes (%s).' % data.shape[0])
    data = data.transpose() # cells as rows and genes as labels
    if scale:
        data = sklearn_scale(data, axis=0,   # over genes, i.e. features (columns)
                             with_mean=True, # subtracting the column means
                             with_std=True)  # scale the data to unit variance
        data = data.transpose()
    if allgenes and verbose:
        print('Using all genes (%s).' % data.shape[0])
    if method == 'irlb':
        ret = irlb(data, ncomp, seed)
    elif method == 'svd':
        ret = svd(data, ncomp)
    else:
        raise Exception('Unkown PCA method spefified. Valid choices are: irlb and svd')
    ret.index = obj.norm.columns
    obj.dr[method] = ret
    obj.set_assay(sys._getframe().f_code.co_name, method)

def tsne(obj, target='irlb', perplexity=30, n_iter=2000, seed=42, verbose=False, **args):
    """
    Projects data to a two dimensional space using the tSNE algorithm.
    
    Notes
    -----
    Calls :py:func:`sklearn.manifold.TSNE`.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    target : `{'irlb', 'svd', 'norm'}`
        What to run tSNE on.
    perplexity : `float`
        From [1]: The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets usually require
        a larger perplexity. Consider selecting a value between 5 and 50. Different
        values can result in significanlty different results.
    n_iter : `int`
        Number of iterations.
    seed : `int`
        For reproducibility.
    verbose : `bool`
        Be verbose.

    References
    ----------
    [0] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
    Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if not target in ('irlb', 'svd', 'norm'):
        raise Exception('target can be one of: "irlb", "svd" or "norm"')
    if target == 'norm':
        X = obj.norm
    else:
        if not target in obj.dr:
            raise Exception('%s was not found, please run adobo.dr.pca(...) first.')
        else:
            X = obj.dr[target]
    tsne = sklearn.manifold.TSNE(n_components=2,
                                 n_iter=n_iter,
                                 perplexity=perplexity,
                                 random_state=seed,
                                 verbose=verbose,
                                 **args)
    emb = tsne.fit_transform(X)
    obj.dr['tsne'] = emb
    obj.set_assay(sys._getframe().f_code.co_name)

def umap(obj, target='irlb', seed=42, **args):
    """
    Projects data to a two dimensional space using the UMAP algorithm.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    target : `{'irlb', 'svd', 'norm'}`
        What to run tSNE on.
    seed : `int`
        For reproducibility.
    verbose : `bool`
        Be verbose.

    References
    ----------
    McInnes L, Healy J, Melville J, arxiv, 2018

    https://arxiv.org/abs/1802.03426
    https://github.com/lmcinnes/umap
    https://umap-learn.readthedocs.io/en/latest/

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if target == 'norm':
        X = obj.norm
    else:
        if not target in obj.dr:
            raise Exception('%s was not found, please run adobo.dr.pca(...) first.')
        else:
            X = obj.dr[target]
    reducer = um.UMAP(random_state=seed, **args)
    emb = reducer.fit_transform(X)
    emb = pd.DataFrame(emb)
    obj.dr['umap'] = emb
    obj.set_assay(sys._getframe().f_code.co_name)
