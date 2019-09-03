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
from sklearn.preprocessing import scale
import sklearn.manifold

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
        Number of components to return.
    seed : `int`
        For reproducibility.
    
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
        Number of components to return.
    
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
    x = scale(inp, with_mean=True, with_std=False)
    s = scipy.linalg.svd(x)
    v = s[2].transpose()
    d = s[1]
    s_d = d/np.sqrt(x.shape[0]-1)
    retx = x.dot(v)
    retx = retx[:, 0:ncomp]
    comp = retx
    comp = pd.DataFrame(comp)
    return comp

def pca(obj, method='irlb', ncomp=75, allgenes=False, verbose=False, seed=42):
    """Principal Component Analysis
    
    Notes
    -----
    A wrapper function around the individual normalization functions, which can also be
    called directly.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : `{'irlb', 'svd'}`
        Method to use for PCA. This does not matter much. Default: irlb
    ncomp : `int`
        Number of components to return. Default: 75
    allgenes : `bool`
        Use all genes instead of only HVG.
    verbose : `bool`
        Be noisy or not.
    seed : `int`
        For reproducibility (only irlb).

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
    if allgenes and verbose:
        print('Using all genes (%s).' % data.shape[0])
    if method == 'irlb':
        ret = irlb(data, ncomp, seed)
    elif method == 'svd':
        ret = svd(data, ncomp)
    else:
        raise Exception('Unkown PCA method spefified. Valid choices are: irlb and svd')
    ret.index = data.columns
    obj.dr[method] = ret
    obj.set_assay(sys._getframe().f_code.co_name, method)

def tsne(obj, target='irlb', perplexity=30, n_iter=2000, seed=42):
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

    References
    ----------
    [0] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
    Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.
    
    [1] https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if not ('irlb', 'svd', 'norm') in target:
        raise Exception('target can be one of: "irlb", "svd" or "norm"')
    if target == 'norm':
        X = obj.norm
    else:
        if not target in obj.dr:
            raise Exception('%s was not found, please run run_PCA(...) first.')
    #tsne = sklearn.manifold.TSNE(n_components=2,
    #                             n_iter=n_iter,
    #                             perplexity=perplexity,
    #                             random_state=seed)
    #self.embeddings = tsne.fit_transform(self.pca_components)
    #self.embeddings = pd.DataFrame(self.embeddings,
    #                               index=self.pca_components.index,
    #                               columns=[1, 2])
