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

from . import irlbpy

def irlb(data_norm, ncomp=75):
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
    lanc = irlbpy.lanczos(inp, nval=ncomp, maxit=1000, seed=42)
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

def run_PCA(obj, method='irlb', ncomp=75, allgenes=False, verbose=False):
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
        ret = irlb(data, ncomp)
    elif method == 'svd':
        ret = svd(data, ncomp)
    else:
        raise Exception('Unkown PCA method spefified. Valid choices are: irlb and svd')
    obj.dr[method] = ret
    obj.set_assay(sys._getframe().f_code.co_name, method)
