# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Statistical helper functions.
"""

import pandas as pd
import numpy as np
import scipy.special
from statsmodels import robust

def bw_nrd(x):
    """Selects bandwidth for gaussian kernels

    Parameters
    ----------
    x : `list`
        An input list of data points.

    References
    ----------
    .. [1] Scott, D. W. (1992) _Multivariate Density Estimation: Theory,
           Practice, and Visualization._ New York: Wiley.

    Returns
    -------
    int
        KDE bandwidth.
    """

    r = np.quantile(x, (0.25, 0.75))
    h = float(r[1]-r[0])/1.34
    bw = 1.06 * min(np.sqrt(np.var(x)), h) * len(x)**(-1/5)
    return bw

def row_geometric_mean(mat, eps=1):
    """Calculates the geometric mean for every row in a data frame

    Parameters
    ----------
    mat : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes,
        columns=cells).
    eps : float
        A small constant to avoid log(0)=-inf (default: 1).

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Geometric_mean

    Returns
    -------
    :class:`pandas.Series`
        Computed values.
    """
    return mat.apply(lambda x: np.exp(np.mean(np.log(x+eps)))-eps, axis=1)

def theta_ml(y, mu, limit=10, eps=np.finfo(float).eps, verbose=False):
    """Estimates theta of the Negative Binomial Distribution using maximum likelihood

    Notes
    -----
    Adapted from the theta.ml function of the R package MASS.

    Parameters
    ----------
    y : `list`
        List of observed values from the negative binomial.
    mu : `list`
        Estimated mean vector.
    limit : `int`
        Maximum number of iterations (default: 10).

    Returns
    -------
    float
        Estimated theta.
    """
    digamma = scipy.special.digamma
    trigamma = lambda x: scipy.special.polygamma(1, x)
    log = np.log

    def score(n, th, mu, y, w):
        return sum(w*(digamma(th+y)-digamma(th)+log(th)+1-log(th+mu)-(y+th)/(mu+th)))

    def info(n, th, mu, y, w):
        return sum(w*(-trigamma(th+y)+trigamma(th)-1/th+2/(mu+th)-(y+th)/(mu+th)**2))

    weights = [1]*len(y)
    n = sum(weights)
    t0 = n/sum(weights*(y/mu-1)**2)
    it = 0
    del_ = 1

    while it < limit and np.abs(del_) > eps:
        it = it+1
        t0 = abs(t0)
        i = info(n, t0, mu, y, weights)
        del_ = score(n, t0, mu, y, weights)/i
        t0 = t0 + del_

    if t0 < 0 and verbose:
        print('theta_ml(): estimate truncated at zero')
    if it == limit and verbose:
        print('theta_ml(): iteration limit reached')
    return t0

def _robust_scale(x):
    return (x-np.median(x))/(robust.mad(x)+np.finfo(float).eps)

def _robust_scale_binned(y, x, breaks):
    bins = pd.cut(x, breaks)
    l = []
    for item in pd.DataFrame({'y' : y.values, 'x' : x.values}).groupby(bins.values):
        l.append(_robust_scale(item[1]['y']))
    z = pd.concat(l)
    return z.sort_index()

def is_outlier(y, x, thres=10):
    bin_width = (max(x)-min(x))*bw_nrd(x)/2
    eps_ = np.finfo(float).eps*10
    breaks1 = np.linspace(min(x)-eps_, max(x)+bin_width, 50, endpoint=False)
    breaks2 = np.linspace(min(x)-eps_-bin_width/2, max(x)+bin_width, 50, endpoint=False)
    score1 = _robust_scale_binned(y, x, breaks1)
    score2 = _robust_scale_binned(y, x, breaks2)
    return np.minimum(abs(score1), abs(score2)) > thres

def p_adjust_bh(p):
    """The Benjamini-Hochberg p-value correction for multiple hypothesis testing.

    Parameters
    ----------
    p : `list`
        A list of p-values.

    References
    ----------
    .. [1] Benjamini & Hochberg (1995) Controlling the false discovery rate: a practical
        and powerful approach to multiple testing.  J Royal Statistical Society, Series B

    Returns
    -------
    int
        Adjusted p-values.
    """
    p = np.asfarray(p)
    p = np.ma.array(p, mask=np.isnan(p)) # to handle nan
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(float(len(p)), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    q = q[by_orig]
    q[np.isnan(p.data)] = np.nan # put nan back
    return q.data
