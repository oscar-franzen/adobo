# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for detection of highly variable genes.
"""
import sys
import pandas as pd
import numpy as np

import scipy.stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from .stats import p_adjust_bh
from .glm.glm import GLM
from .glm.families import Gamma
from .log import warning

import warnings
warnings.filterwarnings("ignore")

def seurat(data, ngenes=1000, num_bins=20):
    """Retrieves a list of highly variable genes using Seurat's strategy
    
    Notes
    -----
    The function bins the genes according to average expression, then calculates
    dispersion for each bin as variance to mean ratio. Within each bin, Z-scores are
    calculated and returned. Z-scores are ranked and the top 1000 are selected. Input data
    should be normalized first.

    Parameters
    ----------
    obj : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes, columns=cells).
    ngenes : `int`
        Number of top highly variable genes to return.
    num_bins : `int`
        Number of bins to use.

    References
    ----------
    https://cran.r-project.org/web/packages/Seurat/index.html

    Returns
    -------
    `list`
        A list containing highly variable genes.
    """
    # number of bins
    gene_mean = data.mean(axis=1)
    # equal width (not size) of bins
    bins = pd.cut(gene_mean, num_bins)
    ret = []
    for _, sliced in data.groupby(bins):
        dispersion = sliced.var(axis=1)/sliced.mean(axis=1)
        zscores = (dispersion-dispersion.mean())/dispersion.std()
        ret.append(zscores)

    ret = pd.concat(ret)
    ret = ret.sort_values(ascending=False)
    top_hvg = ret.head(ngenes)
    ret = np.array(top_hvg.index)
    return ret

def brennecke(data_norm, log2, ERCC=pd.DataFrame(), fdr=0.1, minBiolDisp=0.5, ngenes=1000,
              verbose=False):
    """Implements the method of Brennecke et al. (2013) to identify highly variable genes
    
    Notes
    -----
    Fits data using GLM with Fisher Scoring. GLM code copied from (credits to @madrury
    for this code): https://github.com/madrury/py-glm

    Parameters
    ----------
    data_norm : :class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data.
    log2 : `bool`
        If normalized data were log2 transformed or not.
    ERCC : :class:`pandas.DataFrame`
        A pandas data frame containing normalized ERCC spikes.
    fdr : `float`
        False Discovery Rate considered significant.
    minBiolDisp : `float`
        Minimum percentage of variance due to biological factors.
    ngenes : `int`
        Number of top highly variable genes to return.
    verbose : `bool`
        Be verbose or not.

    References
    ----------
    Brennecke et al. (2013) Nature Methods https://doi.org/10.1038/nmeth.2645

    Returns
    -------
    `list`
        A list containing highly variable genes.
    """
    if ERCC.shape[0] == 0:
        ERCC = data_norm
    if log2:
        data_norm = 2**data_norm-1
        ERCC = 2**ERCC-1
    ERCC = ERCC.dropna(axis=1, how='all')

    # technical gene (spikes)
    meansSp = ERCC.mean(axis=1)
    varsSp = ERCC.var(axis=1)
    cv2Sp = varsSp/meansSp**2
    # biological genes
    meansGenes = data_norm.mean(axis=1)
    varsGenes = data_norm.var(axis=1)
    cv2Genes = varsGenes/meansGenes**2

    minMeanForFit = np.quantile(meansSp[cv2Sp > 0.3], 0.8)
    useForFit = meansSp >= minMeanForFit
    
    if np.sum(useForFit) < 20:
        meansAll = data_norm.mean(axis=1).append(meansSp)
        cv2All = cv2Genes.append(cv2Sp)
        minMeanForFit = np.quantile(meansAll[cv2All > 0.3], 0.8)
        useForFit = meansSp >= minMeanForFit
        if verbose:
            print('Using all genes because "useForFit" < 20.')
    n = np.sum(useForFit)
    if n < 30:
        warning('Only %s spike-ins to be used in fitting, may result in poor fit.' % n)

    gamma_model = GLM(family=Gamma())
    x = pd.DataFrame({'a0' : [1]*len(meansSp[useForFit]), 'a1tilde' : 1/meansSp[useForFit]}) 
    # modified to use the identity link function
    gamma_model.fit(np.array(x), y=np.array(cv2Sp[useForFit]))
    a0 = gamma_model.coef_[0]
    a1 = gamma_model.coef_[1]

    psia1theta = a1
    minBiolDisp = minBiolDisp**2

    m = ERCC.shape[1]
    cv2th = a0+minBiolDisp+a0*minBiolDisp

    testDenom = (meansGenes*psia1theta+(meansGenes**2)*cv2th)/(1+cv2th/m)
    p = 1-scipy.stats.chi2.cdf(varsGenes*(m-1)/testDenom, m-1)
    res = pd.DataFrame({'gene' : meansGenes.index, 'pvalue' : p})
    res = res[np.logical_not(res.pvalue.isna())]
    res['padj'] = p_adjust_bh(res.pvalue)
    res = res[res['padj'] < fdr]
    res = res.sort_values('pvalue')
    return np.array(res.head(ngenes)['gene'])

def scran(data_norm, ERCC, log2, ngenes=1000):
    """This function implements the approach from the scran R package
    
    Notes
    -----
    Expression counts should be normalized and on a log scale.

    Outline of the steps:

    1. fits a polynomial regression model to mean and variance of the technical genes
    2. decomposes the total variance of the biological genes by subtracting the
       technical variance predicted by the fit
    3. sort based on biological variance

    Parameters
    ----------
    data_norm : :class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data.
    ERCC : :class:`pandas.DataFrame`
        A pandas data frame containing normalized ERCC spikes.
    log2 : `bool`
        If normalized data were log2 transformed or not.
    ngenes : `int`
        Number of top highly variable genes to return.

    References
    ----------
    Lun ATL, McCarthy DJ, Marioni JC (2016). “A step-by-step workflow for low-level
    analysis of single-cell RNA-seq data with Bioconductor.” F1000Research,
    doi.org/10.12688/f1000research.9501.2

    Returns
    -------
    `list`
        A list containing highly variable genes.
    """
    if ERCC.shape[0] == 0:
        raise Exception('scran() for HVG requires ERCC spikes.')
    if log2:
        data_norm = 2**data_norm-1
        ERCC = 2**ERCC-1
    ERCC = ERCC.dropna(axis=1, how='all')
    means_tech = ERCC.mean(axis=1)
    vars_tech = ERCC.var(axis=1)

    to_fit = np.log(vars_tech)
    arr = [list(item) for item in zip(*sorted(zip(means_tech, to_fit)))]

    x = arr[0]
    y = arr[1]

    poly_reg = PolynomialFeatures(degree=4)
    x_poly = poly_reg.fit_transform(np.array(x).reshape(-1, 1))
    pol_reg = LinearRegression()
    pol_reg.fit(x_poly, y)

    #plt.scatter(x, y, color='red')
    #plt.plot(x, pol_reg.predict(poly_reg.fit_transform(np.array(x).reshape(-1,1))), color='blue')
    #plt.xlabel('mean')
    #plt.ylabel('var')
    #plt.show()

    # predict and remove technical variance
    bio_means = data_norm.mean(axis=1)
    vars_pred = pol_reg.predict(poly_reg.fit_transform(np.array(bio_means).reshape(-1, 1)))
    vars_bio_total = data_norm.var(axis=1)

    # biological variance component
    vars_bio_bio = vars_bio_total - vars_pred
    vars_bio_bio = vars_bio_bio.sort_values(ascending=False)
    return vars_bio_bio.head(ngenes).index.values

def find_hvg(obj, method='seurat', ngenes=1000, verbose=False):
    """Finding highly variable genes
    
    Notes
    -----
    A wrapper function around the individual HVG functions, which can also be called
    directly.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : `{'seurat', 'brennecke'}`
        Specifies the method to be used.
    ngenes : `int`
        Number of genes to return.
    verbose : `bool`
        Be verbode or not.

    References
    ----------
    Yip et al. (2018) Briefings in Bioinformatics
    https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bby011/4898116
    
    See Also
    --------
    seurat

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    data = obj.norm
    data_ERCC = obj.norm_ERCC
    
    if method == 'seurat':
        hvg = seurat(data, ngenes)
    elif method == 'brennecke':
        hvg = brennecke(data, obj.norm_log2, data_ERCC, ngenes, verbose)
    obj.hvg = hvg
    obj.set_assay(sys._getframe().f_code.co_name, method)
