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

import pandas as pd
import numpy as np

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
    [0] https://cran.r-project.org/web/packages/Seurat/index.html

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

def brennecke(data_norm, norm_ERCC=None, fdr=0.1, minBiolDisp=0.5):
    """Implements the method of Brennecke et al. (2013) to identify highly variable genes
    
    Notes
    -----
    Fits data using GLM with Fisher Scoring. GLM code copied from (credits to @madrury
    for this code): https://github.com/madrury/py-glm

    Parameters
    ----------
    data_norm : :class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data
        (rows=genes, columns=cells).
    norm_ERCC : :class:`pandas.DataFrame`
        A pandas data frame containing normalized ERCC spikes.
    ngenes : `int`
        Number of top highly variable genes to return.
    num_bins : `int`
        Number of bins to use.

    References
    ----------
    [0] Brennecke et al. (2013) Nature Methods https://doi.org10.1038/nmeth.2645

    Returns
    -------
    `list`
        A list containing highly variable genes.
    """
    if norm_ERCC == None:
        norm_ERCC = data_norm

    data_norm = 2**data_norm-1
    norm_ERCC = 2**norm_ERCC-1

    norm_ERCC = norm_ERCC.dropna(axis=1, how='all')

    # technical gene (spikes)
    meansSp = norm_ERCC.mean(axis=1)
    varsSp = norm_ERCC.var(axis=1)
    cv2Sp = varsSp/meansSp**2

    # biological genes
    meansGenes = data_norm.mean(axis=1)
    varsGenes = data_norm.var(axis=1)

    minMeanForFit = np.quantile(meansSp[cv2Sp > 0.3], 0.8)
    useForFit = meansSp >= minMeanForFit

    if np.sum(useForFit) < 20:
        meansAll = data_norm.mean(axis=1)
        cv2All = data_norm.var(axis=1)
        minMeanForFit = np.quantile(meansAll[cv2All > 0.3], 0.8)
        useForFit = meansSp >= minMeanForFit

    gamma_model = GLM(family=Gamma())

    x = pd.DataFrame({'a0' : [1]*len(meansSp[useForFit]), 'a1tilde' : 1/meansSp[useForFit]})

    # modified to use the identity link function
    gamma_model.fit(np.array(x), y=np.array(cv2Sp[useForFit]))
    a0 = gamma_model.coef_[0]
    a1 = gamma_model.coef_[1]

    psia1theta = a1
    minBiolDisp = minBiolDisp**2

    m = norm_ERCC.shape[1]
    cv2th = a0+minBiolDisp+a0*minBiolDisp

    testDenom = (meansGenes*psia1theta+(meansGenes**2)*cv2th)/(1+cv2th/m)
    q = varsGenes * (m - 1)/testDenom

    p = 1-scipy.stats.chi2.cdf(q, m-1)
    padj = p_adjust_bh(p)
    res = pd.DataFrame({'gene': meansGenes.index, 'pvalue' : p, 'padj' : padj})
    filt = res[res['padj'] < fdr]['gene']

    return np.array(filt.head(self.hvg_n))

def find_hvg(obj, method='seurat', ngenes=1000):
    """Finding highly variable genes
    
    Notes
    -----
    A wrapper function around the individual HVG functions, which can also be called
    directly.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : `{'seurat'}`
        Specifies the method to be used.

    References
    ----------
    [0] Yip et al. (2018) Briefings in Bioinformatics
        https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bby011/4898116
    
    See Also
    --------
    seurat

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    
    data = obj.norm
    data_ERCC = obj._exp_ERCC
    
    if method == 'seurat':
        hvg = seurat(data, ngenes)
    elif method == 'brennecke':
        hvg = brennecke(data, data_ERCC)
    
    obj.hvg = hvg
    obj.set_assay(sys._getframe().f_code.co_name, method)
