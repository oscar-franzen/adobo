# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
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
from scipy.stats import gaussian_kde
from scipy.stats import norm
from scipy.optimize import minimize

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from .glm.glm import GLM
from .glm.families import Gamma
from ._stats import p_adjust_bh
from ._log import warning

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
    .. [1] https://cran.r-project.org/web/packages/Seurat/index.html

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

def brennecke(data_norm, log, ercc=None, fdr=0.1, ngenes=1000,
              minBiolDisp=0.5, verbose=False):
    """Implements the method of Brennecke et al. (2013) to identify highly variable genes

    Notes
    -----
    Fits data using GLM with Fisher Scoring. GLM code copied from (credits to @madrury
    for this code): https://github.com/madrury/py-glm

    Parameters
    ----------
    data_norm : :class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data.
    log : `bool`
        If normalized data were log transformed or not.
    ercc : :class:`pandas.DataFrame`
        A pandas data frame containing normalized ercc spikes.
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
    .. [1] Brennecke et al. (2013) Nature Methods https://doi.org/10.1038/nmeth.2645

    Returns
    -------
    `list`
        A list containing highly variable genes.
    """
    if type(ercc) != None:
        ercc = data_norm
    if log:
        data_norm = 2**data_norm-1
        ercc = 2**ercc-1
    ercc = ercc.dropna(axis=1, how='all')
    # technical gene (spikes)
    meansSp = ercc.mean(axis=1)
    varsSp = ercc.var(axis=1)
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
    d = {'a0' : [1]*len(meansSp[useForFit]), 'a1tilde' : 1/meansSp[useForFit]}
    x = pd.DataFrame(d)
    # modified to use the identity link function
    gamma_model.fit(np.array(x), y=np.array(cv2Sp[useForFit]))
    a0 = gamma_model.coef_[0]
    a1 = gamma_model.coef_[1]
    psia1theta = a1
    minBiolDisp = minBiolDisp**2
    m = ercc.shape[1]
    cv2th = a0+minBiolDisp+a0*minBiolDisp
    testDenom = (meansGenes*psia1theta+(meansGenes**2)*cv2th)/(1+cv2th/m)
    p = 1-scipy.stats.chi2.cdf(varsGenes*(m-1)/testDenom, m-1)
    res = pd.DataFrame({'gene' : meansGenes.index, 'pvalue' : p})
    res = res[np.logical_not(res.pvalue.isna())]
    res['padj'] = p_adjust_bh(res.pvalue)
    res = res[res['padj'] < fdr]
    if res.shape[0] == 0:
        warning('No highly variable genes identified.')
    res = res.sort_values('pvalue')
    return np.array(res.head(ngenes)['gene'])

def scran(data_norm, log, ngenes=1000, ercc=None):
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
    log : `bool`
        If normalized data were log transformed or not.
    ercc : :class:`pandas.DataFrame`
        A pandas data frame containing normalized ercc spikes.
    ngenes : `int`
        Number of top highly variable genes to return.

    References
    ----------
    .. [1] Lun ATL, McCarthy DJ, Marioni JC (2016). “A step-by-step workflow for low-level
           analysis of single-cell RNA-seq data with Bioconductor.” F1000Research,
           https://doi.org/10.12688/f1000research.9501.2

    Returns
    -------
    `list`
        A list containing highly variable genes.
    """
    if type(ercc) == None:
        raise Exception('adobo.hvg.scran requires ERCC spikes.')
    if log:
        data_norm = 2**data_norm-1
        ercc = 2**ercc-1
    ercc = ercc.dropna(axis=1, how='all')
    means_tech = ercc.mean(axis=1)
    vars_tech = ercc.var(axis=1)
    to_fit = np.log(vars_tech+1)
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

def chen2016(data_norm, log, fdr=0.1, ngenes=1000):
    """
    This function implements the approach from Chen (2016) to identify highly variable
    genes.

    Notes
    -----
    Expression counts should be normalized and on a log scale.

    Parameters
    ----------
    data_norm : :class:`pandas.DataFrame`
        A pandas data frame containing normalized gene expression data.
    log : `bool`
        If normalized data were log transformed or not.
    fdr : `float`
        False Discovery Rate considered significant.
    ngenes : `int`
        Number of top highly variable genes to return.

    References
    ----------
    .. [1] https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-016-2897-6
    .. [2] https://github.com/hillas/scVEGs/blob/master/scVEGs.r

    Returns
    -------
    `list`
        A list containing highly variable genes.
    """
    if log:
        data_norm = 2**data_norm-1
    avg = data_norm.mean(axis=1)
    norm_data = data_norm[avg > 0]
    rows = data_norm.shape[0]
    avg = data_norm.mean(axis=1)
    std = data_norm.std(axis=1)
    cv = std / avg
    xdata = avg
    ydata = np.log10(cv)
    r = np.logical_not(ydata.isna())
    ydata = ydata[r]
    xdata = xdata[r]
    A = np.vstack([np.log10(xdata), np.ones(len(xdata))]).T
    res = np.linalg.lstsq(A, ydata, rcond=None)[0]

    def predict(k, m, x):
        y = k*x+m
        return y
    xSeq = np.arange(min(np.log10(xdata)), max(np.log10(xdata)), 0.005)
    def h(i):
        a = np.log10(xdata) >= (xSeq[i] - 0.05)
        b = np.log10(xdata) < (xSeq[i] + 0.05)
        return np.sum((a & b))
    gapNum = [h(i) for i in range(0, len(xSeq))]
    cdx = np.nonzero(np.array(gapNum) > rows*0.005)[0]
    xSeq = 10 ** xSeq
    ySeq = predict(*res, np.log10(xSeq))
    yDiff = np.diff(ySeq, 1)
    ix = np.nonzero((yDiff > 0) & (np.log10(xSeq[0:-1]) > 0))[0]
    if len(ix) == 0:
        ix = len(ySeq) - 1
    else:
        ix = ix[0]
    xSeq_all = 10**np.arange(min(np.log10(xdata)), max(np.log10(xdata)), 0.001)
    xSeq = xSeq[cdx[0]:ix]
    ySeq = ySeq[cdx[0]:ix]
    reg = LinearRegression().fit(np.log10(xSeq).reshape(-1, 1), ySeq)
    ydataFit = reg.predict(np.log10(xSeq_all).reshape(-1, 1))
    logX = np.log10(xdata)
    logXseq = np.log10(xSeq_all)
    cvDist = []
    for i in range(0, len(logX)):
        cx = np.nonzero((logXseq >= (logX[i] - 0.2)) & (logXseq < (logX[i] + 0.2)))[0]
        tmp = np.sqrt((logXseq[cx] - logX[i])**2 + (ydataFit[cx] - ydata[i])**2)
        tx = np.argmin(tmp)
        if logXseq[cx[tx]] > logX[i]:
            if ydataFit[cx[tx]] > ydata[i]:
                cvDist.append(-1*tmp[tx])
            else:
                cvDist.append(tmp[tx])
        elif logXseq[cx[tx]] <= logX[i]:
            if ydataFit[cx[tx]] < ydata[i]:
                cvDist.append(tmp[tx])
            else:
                cvDist.append(-1*tmp[tx])
    cvDist = np.log(10**np.array(cvDist))
    dor = gaussian_kde(cvDist)
    dor_y = dor(cvDist)
    distMid = cvDist[np.argmax(dor_y)]
    dist2 = cvDist - distMid
    a = dist2[dist2 <= 0]
    b = abs(dist2[dist2 < 0])
    c = distMid
    tmpDist = np.concatenate((a, b))
    tmpDist = np.append(tmpDist, c)
    # estimate mean and sd using maximum likelihood
    distFit = norm.fit(tmpDist)
    pRaw = 1-norm.cdf(cvDist, loc=distFit[0], scale=distFit[1])
    pAdj = p_adjust_bh(pRaw)
    res = pd.DataFrame({'gene': norm_data.index, 'pvalue' : pRaw, 'padj' : pAdj})
    res = res.sort_values(by='pvalue')
    filt = res[res['padj'] < fdr]['gene']
    return np.array(filt.head(ngenes))

def mm(data_norm, log, fdr=0.1, ngenes=1000):
    """
    This function implements the approach from Andrews (2018).

    Notes
    -----
    Input should be normalized but nog log'ed.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame containing normalized counts.
    fdr : `float`
        False Discovery Rate considered significant.
    ngenes : `int`
        Number of top highly variable genes to return.

    References
    ----------
    .. [1] https://doi.org/10.1093/bioinformatics/bty1044
    .. [2] https://github.com/tallulandrews/M3Drop

    Returns
    -------
    `list`
        A list containing highly variable genes.
    """
    if log:
        data_norm = 2**data_norm-1
    ncells = data_norm.shape[1]
    gene_info_p = 1-np.sum(data_norm > 0, axis=1)/ncells
    gene_info_p_stderr = np.sqrt(gene_info_p*(1-gene_info_p)/ncells)
    gene_info_s = data_norm.mean(axis=1)
    gene_info_s_stderr = np.sqrt((np.mean(data_norm**2, axis=1)-gene_info_s**2)/ncells)
    # maximum likelihood estimate of model parameters
    s = gene_info_s
    p = gene_info_p

    def neg_loglike(theta):
        krt = theta[0]
        sigma = theta[1]
        R = p-(1-(s/(krt+s)))
        R = norm.logpdf(R, 0, sigma)
        return -1*np.sum(R)

    theta_start = np.array([3, 0.25])
    res = minimize(neg_loglike, theta_start, method='Nelder-Mead', options={'disp': True})
    est = res.x
    krt = est[0]
    # testing step
    p_obs = gene_info_p
    always_detected = p_obs == 0
    p_obs[p_obs == 0] = min(p_obs[p_obs > 0])/2
    p_err = gene_info_p_stderr
    S_mean = gene_info_s
    K_equiv = p_obs*S_mean/(1-p_obs)
    S_err = gene_info_s_stderr
    K_equiv_err = abs(K_equiv)*np.sqrt((S_err/S_mean)**2 + (p_err/p_obs)**2)
    K_equiv_log = np.log(K_equiv)
    thing = K_equiv-K_equiv_err
    thing[thing <= 0] = 10**-100
    K_equiv_err_log = abs(np.log(thing)-K_equiv_log)
    K_equiv_err_log[K_equiv-K_equiv_err <= 0] = 10**10
    K_obs_log = np.log(krt)
    K_err_log = np.std(K_equiv_log-K_obs_log)/np.sqrt(len(K_equiv_log))
    Z = (K_equiv_log - K_obs_log)/np.sqrt(K_equiv_err_log**2+K_err_log**2)
    pval = 1 - norm.cdf(Z)
    pval[always_detected] = 1
    res = pd.DataFrame({'gene': data_norm.index, 'pvalue' : pval})
    res = res[np.logical_not(res.pvalue.isna())]
    res['padj'] = p_adjust_bh(res.pvalue)
    res = res.sort_values('pvalue')
    return res.head(ngenes)['gene']

def find_hvg(obj, method='seurat', normalization=None, ngenes=1000, fdr=0.1,
             use_combat=False, verbose=False):
    """Finding highly variable genes

    Notes
    -----
    A wrapper function around the individual HVG functions, which can also be called
    directly.
    
    The method 'brennecke' should not be applied on 'fqn' normalized data.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : `{'seurat', 'brennecke', 'scran', 'chen2016', 'mm'}`
        Specifies the method to be used.
    normalization : `str`
        The name of the normalization to operate on. If this is empty or None then the
        function will be applied on all normalizations available.
    ngenes : `int`
        Number of genes to return.
    fdr : `float`
        False Discovery Rate threshold for significant genes applied to those methods
        that use it (brennecke, chen2016, mm). Note that the number of returned genes
        might be fewer than specified by `ngenes` because of FDR consideration.
    use_combat : `bool`
        Use combat-adjusted data. Default: False
    verbose : `bool`
        Be verbose or not.

    Example
    -------    
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp)
    >>> ad.hvg.find_hvg(exp)

    References
    ----------
    .. [1] Yip et al. (2018) Briefings in Bioinformatics
           https://academic.oup.com/bib/advance-article/doi/10.1093/bib/bby011/4898116

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if not obj.norm_data:
        raise Exception('Run normalization first before running find_hvg. See here: \
https://oscar-franzen.github.io/adobo/adobo.html#adobo.normalize.norm')
    targets = {}
    norm = normalization
    if norm is None or norm == '':
        targets = obj.norm_data
    else:
        targets[norm] = obj.norm_data[norm]
    # remove previous cluster analysis, b/c this changes after running hvg
    obj.delete(('clusters', 'dr'))
    for k in targets:
        item = targets[k]
        if verbose:
            print('Running on the %s normalization' % k)
        if use_combat:
            if verbose:
                print('Using data from ComBat.')
            data = item['combat']
        else:
            data = item['data']
        data_ercc = item.get('norm_ercc', None)
        log = item['log']
        if method == 'seurat':
            hvg = seurat(data, ngenes)
        elif method == 'brennecke':
            hvg = brennecke(data_norm=data, log=log, ercc=data_ercc, fdr=fdr,
                            ngenes=ngenes, minBiolDisp=0.5, verbose=verbose)
        elif method == 'scran':
            hvg = scran(data, log, ngenes, data_ercc)
        elif method == 'chen2016':
            hvg = chen2016(data, log, fdr, ngenes)
        elif method == 'mm':
            hvg = mm(data, log, fdr, ngenes)
        else:
            raise Exception('Unknown HVG method specified. Valid choices are: seurat, \
brennecke, scran, chen2016 and mm')
        obj.norm_data[k]['hvg'] = {'genes' : hvg, 'method' : method}
    obj.set_assay(sys._getframe().f_code.co_name, method)
