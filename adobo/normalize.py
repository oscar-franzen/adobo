# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
This module contains functions to normalize raw read counts.
"""

import sys
import time
from multiprocessing import Pool
import psutil
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg
import patsy
from tqdm import tqdm

from ._stats import bw_nrd, row_geometric_mean, theta_ml, is_outlier


def _vsn_model_pars(X, g, data_step1):
    y = X.loc[g, :]
    mod = sm.GLM(y, sm.add_constant(data_step1['log_umi']),
                 family=sm.families.Poisson())
    res = mod.fit()
    s = res.summary()
    mu = res.fittedvalues
    theta = theta_ml(y, mu)
    coef = res.params
    return{'gene': g,
           'theta': theta,
           'log_umi': coef['log_umi'],
           'const': coef['const']}


def vsn(data, min_cells=5, gmean_eps=1, ngenes=2000, nworkers='auto',
verbose=False):
    """Performs variance stabilizing normalization based on a negative
    binomial regression model with regularized parameters

    Notes
    -----
    Use only with UMI counts. Adopts a subset of the functionality of
    `vst` in the R package `sctransform`.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts
        (rows=genes, columns=cells).
    min_cells : `int`
        Minimum number of cells expressing a gene for the gene to be
        used. Default: 10
    gmean_eps : `float`
        A small constant to avoid log(0)=-Inf. Default: 1
    ngenes : `int`
        Number of genes to use when estimating parameters. Default:
        2000
    nworkers : `int` or `{'auto'}`
        If a string, then the only accepted value is 'auto', and the
        number of worker processes will be the total number of
        detected physical cores. If an integer then it specifies the
        number of worker processes. Default: 'auto'
    verbose : `bool`
        Be verbose or not. Default: False

    References
    ----------
    .. [1] https://cran.r-project.org/web/packages/sctransform/index.html
    .. [2] https://www.biorxiv.org/content/10.1101/576827v1

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp, method='vsn')

    Returns
    -------
    :class:`pandas.DataFrame`
        A data matrix with adjusted counts.
    """
    start_time = time.time()
    if type(nworkers) == str:
        if nworkers == 'auto':
            nworkers = psutil.cpu_count(logical=False)
        else:
            raise Exception('Invalid value for parameter "nworkers".')
    if verbose:
        print('%s worker processes will be used' % nworkers)
    bw_adjust = 3  # Kernel bandwidth adjustment factor
    # numericals functions
    log10 = np.log10
    log = np.log
    exp_ = np.exp
    mean = np.mean
    sqrt = np.sqrt
    # data summary
    cell_attr = pd.DataFrame({'counts': data.sum(axis=0),
                              'genes': (data > 0).sum(axis=0)})
    cell_attr['log_umi'] = log10(cell_attr.counts)
    cell_attr['log_gene'] = log10(cell_attr.genes)
    cell_attr['umi_per_gene'] = cell_attr.counts/cell_attr.genes
    cell_attr['log_umi_per_gene'] = log10(cell_attr.umi_per_gene)

    genes_cell_count = (data > 0).sum(axis=1)
    genes = genes_cell_count[genes_cell_count >= min_cells].index
    X = data[data.index.isin(genes)]
    genes_log_gmean = log10(row_geometric_mean(X, gmean_eps))

    cells_step1 = X.columns
    genes_step1 = X.index
    genes_log_gmean_step1 = genes_log_gmean
    data_step1 = cell_attr

    if ngenes < len(genes_step1):
        bw = bw_nrd(genes_log_gmean_step1)
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        ret = kde.fit(genes_log_gmean_step1[:, None])
        # TODO: score_samples is slower than density() in R
        weights = 1/exp_(kde.score_samples(genes_log_gmean_step1[:, None]))
        genes_step1 = np.random.choice(X.index, ngenes, replace=False,
                                       p=weights/sum(weights))
        X = X.loc[X.index.isin(genes_step1), X.columns.isin(cells_step1)]
        X = X.reindex(genes_step1)
        genes_log_gmean_step1 = log10(row_geometric_mean(X, gmean_eps))

    pool = Pool(nworkers)
    model_pars = []
    pbar = tqdm(total=len(genes_step1))

    def _update_results(y):
        model_pars.append(y)
        pbar.update()

    for g in genes_step1:
        args = (X, g, data_step1)
        x = pool.apply_async(_vsn_model_pars, args=args,
                             callback=_update_results)
        # print(x.get())

    pool.close()
    pool.join()

    model_pars = pd.DataFrame(model_pars)
    model_pars.index = model_pars['gene']
    model_pars = model_pars.drop('gene', axis=1)
    model_pars.theta = log10(model_pars.theta)

    # remove outliers
    outliers = model_pars.apply(
        lambda x: is_outlier(x, genes_log_gmean_step1), axis=0)
    outliers = outliers.any(axis=1)
    model_pars = model_pars[np.logical_not(outliers.values)]

    genes_step1 = model_pars.index.values
    genes_log_gmean_step1 = genes_log_gmean_step1[np.logical_not(
        outliers.values)]

    x_points = np.maximum(genes_log_gmean, min(genes_log_gmean_step1))
    x_points = np.minimum(x_points, max(genes_log_gmean_step1))

    model_pars_fit = model_pars.apply(
        lambda x: KernelReg(x,
                            genes_log_gmean_step1,
                            bw='aic',
                            var_type='c').fit(x_points)[0], axis=0)

    model_pars_fit.index = x_points.index
    model_pars.theta = 10**model_pars.theta
    model_pars_fit.theta = 10**model_pars_fit.theta
    model_pars_final = model_pars_fit

    regressor_data_final = pd.DataFrame({'const': [1]*len(cell_attr['log_umi']),
                                         'log_umi': cell_attr['log_umi']})

    coefs = model_pars_final.loc[:, ('const', 'log_umi')]
    mu = exp_(np.dot(coefs, regressor_data_final.transpose()))
    y = data[data.index.isin(model_pars_final.index)]

    # pearson residuals
    mu2 = pd.DataFrame(mu**2)
    t = (y-mu)
    variance = mu+mu2.div(model_pars_final.loc[:, 'theta'].values, axis=0)
    n = sqrt(variance)

    n.index = t.index
    n.columns = t.columns
    pr = t/n

    coefs = model_pars_final.loc[:, ('const', 'log_umi')]
    theta = model_pars_final.loc[:, ('theta')]
    med = np.median(cell_attr['log_umi'])

    # correct counts
    regressor_data = pd.DataFrame({'const': [1]*len(cell_attr['log_umi']),
                                   'log_umi': [med]*len(cell_attr['log_umi'])})
    mu = exp_(np.dot(coefs, regressor_data.transpose()))
    variance = mu+pd.DataFrame(mu**2).div(theta.values, axis=0)
    variance.index = pr.index
    variance.columns = pr.columns
    corrected_data = mu + pr * sqrt(variance)
    y = abs(round(corrected_data))
    end_time = time.time()
    if verbose:
        print('Analysis took %.2f minutes' % ((end_time-start_time)/60))
    return y


def clr(data, axis='genes'):
    """Performs centered log ratio normalization similar to Seurat

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts
        (rows=genes, columns=cells).
    axis : {'genes', 'cells'}
        Normalize over genes or cells. Default: 'genes'

    References
    ----------
    .. [1] Hafemeister et al. (2019) https://www.biorxiv.org/content/10.1101/576827v1

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp, method='clr')

    Returns
    -------
    :class:`pandas.DataFrame`
        A normalized data matrix with same dimensions as before.
    """

    if axis == 'genes':
        axis = 1
    elif axis == 'cells':
        axis = 0
    else:
        raise Exception('Unknown axis specified.')

    r = data.apply(lambda x: np.log1p(
        x/np.exp(sum(np.log1p(x[x > 0]))/len(x))), axis=axis)
    return r


def standard(data, scaling_factor=10000):
    """Performs a standard normalization by scaling with the total
    read depth per cell and then multiplying with a scaling factor.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts
        (rows=genes, columns=cells).
    scaling_factor : `int`
        Scaling factor used to multiply the scaled counts
        with. Default: 10000

    References
    ----------
    .. [1] Evans et al. (2018) Briefings in Bioinformatics
           https://academic.oup.com/bib/article/19/5/776/3056951

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp, method='standard')

    Returns
    -------
    :class:`pandas.DataFrame`
        A normalized data matrix with same dimensions as before.
    """
    col_sums = data.sum(axis=0).values
    data_norm = (data / col_sums) * scaling_factor
    return data_norm


def rpkm(data, gene_lengths):
    """Normalize expression values as RPKM

    Notes
    -----
    This method should be used if you need to adjust for gene length,
    such as in a SMART-Seq2 protocol.

    Parameters
    ----------
    obj : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts
        (rows=genes, columns=cells).
    gene_lengths : :class:`pandas.Series` or `str`
        Should contain the gene lengths in base pairs and gene names
        set as index. The names must match the gene names used in
        `data`. Normally gene lengths should be the combined length of
        exons for every gene. If gene_lengths is a `str` then it is
        taken as a file path and loads it; first column is gene names
        and second column is the length, field separator is one space;
        an alternative format is a single column of combined exon
        lengths where the total number of rows matches the number of
        rows in the raw read counts matrix and with the same order.

    References
    ----------
    .. [1] Conesa et al. (2016) Genome Biology
           https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0881-8

    Returns
    -------
    :class:`pandas.DataFrame`
        A normalized data matrix with same dimensions as before.
    """
    mat = data
    if type(gene_lengths) == str:
        gene_lengths = pd.read_csv(
            gene_lengths, header=None, sep=' ', squeeze=True)
        if type(gene_lengths) != pd.core.series.Series:
            gene_lengths = pd.Series(
                gene_lengths[1].values, index=gene_lengths[0])
        else:
            gene_lengths.index = mat.index
            # remove NaN
            keep = np.logical_not(gene_lengths.isna())
            gene_lengths = gene_lengths[keep]
            mat = mat[keep]

    # take the intersection
    mat = mat[mat.index.isin(gene_lengths.index)]
    gene_lengths = gene_lengths[gene_lengths.index.isin(mat.index)]
    gene_lengths = gene_lengths.reindex(mat.index)
    # gene length in kilobases
    kb = gene_lengths/1000

    def _per_cell(x):
        s = sum(x)/10**6
        rpm = x/s
        rpkm = rpm/kb
        return rpkm

    ret = pd.DataFrame([_per_cell(i[1])
                        for i in mat.transpose().iterrows()]).transpose()
    ret.columns = mat.columns
    return ret


def fqn(data):
    """Performs full quantile normalization (FQN)

    Notes
    -----
    FQN has been shown to perform well on single cell data and was a
    popular normalization scheme for microarray data. The present
    function does not handle ties well.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts
        (rows=genes, columns=cells).

    References
    ----------
    .. [1] Bolstad et al. (2003) Bioinformatics
        https://academic.oup.com/bioinformatics/article/19/2/185/372664
    .. [2] Cole et al. (2019) Cell Systems
        https://www.biorxiv.org/content/10.1101/235382v2

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp, method='fqn')

    Returns
    -------
    :class:`pandas.DataFrame`
        A normalized data matrix with same dimensions as before.
    """
    ncells = data.shape[1]
    O = []  # to hold the ordered indices for each cell
    S = []  # to hold the sorted values for each cell
    m = data.to_numpy()
    for cc in np.arange(0, ncells):
        values = m[:, cc]
        ix = values.argsort()
        x = values[ix]
        O.append(ix)
        S.append(list(x))
    S = csr_matrix(np.array(S).transpose())
    # calc average distribution per gene
    avg = np.array(S.mean(axis=1).flatten())[0]
    L = []
    for cc in np.arange(0, ncells):
        loc = O[cc]
        L.append(list(pd.Series(avg, index=loc).sort_index().values))
    df = pd.DataFrame(L, index=data.columns)
    df.columns = data.index
    df = df.transpose()
    return df


def clean_matrix(data, obj, remove_low_qual=True, remove_mito=True,
                 meta=False):
    if remove_low_qual:
        # Remove low quality cells
        remove = obj.meta_cells.status[obj.meta_cells.status != 'OK']
        data = data.drop(remove.index, axis=1, errors='ignore')
        # Remove uninformative genes (e.g. lowly expressed)
        v = np.logical_and(obj.meta_genes.status != 'OK',
                           obj.meta_genes.ERCC != True)
        remove = obj.meta_genes.status[v]
        data = data.drop(remove.index, axis=0, errors='ignore')
    # Remove mitochondrial genes
    if remove_mito:
        remove = obj.meta_genes[obj.meta_genes.mitochondrial == True]
        data = data.drop(remove.index, axis=0, errors='ignore')
    if meta:
        md = obj.meta_cells.copy()
        md = md.loc[md.index.isin(data.columns), :]
        md = md.reindex(data.columns)
        return data, md
    return data


def norm(obj, method='standard', name=None, use_imputed=False,
         log=True, log_func=np.log2, small_const=1,
         remove_low_qual=True, remove_mito=True, gene_lengths=None,
         scaling_factor=10000, axis='genes', ngenes=2000,
         nworkers='auto', retx=False, verbose=False):
    """Normalizes gene expression data

    Notes
    -----
    A wrapper function around the individual normalization functions,
    which can also be called directly.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : `{'standard', 'rpkm', 'fqn', 'clr', 'vsn'}`
        Specifies the method to use. `standard` refers to the simplest
        normalization strategy involving scaling genes by total number
        of reads per cell. `rpkm` performs RPKM normalization and
        requires the `gene_lengths` parameter to be set.  `fqn`
        performs a full-quantile normalization. `clr` performs
        centered log ratio normalization. `vsn` performs a variance
        stabilizing normalization.  Default: standard
    name : `str`
        A choosen name for the normalization. It is used for storing
        and retrieving this normalization for plotting later. If
        `None` or an empty string, then it is set to the value of
        `method`.
    use_imputed : `bool`
        Use imputed data. If set to True, then
        :func:`adobo.preproc.impute` must have been run
        previously. Default: False
    log : `bool`
        Perform log transformation. Default: True
    log_func : `numpy.func`
        Logarithmic function to use. For example: np.log2, np.log1p,
        np.log10, etc.  Default: np.log2
    small_const : `float`
        A small constant to add to expression values to avoid log'ing
        genes with zero expression. Default: 1
    remove_low_qual : `bool`
        Remove low quality cells and uninformative genes identified by
        prior steps.  Default: True
    remove_mito : `bool`
        Remove mitochondrial genes (if these have been detected with
        `adobo.preproc.find_mitochondrial_genes`). Default: True
    gene_lengths : :class:`pandas.Series` or `str`
        A :class:`pandas.Series` containing the gene lengths in base
        pairs and gene names set as index. The names must match the
        gene names used in `data` (the order does not need to match
        and any symbols not found in the data will be discarded).
        Normally gene lengths should be the combined length of exons
        for every gene. If gene_lengths is a `str` then it is taken as
        a filename and loaded; first column is gene names and second
        column is the length, field separator is one space.
        `gene_lengths` needs to be set _only_ if
        method='rpkm'. Default: None
    scaling_factor : `int`
        Scaling factor used to multiply the scaled counts with. Only
        used for `method="depth"`. Default: 10000
    axis : `{'genes', 'cells'}`
        Only applicable when `method="clr"`, defines the axis to
        normalize across.  Default: 'genes'
    ngenes : `int`
        For method='vsn', number of genes to use when estimating
        parameters. Default: 2000
    nworkers : `int` or `{'auto'}`
        For method='vsn'. If a string, then the only accepted value is
        'auto', and the number of worker processes will be the total
        number of detected physical cores.  If an integer then it
        specifies the number of worker processes. Default: 'auto'
    retx : `bool`
        Return the normalized data as well. Default: False
    verbose : `bool`
        Be verbose or not. Default: False

    References
    ----------
    .. [1] Cole et al. (2019) Cell Systems
           https://www.biorxiv.org/content/10.1101/235382v2

    Example
    -------
    >>> import adobo as ad
    >>> exp = ad.IO.load_from_file('pbmc8k.mat.gz', bundled=True)
    >>> ad.normalize.norm(exp)

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if name is None or name == '':
        name = method
    if method == 'rpkm' and gene_lengths == None:
        raise Exception(
            'The `gene_lengths` parameter needs to be set when method is RPKM.')
    if use_imputed:
        if obj.imp_count_data.shape[0] == 0:
            raise Exception(
                'No imputed data found. Run adobo.preproc.impute() first.')
        else:
            data = obj.imp_count_data
    else:
        data = obj.count_data
    data = clean_matrix(data, obj)
    if method == 'standard':
        norm = standard(data, scaling_factor)
        norm_method = 'standard'
    elif method == 'rpkm':
        norm = rpkm(data, gene_lengths)
        norm_method = 'rpkm'
    elif method == 'fqn':
        norm = fqn(data)
        norm_method = 'fqn'
    elif method == 'clr':
        norm = clr(data, axis)
        norm_method = 'clr'
    elif method == 'vsn':
        norm = vsn(data, ngenes=ngenes, nworkers=nworkers, verbose=verbose)
        norm_method = 'vsn'
    else:
        raise Exception('Unknown normalization method.')
    if log:
        norm = log_func(norm+small_const)
    ne = None
    if np.any(obj.meta_genes.ERCC):
        # Save normalized ERCC
        ne = norm[obj.meta_genes.ERCC]
        # Remove ERCC so that they are not included in downstream analyses
        norm = norm[np.logical_not(obj.meta_genes.ERCC)]
    if obj.sparse:
        norm = norm.astype(pd.SparseDtype("float64", 0))
    obj.norm_data[name] = {'data': norm,
                           'method': method,
                           'log': log,
                           'norm_ercc': ne,
                           'dr': {},
                           'clusters': {},
                           'slingshot': {},
                           'de': {},
                           'combat': {}}
    obj.set_assay(sys._getframe().f_code.co_name, norm_method)
    if retx:
        return norm


def ComBat(obj, normalization=None, meta_cells_var=None,
           mean_only=True, par_prior=True, verbose=False):
    """Adjust for batch effects in datasets where the batch covariate
    is known

    Notes
    -----
    ComBat is a classical method for batch correction and it has been
    shown to perform well on single cell data. The drawback of using
    ComBat is that all cells in a batch is used for estimating model
    parameters. This implementation follows the ComBat function in the
    R package SVA.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    normalization : `str`
        The name of the normalization to operate on. If this is empty
        or None then the function will be applied on all
        normalizations available.
    meta_cells_var : `str`
        Meta data variable. Should be a column name in
        :py:attr:`data.dataset.meta_cells`.
    mean_only : `bool`
        Mean only version of ComBat. Default: True
    par_prior : `bool`
        True indicates parametric adjustments will be used, False
        indicates non-parametric adjustments will be used. Default:
        True
    verbose : `bool`
        Be verbose or not. Default: False

    References
    ----------
    .. [1] Johnson et al. (2007) Biostatistics. Adjusting batch
           effects in microarray expression data using empirical Bayes
           methods.
    .. [2] Buttner et al. (2019) Nat Met. A test metric for assessing
           single-cell RNA-seq batch correction

    Returns
    -------
    Modifies the passed object.
    """
    if not meta_cells_var:
        raise Exception('"meta_cells_var" cannot be empty.')
    if not mean_only:
        raise NotImplementedError('mean_only=False is not implemented.')
    if not par_prior:
        raise NotImplementedError('par_prior=False is not implemented.')
    if normalization == None or normalization == '':
        norm = list(obj.norm_data.keys())[-1]
    else:
        norm = normalization
    X = obj.norm_data[norm]['data']
    batch = obj.meta_cells.loc[:, meta_cells_var]
    batch = list(batch[batch.index.isin(X.columns)])
    # full design matrix
    dm = patsy.dmatrix('~ 0 + batch', pd.DataFrame({'batch': batch}))
    if verbose:
        print('Found %s batches' % dm.shape[1])
    nbatch = dm.shape[1]
    mm = np.dot(np.asarray(dm).T, X.to_numpy().T)
    i = np.dot(np.asarray(dm).T, np.asarray(dm))
    B_hat = np.linalg.solve(i, mm)
    nbatches = np.asarray(dm).sum(axis=0)
    grand_mean = np.dot(nbatches/np.sum(nbatches), B_hat)
    var_pooled = np.dot((X-np.dot(dm, B_hat).T)**2,
                        [1/np.sum(nbatches)]*int(np.sum(nbatches)))
    stand_mean = np.dot(np.matrix(grand_mean).T,
                        np.matrix([1]*int(np.sum(nbatches))))
    tmp = np.array(dm)
    tmp[:, :] = 0
    stand_mean = stand_mean + np.dot(tmp, B_hat).T
    sdata = (X - stand_mean)/np.dot(np.matrix(np.sqrt(var_pooled)).T,
                                    np.matrix([1]*int(np.sum(nbatches))))
    gamma_hat = np.linalg.solve(np.dot(np.array(dm).T, np.array(dm)),
                                np.dot(np.array(dm).T, sdata.T))
    if mean_only:
        delta_hat = np.ones(shape=(dm.shape[1], X.shape[0]))
    else:
        raise NotImplementedError('mean_only=False is not implemented.')
    gamma_bar = np.nanmean(gamma_hat, axis=1)
    t2 = np.nanvar(gamma_hat, axis=1)

    def aprior(x):
        m = np.mean(x)
        s2 = np.var(x)
        return (2 * s2 + m**2)/s2

    def bprior(x):
        m = np.mean(x)
        s2 = np.var(x)
        return (2 * s2 + m**3)/s2

    a_prior = np.apply_along_axis(aprior, 1, delta_hat)
    b_prior = np.apply_along_axis(bprior, 1, delta_hat)

    def postmean(g_hat, g_bar, n, d_star, t2):
        return (t2 * n * g_hat + d_star * g_bar)/(t2 * n + d_star)

    gamma_star = []
    delta_star = []

    for i in np.arange(0, nbatch):
        gamma_star.append(postmean(gamma_hat[i, :], gamma_bar[i], 1, 1, t2[i]))
        delta_star.append([1]*X.shape[0])

    gamma_star = np.asarray(gamma_star)
    delta_star = np.asarray(delta_star)

    bayesdata = sdata
    # iterate all cells per batch
    res = []
    for i in np.arange(0, np.asarray(dm).shape[1]):
        X_batch = bayesdata.iloc[:, np.asarray(dm)[:, i] == 1]
        X_batch_dm = np.asarray(dm)[np.asarray(dm)[:, i] == 1, :]
        lp = np.dot(np.matrix(np.sqrt(delta_star[i, ])).T,
                    np.matrix([1]*X_batch.shape[1]))
        r = (X_batch - np.dot(X_batch_dm, gamma_star).T)/lp
        res.append(r)
    res = pd.concat(res, axis=1)
    res = res.reindex(X.columns, axis=1)
    bd = res*np.dot(np.matrix(np.sqrt(var_pooled)).T,
                    np.matrix([1]*X.shape[1]))+stand_mean
    obj.set_assay(sys._getframe().f_code.co_name)
    if obj.sparse:
        bd = bd.astype(pd.SparseDtype("float64", 0))
    obj.norm_data[norm]['combat'] = bd
