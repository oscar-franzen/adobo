# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
This module contains functions to normalize raw read counts.
"""

import sys

import numpy as np
import pandas as pd

from sklearn.neighbors import KernelDensity
import statsmodels.api as sm
from statsmodels.nonparametric.kernel_regression import KernelReg

from .log import warning

def vsn(data, min_cells=10, gmean_eps=1, n_genes=2000):
    """Performs variance stabilizing normaliztion based on a negative binomial regression
    model with regularized parameters
    
    Notes
    -----
    Adopts a subset of the functionality of the vst function in the R package sctransform [0].
    
    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes, columns=cells).
    min_cells : `int`
        Minimum number of cells expressing a gene for the gene to be used (default: 10).
    gmean_eps : `float`
        A small constant to avoid log(0)=-Inf (default: 1).
    n_genes : `int`
        Number of genes to use when estimating parameters (default: 2000).
    
    References
    ----------
        [0] https://cran.r-project.org/web/packages/sctransform/index.html
        [1] https://www.biorxiv.org/content/10.1101/576827v1
    
    Returns
    -------
    :class:`pandas.DataFrame`
        A normalized data matrix with same dimensions as before.
    """
    
    bw_adjust = 3 # Kernel bandwidth adjustment factor
    
    # numericals functions
    log10 = np.log10
    log = np.log
    exp = np.exp
    mean = np.mean
    sqrt = np.sqrt

    # data summary
    cell_attr = pd.DataFrame({'counts' : data.sum(axis=0),
                             'genes' : (data>0).sum(axis=0)})
    cell_attr['log_umi'] = log10(cell_attr.counts)
    cell_attr['log_gene'] = log10(cell_attr.genes)
    cell_attr['umi_per_gene'] = cell_attr.counts/cell_attr.genes
    cell_attr['log_umi_per_gene'] = log10(cell_attr.umi_per_gene)
    
    genes_cell_count = (data>0).sum(axis=1)
    genes = genes_cell_count[genes_cell_count>=10].index
    X = data[data.index.isin(genes)]
    genes_log_gmean = log10(row_geometric_mean(X, gmean_eps))
    
    cells_step1 = X.columns
    genes_step1 = X.index
    genes_log_gmean_step1 = genes_log_gmean
    data_step1 = cell_attr
    
    if n_genes < len(genes_step1):
        bw = bw_nrd(genes_log_gmean_step1)
        kde = KernelDensity(bandwidth=bw, kernel='gaussian')
        ret = kde.fit(genes_log_gmean_step1[:, None])
        # TODO: score_samples is slower than density() in R
        weights = 1/np.exp(kde.score_samples(genes_log_gmean_step1[:,None]))
        genes_step1 = np.random.choice(X.index, n_genes, replace=False, p=weights/sum(weights))
        X = X.loc[X.index.isin(genes_step1), X.columns.isin(cells_step1)]
        X = X.reindex(genes_step1)
        genes_log_gmean_step1 = log10(row_geometric_mean(X, gmean_eps))
    
    model_pars = []
    for g in genes_step1:
        y = X.loc[g,:]
        mod = sm.GLM(y, sm.add_constant(data_step1['log_umi']), family=sm.families.Poisson())
        res = mod.fit()
        s = res.summary()
        mu = res.fittedvalues
        theta = theta_ml(y, mu)
        coef = res.params
        model_pars.append({'gene' : g,
                           'theta' : theta,
                           'log_umi' : coef['log_umi'],
                           'const' : coef['const']})
    model_pars = pd.DataFrame(model_pars)
    model_pars.index = model_pars['gene']
    model_pars = model_pars.drop('gene', axis=1)
    model_pars.theta = log10(model_pars.theta)
    
    # remove outliers
    outliers = model_pars.apply(lambda x : is_outlier(x, genes_log_gmean_step1), axis=0)
    outliers = outliers.any(axis=1)
    model_pars = model_pars[np.logical_not(outliers.values)]
    
    genes_step1 = model_pars.index.values
    genes_log_gmean_step1 = genes_log_gmean_step1[np.logical_not(outliers.values)]
    
    x_points = np.maximum(genes_log_gmean, min(genes_log_gmean_step1))
    x_points = np.minimum(x_points, max(genes_log_gmean_step1))
    
    model_pars_fit = model_pars.apply(
        lambda x : KernelReg(x, genes_log_gmean_step1, bw='aic', var_type='c').fit(x_points)[0],
        axis = 0
    )
    
    model_pars_fit.index = x_points.index
    model_pars.theta = 10**model_pars.theta
    model_pars_fit.theta = 10**model_pars_fit.theta
    model_pars_final = model_pars_fit
    
    regressor_data_final = pd.DataFrame({'const' : [1]*len(cell_attr['log_umi']),
                                         'log_umi' : cell_attr['log_umi']})
    
    mu = exp(np.dot(model_pars_final.loc[:, ('const','log_umi')],
             regressor_data_final.transpose()))
    y = data[data.index.isin(model_pars_final.index)]
    
    # pearson residuals
    mu2 = pd.DataFrame(mu**2)
    t = (y-mu)
    n = sqrt((mu+mu2.div(model_pars_final.loc[:, 'theta'].values, axis=0)))
    
    n.index = t.index
    n.columns = t.columns
    
    pr = t/n
    return pr

def clr_normalization(data, axis='genes'):
    """Performs centered log ratio normalization similar to Seurat

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes, columns=cells).
    axis : {'genes', 'cells'}
        Normalize over genes or cells (default: genes).
        
    References
    ----------
    [0] Hafemeister et al. (2019)
        https://www.biorxiv.org/content/10.1101/576827v1

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
        
    r = data.apply(lambda x : np.log1p(x/np.exp(sum(np.log1p(x[x>0]))/len(x))), axis=axis)
    return r

def standard_normalization(data, scaling_factor):
    """Performs a standard normalization by scaling with the total read depth per cell and
    then multiplying with a scaling factor.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes, columns=cells).
    scaling_factor : `int`
        Scaling factor used to multiply the scaled counts with (default: 10000).

    References
    ----------
    [0] Evans et al. (2018) Briefings in Bioinformatics
        https://academic.oup.com/bib/article/19/5/776/3056951

    Returns
    -------
    :class:`pandas.DataFrame`
        A normalized data matrix with same dimensions as before.
    """
    col_sums = data.apply(lambda x: sum(x), axis=0)
    data_norm = (data / col_sums) * scaling_factor
    return data_norm

def rpkm(data, gene_lengths):
    """Normalize expression values as RPKM
    
    Notes
    -----
    This method should be used if you need to adjust for gene length, such as in
    a SMART-Seq2 protocol.

    Parameters
    ----------
    obj : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes, columns=cells).
    gene_lengths : `:class:`pandas.Series`` or `str`
        Should contain the gene lengths in base pairs and gene names set as index. The
        names must match the gene names used in `data`. Normally gene lengths should be
        the combined length of exons for every gene. If gene_lengths is a `str` then it is
        taken as a filename and loaded; first column is gene names and second column is
        the length, field separator is one space.

    References
    ----------
    [0] Conesa et al. (2016) Genome Biology
        https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-0881-8

    Returns
    -------
    :class:`pandas.DataFrame`
        A normalized data matrix with same dimensions as before.
    """
    
    mat = data
    
    if type(gene_lengths) == str:
        gene_lengths = pd.read_csv(gene_lengths, header=None, sep=' ')
        gene_lengths = pd.Series(gene_lengths[1].values, index=gene_lengths[0])
    
    # take the intersection
    mat = mat[mat.index.isin(gene_lengths.index)]
    gene_lengths = gene_lengths[gene_lengths.index.isin(mat.index)]
    gene_lengths = gene_lengths.reindex(mat.index)

    # gene length in kilobases
    kb = gene_lengths/1000

    def _foo(x):
        s = sum(x)/1000000
        rpm = x/s
        rpkm = rpm/kb
        return rpkm

    ret = mat.apply(_foo, axis=0)
    return ret

def full_quantile_normalization(data):
    """Performs full quantile normalization (FQN). FQN was shown to perform well on single
    cell data [1].
    
    Notes
    -----
    FQN was a popular normalization scheme for microarray data.
    The present function does not handle ties well.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes, columns=cells).

    References
    ----------
    [0] Bolstad et al. (2003) Bioinformatics
        https://academic.oup.com/bioinformatics/article/19/2/185/372664
    [1] Cole et al. (2019) Cell Systems
        https://www.biorxiv.org/content/10.1101/235382v2

    Returns
    -------
    :class:`pandas.DataFrame`
        A normalized data matrix with same dimensions as before.
    """
    ncells = data.shape[1]
    ngenes = data.shape[0]    
    # to hold the ordered indices for each cell
    O = []
    # to hold the sorted values for each cell
    S = []

    for cc in np.arange(0,ncells):
        values = data.iloc[:,cc]
        ix = values.argsort().values
        x = values[ix]
        O.append(ix)
        S.append(x.values)
    S = pd.DataFrame(S).transpose()
    
    # calc average distribution per gene
    avg = S.mean(axis=1)
    L = []
    for cc in np.arange(0,ncells):
        loc = O[cc]
        L.append(pd.Series(avg.values, index=loc).sort_index())
    df = pd.DataFrame(L, index=data.columns)
    df.columns = data.index
    df = df.transpose()
    return df

def norm(obj, method='standard', log2=True, small_const=1, remove_low_qual_cells=True,
         gene_lengths=None, scaling_factor=10000):
    r"""Normalizes gene expression data
    
    Notes
    -----
    A wrapper function around the individual normalization functions, which can also be
    called directly.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : `{'standard', 'rpkm', 'fqn', 'clr', 'vsn'}`
        Specifies the method to use. `standard` refers to the simplest normalization
        strategy involving scaling genes by total number of reads per cell. `rpkm`
        performs RPKM normalization and requires the `gene_lengths` parameter to be set.
        `fqn` performs a full-quantile normalization. `clr` performs centered log ratio
        normalization. `vsn` performs a variance stabilizing normalization.
    log2 : `bool`
        Perform log2 transformation. Not done if method='vsn' (default: True)
    small_const : `float`
        A small constant to add to expression values to avoid log'ing genes with zero
        expression (default: 1).
    remove_low_qual_cells : `bool`
        Remove low quality cells identified using :py:meth:`adobo.preproc.find_low_quality_cells`.
    gene_lengths : `:class:`pandas.Series`` or `str`
        A series containing the gene lengths in base pairs and gene names set as index. The
        names must match the gene names used in `data` (the order does not need to match
        and any symbols not found in the data will be discarded). Normally gene lengths
        should be the combined length of exons for every gene. If gene_lengths is a `str`
        then it is taken as a filename and loaded; first column is gene names and second
        column is the length, field separator is one space. gene_lengths is only needed
        if method='rpkm'.
    scaling_factor : `int`
        Scaling factor used to multiply the scaled counts with. Only used for
        `method="depth"` (default: 10000).
    axis : {'genes', 'cells'}
        Only applicable when `method="clr"`, defines the axis to normalize across,
        (default: 'genes').

    References
    ----------
    [0] Cole et al. (2019) Cell Systems
        https://www.biorxiv.org/content/10.1101/235382v2

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    data = obj.exp_mat
    if remove_low_qual_cells:
        if not obj.get_assay('find_low_quality_cells'):
            warning('remove_low_qual_cells set to True but find_low_quality_cells() has \
not been performed')
        else:
            data = data.drop(obj.low_quality_cells, axis=1)
    if method == 'standard':
        norm = standard_normalization(data, scaling_factor)
        obj.norm_method='standard'
    elif method == 'rpkm':
        norm = rpkm(data, gene_lengths)
        obj.norm_method='rpkm'
    elif method == 'fqn':
        norm = full_quantile_normalization(data)
        obj.norm_method='fqn'
    elif method == 'clr':
        norm = clr_normalization(data, axis)
        obj.norm_method='clr'
    elif method == 'vsn':
        norm = vsn(data)
        obj.norm_method='vsn'
    else:
        raise Exception('Unknown normalization method.')
    if log2 and method!='vsn':
        norm = np.log2(norm+small_const)
    obj.norm = norm
    obj.set_assay(sys._getframe().f_code.co_name)
