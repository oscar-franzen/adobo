# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for pre-processing scRNA-seq data.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet

# Suppress warnings from sklearn
def _warn(*args, **kwargs):
    pass
import warnings
warnings.warn = _warn

def simple_filter(obj, minreads=1000, minexpgenes=0.001, verbose=False):
    """Removes cells with too few reads and genes with very low expression

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    minreads : `int`, optional
        Minimum number of reads per cell required to keep the cell (default: 1000).
    minexpgenes : `str`, optional
        If this value is a float, then at least that fraction of cells must express the
        gene. If integer, then it denotes the minimum that number of cells must express
        the gene (default: 0.001).
    verbose : `bool`, optional
        Be verbose or not (default: False).

    Returns
    -------
    int
        Number of cells removed.
    int
        Number of genes removed.
    """
    exp_mat = obj.exp_mat
    cell_counts = exp_mat.sum(axis=0)
    r = cell_counts > minreads
    exp_mat = exp_mat[exp_mat.columns[r]]
    cr = np.sum(np.logical_not(r))
    gr = 0
    if verbose:
        print('%s cells removed' % cr)
    if minexpgenes > 0:
        if type(minexpgenes) == int:
            genes_expressed = exp_mat.apply(lambda x: sum(x > 0), axis=1)
            target_genes = genes_expressed[genes_expressed>minexpgenes].index
            gr = np.sum(genes_expressed <= minexpgenes)
            d = '{0:,g}'.format(gr)
            exp_mat = exp_mat[exp_mat.index.isin(target_genes)]
            if verbose:
                print('Removed %s genes.' % d)
        else:
            genes_expressed = exp_mat.apply(lambda x: sum(x > 0)/len(x), axis=1)
            gr = np.sum(genes_expressed <= minexpgenes)
            d = '{0:,g}'.format(gr)
            exp_mat = exp_mat[genes_expressed > minexpgenes]
            if verbose:
                print('Removed %s genes.' % d)
    obj.exp_mat = exp_mat
    obj.set_assay(sys._getframe().f_code.co_name)
    return cr, gr

def remove_empty(obj, verbose=False):
    """Removes empty cells and genes

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    verbose : boolean, optional
        Be verbose or not (default: False).

    Returns
    -------
    int
        Number of empty cells removed.
    int
        Number of empty genes removed.
    """
    exp_mat = obj.exp_mat
    data_zero = exp_mat == 0
    
    cells = data_zero.sum(axis=0)
    genes = data_zero.sum(axis=1)
    
    total_genes = exp_mat.shape[0]
    total_cells = exp_mat.shape[1]
    
    ecr=0
    egr=0

    if np.sum(cells == total_cells) > 0:
        r = np.logical_not(cells == total_cells)
        exp_mat = exp_mat[exp_mat.columns[r]]
        ecr = np.sum(cells == total_cells)
        if verbose:
            print('%s empty cells will be removed' % (ecr))
    if np.sum(genes == total_genes) > 0:
        r = np.logical_not(genes == total_genes)
        exp_mat = exp_mat.loc[exp_mat.index[r]]
        egr = np.sum(genes == total_genes)
        if verbose:
            print('%s empty genes will be removed' % (egr))
    obj.set_assay(sys._getframe().f_code.co_name)
    return ecr, egr

def detect_mito(obj, mito_pattern='^mt-', verbose=False):
    """Remove mitochondrial genes

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    mito_pattern : `str`, optional
        A regular expression matching mitochondrial gene symbols (default: "^mt-")
    verbose : boolean, optional
        Be verbose or not (default: False).

    Returns
    -------
    int
        Number of mitochondrial genes detected.
    """
    exp_mat = obj.exp_mat
    mt_count = exp_mat.index.str.contains(mito_pattern, regex=True, case=False)
    if np.sum(mt_count) > 0:
        exp_mito = exp_mat.loc[exp_mat.index[mt_count]]
        exp_mat = exp_mat.loc[exp_mat.index[np.logical_not(mt_count)]]
        obj.exp_mat = exp_mat
        obj.exp_mito = exp_mito
    nm = np.sum(mt_count)
    if verbose:
        print('%s mitochondrial genes detected and removed' % nm)
    obj.set_assay(sys._getframe().f_code.co_name)
    return nm
        
def detect_ercc_spikes(obj, ercc_pattern='^ercc[_-]\S+$', verbose=False):
    """Moves ercc (if present) to a separate container

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    ercc_pattern : `str`, optional
        A regular expression matching ercc gene symbols (default: "ercc[_-]\S+$").
    verbose : `bool`, optional
        Be verbose or not (default: False).

    Returns
    -------
    int
        Number of detected ercc spikes.
    """
    exp_mat = obj.exp_mat
    s = exp_mat.index.str.contains(ercc_pattern)
    exp_ercc = exp_mat[s]
    exp_mat = exp_mat[np.logical_not(s)]
    nd = np.sum(s)
    obj.exp_mat = exp_mat
    obj.exp_ercc = exp_ercc
    obj.ercc_pattern = ercc_pattern
    obj.set_assay(sys._getframe().f_code.co_name)
    if verbose:
        print('%s ercc spikes detected' % nd)
    return nd

def find_low_quality_cells(obj, rRNA_genes, sd_thres=3, seed=42, verbose=False):
    """Statistical detection of low quality cells using Mahalanobis distances
    
    Notes
    ----------------
    Mahalanobis distances are computed from five quality metrics. A robust estimate of
    covariance is used in the Mahalanobis function. Cells with Mahalanobis distances of
    three standard deviations from the mean are by default considered outliers.
    The five metrics are:

        1. log-transformed number of molecules detected
        2. the number of genes detected
        3. the percentage of reads mapping to ribosomal
        4. mitochondrial genes
        5. ercc recovery (if available)

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    rRNA_genes : `list` or `str`
        Either a list of rRNA genes or a string containing the path to a file containing
        the rRNA genes (one gene per line).
    sd_thres : `float`, optional
        Number of standard deviations to consider significant, i.e. cells are low quality
        if this. Set to higher to remove fewer cells (default: 3).
    seed : `float`, optional
        For the random number generator (default: 42).
    verbose : `bool`, optional
        Be verbose or not (default: False).

    Returns
    -------
    list
        A list of low quality cells that were identified, and also modifies the passed
        object.
    """
    
    if obj.exp_mito.shape[0] == 0:
        raise Exception('No mitochondrial genes found. Run detect_mito() first.')
    if obj.exp_ercc.shape[0] == 0:
        raise Exception('No ercc spikes found. Run detect_ercc() first.')
    if type(rRNA_genes) == str:
        rRNA_genes = pd.read_csv(rRNA_genes, header=None)
        rRNA_genes = rRNA_genes.iloc[:,0].values
    
    data = obj.exp_mat
    data_mt = obj.exp_mito
    data_ercc = obj.exp_ercc

    if not obj.get_assay('detect_ercc_spikes'):
        raise Exception('auto_clean() needs ercc spikes')
    if not obj.get_assay('detect_mito'):
        raise Exception('No mitochondrial genes found. Run detect_mito() first.')

    reads_per_cell = data.sum(axis=0) # no. reads/cell
    no_genes_det = np.sum(data > 0, axis=0)
    data_rRNA = data.loc[data.index.intersection(rRNA_genes)]
    
    perc_rRNA = data_rRNA.sum(axis=0)/reads_per_cell*100
    perc_mt = data_mt.sum(axis=0)/reads_per_cell*100
    perc_ercc = data_ercc.sum(axis=0)/reads_per_cell*100

    qc_mat = pd.DataFrame({'reads_per_cell' : np.log(reads_per_cell),
                           'no_genes_det' : no_genes_det,
                           'perc_rRNA' : perc_rRNA,
                           'perc_mt' : perc_mt,
                           'perc_ercc' : perc_ercc})
    robust_cov = MinCovDet(random_state=seed).fit(qc_mat)
    mahal_dists = robust_cov.mahalanobis(qc_mat)

    MD_mean = np.mean(mahal_dists)
    MD_sd = np.std(mahal_dists)

    thres_lower = MD_mean - MD_sd * sd_thres
    thres_upper = MD_mean + MD_sd * sd_thres

    res = (mahal_dists < thres_lower) | (mahal_dists > thres_upper)
    low_quality_cells = data.columns[res].values
    
    if verbose:
        print('%s low quality cell(s) identified' % len(low_quality_cells))
    obj.low_quality_cells = low_quality_cells
    obj.set_assay(sys._getframe().f_code.co_name)
    return low_quality_cells
