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

def reset_filters(obj):
    """Resets cell and gene filters

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    obj.meta_cells.status[obj.meta_cells.status!='OK'] = 'OK'
    obj.meta_genes.status[obj.meta_genes.status!='OK'] = 'OK'

def simple_filter(obj, minreads=1000, maxreads=None, minexpgenes=0.001, verbose=False):
    """Removes cells with too few reads and genes with very low expression

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    minreads : `int`, optional
        Minimum number of reads per cell required to keep the cell. Default: 1000
    maxreads : `int`, optional
        Set a maximum number of reads allowed. Useful for filtering out suspected doublets.
    minexpgenes : `str`, optional
        If this value is a float, then at least that fraction of cells must express the
        gene. If integer, then it denotes the minimum that number of cells must express
        the gene. Default: 0.001
    verbose : `bool`, optional
        Be verbose or not. Default: False

    Returns
    -------
    int
        Number of cells removed.
    int
        Number of genes removed.
    """
    count_data = obj.count_data
    cell_counts = obj.meta_cells.total_reads
    cells_remove = cell_counts < minreads
    obj.meta_cells.status[cells_remove] = 'EXCLUDE'
    if maxreads:
        cells_remove = cell_counts > maxreads
        obj.meta_cells.status[cells_remove] = 'EXCLUDE'
    genes_removed = 0
    if minexpgenes > 0:
        if type(minexpgenes) == int:
            genes_exp = obj.meta_genes.expressed
            genes_remove = genes_exp<minexpgenes
            obj.meta_genes.status[genes_remove] = 'EXCLUDE'
        else:
            genes_exp = obj.meta_genes.expressed_perc
            genes_remove = genes_exp<minexpgenes
            obj.meta_genes.status[genes_remove] = 'EXCLUDE'
    obj.set_assay(sys._getframe().f_code.co_name)
    r = np.sum(obj.meta_cells.status=='EXCLUDE')
    if verbose:
        s = '%s cells and %s genes were removed'
        print(s % (r, np.sum(genes_remove)))
    return r, np.sum(genes_remove)

def find_mitochondrial_genes(obj, mito_pattern='^mt-', genes=None, verbose=False):
    """Find mitochondrial genes and adds percent mitochondrial expression of total
    expression to the cellular meta data

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    mito_pattern : `str`
        A regular expression matching mitochondrial gene symbols. Default: "^mt-"
    genes : `list`, optional
        Instead of using `mito_pattern`, specify a `list` of genes that are mitochondrial.
    verbose : boolean
        Be verbose or not. Default: False

    Returns
    -------
    int
        Number of mitochondrial genes detected.
    """
    count_data = obj.count_data
    if genes is None:
        mito = count_data.index.str.contains(mito_pattern, regex=True, case=False)
        obj.meta_genes['mitochondrial'] = mito
    else:
        mito = obj.count_data.index.isin(genes)
        obj.meta_genes['mitochondrial'] = mito
    no_found = np.sum(obj.meta_genes['mitochondrial'])
    if no_found > 0:
        mt = obj.meta_genes[obj.meta_genes.mitochondrial].index
        mt_counts = obj.count_data.loc[mt, :]
        mt_counts = mt_counts.sum(axis=0)
        mito_perc = mt_counts / obj.meta_cells.total_reads*100
        obj.add_meta_data(axis='cells', key='mito_perc', data=mito_perc, type_='cont')
    if verbose:
        print('%s mitochondrial genes detected' % no_found)
    obj.set_assay(sys._getframe().f_code.co_name)
    return no_found
        
def find_ercc(obj, ercc_pattern='^ERCC[_-]\S+$', verbose=False):
    """Flag ERCC spikes

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    ercc_pattern : `str`, optional
        A regular expression matching ercc gene symbols. Default: "ercc[_-]\S+$"
    verbose : `bool`, optional
        Be verbose or not. Default: False

    Returns
    -------
    int
        Number of detected ercc spikes.
    """
    count_data = obj.count_data
    ercc = count_data.index.str.contains(ercc_pattern)
    obj.meta_genes['ERCC'] = ercc
    obj.meta_genes['status'][ercc] = 'EXCLUDE'
    no_found = np.sum(ercc)
    obj.ercc_pattern = ercc_pattern
    if no_found>0:
        ercc = obj.meta_genes[obj.meta_genes.ERCC].index
        ercc_counts = obj.count_data.loc[ercc, :]
        ercc_counts = ercc_counts.sum(axis=0)
        ercc_perc = ercc_counts / obj.meta_cells.total_reads*100
        obj.add_meta_data(axis='cells', key='ercc_perc', data=ercc_perc, type_='cont')
    if verbose:
        print('%s ercc spikes detected' % no_found)
    obj.set_assay(sys._getframe().f_code.co_name)
    return no_found

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
    sd_thres : `float`
        Number of standard deviations to consider significant, i.e. cells are low quality
        if this. Set to higher to remove fewer cells. Default: 3
    seed : `float`
        For the random number generator. Default: 42
    verbose : `bool`
        Be verbose or not. Default: False

    Returns
    -------
    list
        A list of low quality cells that were identified, and also modifies the passed
        object.
    """
    
    if np.sum(obj.meta_genes.mitochondrial) == 0:
        raise Exception('No mitochondrial genes found. Run detect_mito() first.')
    if np.sum(obj.meta_genes.ERCC) == 0:
        raise Exception('No ERCC spike-ins found. Run detect_ercc_spikes() first.')
    if type(rRNA_genes) == str:
        rRNA_genes = pd.read_csv(rRNA_genes, header=None)
        obj.meta_genes['rRNA'] = obj.meta_genes.index.isin(rRNA_genes.iloc[:, 0])
    
    if not 'mito' in obj.meta_cells.columns:
        mt_genes = obj.meta_genes.mitochondrial[obj.meta_genes.mitochondrial]
        mito_mat = obj.count_data[obj.count_data.index.isin(mt_genes.index)]
        mito_sum = mito_mat.sum(axis=0)
        obj.meta_cells['mito'] = mito_sum
    if not 'ERCC' in obj.meta_cells.columns:
        ercc = obj.meta_genes.ERCC[obj.meta_genes.ERCC]
        ercc_mat = obj.count_data[obj.count_data.index.isin(ercc.index)]
        ercc_sum = ercc_mat.sum(axis=0)
        obj.meta_cells['ERCC'] = ercc_sum
    if not 'rRNA' in obj.meta_cells.columns:
        rrna_genes = obj.meta_genes.rRNA[obj.meta_genes.rRNA]
        rrna_mat = obj.count_data[obj.count_data.index.isin(rrna_genes.index)]
        rrna_sum = rrna_mat.sum(axis=0)
        obj.meta_cells['rRNA'] = rrna_sum
    
    #data = obj.count_data
    inp_total_reads = obj.meta_cells.total_reads
    inp_detected_genes = obj.meta_cells.detected_genes/inp_total_reads
    inp_rrna = obj.meta_cells.rRNA/inp_total_reads
    inp_mt = obj.meta_cells.mito/inp_total_reads
    inp_ercc = obj.meta_cells.ERCC/inp_total_reads

    qc_mat = pd.DataFrame({'reads_per_cell' : np.log(inp_total_reads),
                           'no_genes_det' : inp_detected_genes,
                           'perc_rRNA' : inp_rrna,
                           'perc_mt' : inp_mt,
                           'perc_ercc' : inp_ercc})
    robust_cov = MinCovDet(random_state=seed).fit(qc_mat)
    mahal_dists = robust_cov.mahalanobis(qc_mat)

    MD_mean = np.mean(mahal_dists)
    MD_sd = np.std(mahal_dists)

    thres_lower = MD_mean - MD_sd * sd_thres
    thres_upper = MD_mean + MD_sd * sd_thres

    res = (mahal_dists < thres_lower) | (mahal_dists > thres_upper)
    low_quality_cells = obj.count_data.columns[res].values
    obj.low_quality_cells = low_quality_cells
    obj.set_assay(sys._getframe().f_code.co_name)
    r = obj.meta_cells.index.isin(low_quality_cells)
    obj.meta_cells.status[r]='EXCLUDE'
    if verbose:
        print('%s low quality cell(s) identified' % len(low_quality_cells))
    return low_quality_cells
