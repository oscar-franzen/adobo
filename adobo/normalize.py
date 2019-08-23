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

import numpy as np
import pandas as pd

def clr_normalization(data, axis='genes'):
    """Performs centered log ratio normalization similar to Seurat

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes,
        columns=cells).
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
        A pandas data frame object containing raw read counts (rows=genes,
        columns=cells).
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

def full_quantile_normalization(data):
    """Performs full quantile normalization (FQN).
    
    Notes
    -----
    FQN was a popular normalization scheme for microarray data.
    The present function does not handle ties well.

    Parameters
    ----------
    data : :class:`pandas.DataFrame`
        A pandas data frame object containing raw read counts (rows=genes,
        columns=cells).

    References
    ----------
    [0] Bolstad et al. (2003) Bioinformatics
        https://academic.oup.com/bioinformatics/article/19/2/185/372664

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

def norm(obj, method='depth', log2=True, small_const=1, remove_low_qual_cells=True,
         exon_lengths=None, scaling_factor=10000):
    r"""Normalizes gene expression data

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : {'standard', 'rpkm'}
        Specifies the method to use. `standard` refers to the simplest normalization
        strategy involving scaling genes by total number of reads per cell. `rpkm`
        performs RPKM normalization and requires the `exon_lengths` parameter to be set.
        `fqn` performs a full-quantile normalization. `clr` performs centered log ratio
        normalization.
    log2 : `bool`
        Perform log2 transformation (default: True)
    small_const : `float`
        A small constant to add to expression values to avoid log'ing genes with zero
        expression (default: 1).
    remove_low_qual_cells : `bool`
        Remove low quality cells identified using :py:meth:`adobo.preproc.find_low_quality_cells`.
    exon_lengths : :class:`pandas.DataFrame`
        A pandas data frame containing two columns; first column should be gene names
        matching the data matrix and the second column should contain exon lengths.
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
    None
    """
    data = obj.exp_mat
    if remove_low_qual_cells:
        data = data.drop(obj.low_quality_cells, axis=1)
    if method == 'standard':
        norm = standard_normalization(data, scaling_factor)
        obj.norm_method='standard'
    elif method == 'rpkm':
        obj.norm_method='rpkm'
    elif method == 'fqn':
        norm = full_quantile_normalization(data)
        obj.norm_method='fqn'
    elif method == 'clr':
        norm = clr_normalization(data, axis)
        obj.norm_method='clr'
    if log2:
        norm = np.log2(norm+small_const)
    obj.norm = norm
