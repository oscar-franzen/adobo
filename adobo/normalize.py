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

def _depth_norm(data, scaling_factor):
    col_sums = data.apply(lambda x: sum(x), axis=0)
    data_norm = (data / col_sums) * scaling_factor

def norm(obj, method='depth', log2=True, small_const=1, remove_low_qual_cells=True,
         exon_lengths=None, scaling_factor=10000):
    """Normalizes gene expression data

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A dataset class object.
    method : {'depth', 'rpkm'}
        Specifies the method to use. `depth` refers to simple normalization strategy
        involving dividing by total number of reads per cell. `rpkm` performs RPKM
        normalization and requires the `exon_lengths` parameter to be set.
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
    scaling_factor : int,
        Scaling factor used to multiply the scaled counts with. Only used for
        `method="depth"` (default: 10000).

    Returns
    -------
    None
    """
    data = obj.exp_mat
    if remove_low_qual_cells:
        data = data.drop(obj.low_quality_cells, axis=1)
    if method == 'depth':
        norm = _depth_norm(data, scaling_factor)
    if log:
        norm = np.log2(norm+small_const)
    obj.norm = norm
