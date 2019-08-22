# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
This module contains a data storage class.
"""

import sys
import os

class dataset:
    """
    Storage container for raw data and analysis results.
    
    Attributes
    ----------
    exp_mat : :class:`pandas.DataFrame`
        Raw read count matrix.
    exp_mito : :class:`pandas.DataFrame`
        Raw read count matrix containing mitochondrial genes.
    exp_ERCC : :class:`pandas.DataFrame`
        Raw read count matrix containing ERCC spikes.
    low_quality_cells : `list`
        Low quality cells identified with :py:meth:`adobo.preproc.find_low_quality_cells`.
    norm : :class:`pandas.DataFrame`
        Normalized gene expression data.
    norm_method : `str`
        Method used for normalization.
    """
    def __init__(self, raw_mat):
        self.exp_mat = raw_mat
        self.exp_mito = None
        self.exp_ERCC = None
        self.low_quality_cells = None
        self.norm = None
        self.norm_method = 'none'
    
    def _print_raw_dimensions(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        return '%s genes and %s cells' % (genes, cells)
    
    def _describe(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        #s = "%s genes and %s cells\n\%s low quality cells" % (genes, cells, len(self.low_quality_cells))
        s = """Raw read counts matrix contains:
%s genes and %s cells
Normalization method: %s""" % (genes, cells, self.norm_method)
        return s
    
    def __repr__(self):
        return self._describe()
