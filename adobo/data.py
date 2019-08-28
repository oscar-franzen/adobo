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

from .constants import ASSAY_NOT_DONE

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

        self._exp_ERCC = ASSAY_NOT_DONE
        self._exp_mito = ASSAY_NOT_DONE
        self._low_quality_cells = ASSAY_NOT_DONE

        self.norm = None
        self.norm_method = ASSAY_NOT_DONE
    
    def _print_raw_dimensions(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        return '%s genes and %s cells' % (genes, cells)
    
    @property
    def low_quality_cells(self):
        if type(self._low_quality_cells) == str:
            return self._low_quality_cells
        else:
            return len(self._low_quality_cells)
    
    @low_quality_cells.setter
    def low_quality_cells(self, val):
        self._low_quality_cells = val
            
    @property
    def exp_mito(self):
        if type(self._exp_mito) == str:
            return self._exp_mito
        else:
            return self._exp_mito.shape[0]

    @exp_mito.setter
    def exp_mito(self, val):
        self._exp_mito = val
    
    @property
    def exp_ERCC(self):
        if type(self._exp_ERCC) == str:
            return self._exp_ERCC
        else:
            return self._exp_ERCC.shape[0]

    @exp_ERCC.setter
    def exp_ERCC(self, val):
        self._exp_ERCC = val
    
    def _describe(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        #s = "%s genes and %s cells\n\%s low quality cells" % (genes, cells, len(self.low_quality_cells))
        s = """Raw read counts matrix contains: %s genes and %s cells
Number of low quality cells: %s
Number of mitochondrial genes: %s
Number of ERCC spikes: %s
Normalization method: %s""" % (genes,
                               cells,
                               self.low_quality_cells,
                               self.exp_mito,
                               self.exp_ERCC,
                               self.norm_method)
        return s
    
    def __repr__(self):
        return self._describe()
