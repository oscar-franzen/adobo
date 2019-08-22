"""
 adobo

 Description:
 An analysis framework for scRNA-seq data.

 How to use:
 https://github.com/oscar-franzen/adobo/

 Contact:
 Oscar Franzen <p.oscar.franzen@gmail.com>
"""

import sys
import os

import pandas as pd
import numpy as np

from sklearn.covariance import MinCovDet

class data:
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
    """
    def __init__(self, raw_mat):
        self.exp_mat = raw_mat
        self.exp_mito = None
        self.exp_ERCC = None
    
    def _print_raw_dimensions(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        return '%s genes and %s cells' % (genes, cells)
    
    def _describe(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        #s = "%s genes and %s cells\n\%s low quality cells" % (genes, cells, len(self.low_quality_cells))
        s = "Raw read counts matrix contains:\n%s genes and %s cells" % (genes, cells)
        return s
    
    def __repr__(self):
        return self._describe()
