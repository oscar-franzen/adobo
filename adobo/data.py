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
    _assays : `dict`
        Holding information about what functions have been applied.
    exp_mat : :class:`pandas.DataFrame`
        Raw read count matrix.
    _exp_mito : :class:`pandas.DataFrame`
        Raw read count matrix containing mitochondrial genes.
    _exp_ERCC : :class:`pandas.DataFrame`
        Raw read count matrix containing ERCC spikes.
    _low_quality_cells : `list`
        Low quality cells identified with :py:meth:`adobo.preproc.find_low_quality_cells`.
    norm : :class:`pandas.DataFrame`
        Normalized gene expression data.
    hvg : `list`
        Containing highly variable genes.
    """
    def __init__(self, raw_mat):
        # holding info about which assays have been done
        self.hvg = []
        self._assays = {}
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

    def get_assay(self, name, lang=False):
        if lang:
            return ('No','Yes')[name in self._assays]
        else:
            return name in self._assays
    
    def set_assay(self, name, key=1):
        self._assays[name] = key
    
    @property
    def low_quality_cells(self):
        return self._low_quality_cells
    
    @low_quality_cells.setter
    def low_quality_cells(self, val):
        self._low_quality_cells = val
            
    @property
    def exp_mito(self):
        return self._exp_mito

    @exp_mito.setter
    def exp_mito(self, val):
        self._exp_mito = val
    
    @property
    def exp_ERCC(self):
        return self._exp_ERCC

    @exp_ERCC.setter
    def exp_ERCC(self, val):
        self._exp_ERCC = val
    
    def _describe(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        s = 'Raw read counts matrix contains: %s genes and %s cells\n' % (genes, cells)
        for key in self._assays:
            s += 'Done: %s (%s)\n' % (key, self._assays[key])
        return s
    
    def info(self):
        if self.get_assay('detect_mito'):
            print('Number of mitochondrial genes found: %s' % self.exp_mito.shape[0])
        if self.get_assay('detect_ERCC_spikes'):
            print('Number of ERCC spikes found: %s ' % self.exp_ERCC.shape[0])
        if self.get_assay('find_low_quality_cells'):
            print('Number of low quality cells found: %s ' % self.low_quality_cells.shape[0])
        if self.get_assay('norm'):
            print('Normalization method: %s ' % self.norm_method)
        s = 'Has HVG discovery been performed? %s' % self.get_assay('hvg', lang=True)
        print(s)
    def __repr__(self):
        return self._describe()
