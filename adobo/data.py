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

import pandas as pd

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
    norm_log2 : `bool`
        True if log2 transform was performed on the normalized data, otherwise False.
    norm_method : `str`
        Containing the method used for normalization.
    norm : :class:`pandas.DataFrame`
        Normalized gene expression data.
    hvg : `list`
        Containing highly variable genes.
    """
    def __init__(self, raw_mat):
        # holding info about which assays have been done
        self.hvg = []
        self.hvg_method = ASSAY_NOT_DONE
        
        self._assays = {}
        self.exp_mat = raw_mat
        self._exp_ERCC = pd.DataFrame()
        self._exp_mito = pd.DataFrame()
        self._low_quality_cells = ASSAY_NOT_DONE

        self.norm = pd.DataFrame()
        self.norm_ERCC = pd.DataFrame()
        self.norm_method = ASSAY_NOT_DONE
        
        self.norm_log2=False
        
        # containing method and components for dimensionality reduction
        self.dr = {}
    
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
    
    def assays(self):
        """
        Displays a basic summary of the dataset and what analyses have been performed on
        it.
        """
        if self.get_assay('detect_mito'):
            print('Number of mitochondrial genes found: %s' % self.exp_mito.shape[0])
        if self.get_assay('detect_ERCC_spikes'):
            print('Number of ERCC spikes found: %s ' % self.exp_ERCC.shape[0])
        if self.get_assay('find_low_quality_cells'):
            print('Number of low quality cells found: %s ' % self.low_quality_cells.shape[0])
        if self.get_assay('norm'):
            print('Normalization method: %s ' % self.norm_method)
            print('Log2 transformed? %s' % (self.norm_log2))
        s = 'Has HVG discovery been performed? %s' % self.get_assay('find_hvg', lang=True)
        print(s)
    def __repr__(self):
        return self._describe()
