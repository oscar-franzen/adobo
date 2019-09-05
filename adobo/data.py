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
import numpy as np

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
    _exp_ercc : :class:`pandas.DataFrame`
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
    dr : `dict`
        A dict of :py:class:`pandas.DataFrame` containing results of dimensionality
        reduction.
    meta_data : `pandas.DataFrame`
        A data frame containing single cell meta data on a per-cell level.
    desc : `str`
        A string describing the dataset.
    """
    def __init__(self, raw_mat, desc='no desc set'):
        # holding info about which assays have been done
        self.hvg = []
        self.hvg_method = ASSAY_NOT_DONE
        self.desc = desc
        
        self._assays = {}
        self.exp_mat = raw_mat
        self._exp_ercc = pd.DataFrame()
        self._exp_mito = pd.DataFrame()
        self._low_quality_cells = ASSAY_NOT_DONE

        self.norm = pd.DataFrame()
        self.norm_ercc = pd.DataFrame()
        self.norm_method = ASSAY_NOT_DONE
        
        self.norm_log2 = False
        
        # meta data for cells
        self.meta_cells = pd.DataFrame(index=raw_mat.columns)
        self.meta_cells['total_reads'] = raw_mat.sum(axis=0)        
        self.meta_cells['status'] = ['OK']*raw_mat.shape[1]
        self.meta_cells['detected_genes'] = np.sum(raw_mat > 0, axis=0)
        
        # meta data for genes
        self.meta_genes = pd.DataFrame(index=raw_mat.index)
        self.meta_genes['expressed'] = raw_mat.apply(lambda x: sum(x > 0), axis=1)
        self.meta_genes['expressed_perc'] = raw_mat.apply(lambda x: sum(x > 0)/len(x), axis=1)
        self.meta_genes['status'] = ['OK']*raw_mat.shape[0]
        self.meta_genes['mitochondrial'] = [None]*raw_mat.shape[0]
        self.meta_genes['ERCC'] = [None]*raw_mat.shape[0]
        
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
    def exp_ercc(self):
        return self._exp_ercc

    @exp_ercc.setter
    def exp_ercc(self, val):
        self._exp_ercc = val
    
    def _describe(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        s = """Description: %s
Raw read counts matrix contains: %s genes and %s cells
""" % (self.desc, genes, cells)

        for key in self._assays:
            s += 'Done: %s (%s)\n' % (key, self._assays[key])
        
        if self.norm.shape[0] > 0:
            genes = '{:,}'.format(self.norm.shape[0])
            cells = '{:,}'.format(self.norm.shape[1])
            q = (genes, cells)
            s += "Normalized gene expression matrix contains: %s genes and %s cells" % q
        return s
    
    def assays(self):
        """
        Displays a basic summary of the dataset and what analyses have been performed on
        it.
        """
        if self.get_assay('detect_mito'):
            print('Number of mitochondrial genes found: %s' % self.exp_mito.shape[0])
        if self.get_assay('detect_ercc_spikes'):
            print('Number of ERCC spikes found: %s ' % self.exp_ercc.shape[0])
        if self.get_assay('find_low_quality_cells'):
            print('Number of low quality cells found: %s ' % self.low_quality_cells.shape[0])
        if self.get_assay('norm'):
            print('Normalization method: %s ' % self.norm_method)
            print('Log2 transformed? %s' % (self.norm_log2))
        s = 'Has HVG discovery been performed? %s' % self.get_assay('find_hvg', lang=True)
        print(s)
    def __repr__(self):
        return self._describe()
