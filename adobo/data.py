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
import joblib

import pandas as pd
import numpy as np

from ._constants import ASSAY_NOT_DONE

class dataset:
    """
    Storage container for raw data and analysis results.
    
    Attributes
    ----------
    _assays : `dict`
        Holding information about what functions have been applied.
    exp_mat : :class:`pandas.DataFrame`
        Raw read count matrix.
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
        A dict of :py:class:`pandas.DataFrame` containing components from dimensionality
        reduction.
    dr_gene_contr : `dict`
        A dict of :py:class:`pandas.DataFrame` containing variable contributions to each
        component. Useful for understanding the contribution of each gene to PCA
        components.
    meta_cells : `pandas.DataFrame`
        A data frame containing meta data for cells.
    meta_genes : `pandas.DataFrame`
        A data frame containing meta data for genes.
    clusters : `list`
        Generated clusters.
    desc : `str`
        A string describing the dataset.
    output_filename : `str`, optional
        A filename that will be used when calling save().
    """
    def __init__(self, raw_mat, desc='no desc set', output_filename=None,
                 input_filename=None):
        # holding info about which assays have been done
        self.hvg = []
        self.hvg_method = ASSAY_NOT_DONE
        self.desc = desc
        self.output_filename=output_filename
        self.input_filename=input_filename
        
        self._assays = {}
        self.exp_mat = raw_mat
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
        
        # containing components from dimensionality reduction techniques
        self.dr = {}
        # containing variable contribution to components
        self.dr_gene_contr = {}
        
        # containing clusters
        self.clusters = []
    
    def _print_raw_dimensions(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        return '%s genes and %s cells' % (genes, cells)
    
    def save(self, compress=True):
        """Serialize the object

        Notes
        -----
        Load the object data with joblib.load

        Parameters
        ----------
        compress : `bool`
            Save with data compression or not. Default: True

        Returns
        -------
        Nothing.
        """
        if not self.output_filename:
            raise Exception('No output filename set.')
        else:
            joblib.dump(self, filename=self.output_filename, compress=compress)

    def get_assay(self, name, lang=False):
        """ Get info if a function has been applied. """
        if lang:
            return ('No','Yes')[name in self._assays]
        else:
            return name in self._assays
    
    def set_assay(self, name, key=1):
        """ Set the assay that was applied. """
        self._assays[name] = key
    
    @property
    def low_quality_cells(self):
        return self._low_quality_cells
    
    @low_quality_cells.setter
    def low_quality_cells(self, val):
        self._low_quality_cells = val
    
    def _describe(self):
        """ Helper function for __repr__. """
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        s = """Filename (input): %s
Description: %s
Raw read counts matrix contains: %s genes and %s cells
""" % (self.input_filename, self.desc, genes, cells)

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
        if self.get_assay('find_mitochondrial_genes'):
            s = np.sum(self.meta_genes['mitochondrial'])
            print('Number of mitochondrial genes found: %s' % s)
        if self.get_assay('detect_ercc_spikes'):
            print('Number of ERCC spikes found: %s ' % np.sum(self.meta_genes['ERCC']))
        if self.get_assay('find_low_quality_cells'):
            print('Number of low quality cells found: %s ' % self.low_quality_cells.shape[0])
        if self.get_assay('norm'):
            print('Normalization method: %s ' % self.norm_method)
            print('Log2 transformed? %s' % (self.norm_log2))
        s = 'Has HVG discovery been performed? %s' % self.get_assay('find_hvg', lang=True)
        print(s)
        if self.get_assay('pca'):
            print('pca has been performed')
        cd = ', '.join([ item['algo'] for item in self.clusters ])
        print('The following community detection methods have been invoked: %s' % cd)

    def __repr__(self):
        return self._describe()
    
    def add_meta_data(self, axis, key, data):
        """Add meta data to the adobo object

        Notes
        -----
        Meta data can be added to cells or genes.

        Parameters
        ----------
        axis : `{'cells', 'genes'}`
            Should the data be added to cells or genes?
        key : `str`
            The variable name for your data. No whitespaces and special characters.
        data : `list`
            A list of data to add. The length must match the length of cells or genes.
            Data can be continuous or categorical.

        Returns
        -------
        Nothing.
        """
        if not axis in ('cells', 'genes'):
            raise Exception('Dimension must be cells or genes.')
        if axis == 'cells':
            self.meta_cells[key] = data
        elif axis == 'genes':
            self.meta_genes[key] = data
