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

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.covariance import MinCovDet

class data:
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
        s = "%s genes and %s cells" % (genes, cells)
        return s
    
    def __repr__(self):
        return self._describe()
