import sys
import os

import pandas as pd
import numpy as np

class data:
    def __init__(self):
        pass
        
    def _print_raw_dimensions(self):
        genes = '{:,}'.format(self.exp_mat.shape[0])
        cells = '{:,}'.format(self.exp_mat.shape[1])
        return '%s genes and %s cells' % (genes, cells)
        
    def __repr__(self):
        return self._print_raw_dimensions()

    def remove_empty(self, verbose=False):
        """ Removes empty cells and genes """
        data_zero = self.exp_mat == 0

        cells = data_zero.sum(axis=0)
        genes = data_zero.sum(axis=1)

        total_genes = self.exp_mat.shape[0]
        total_cells = self.exp_mat.shape[1]

        if np.sum(cells == total_cells) > 0:
            r = np.logical_not(cells == total_cells)
            self.exp_mat = self.exp_mat[self.exp_mat.columns[r]]
            if verbose:
                print('%s empty cells will be removed' % (np.sum(cells == total_cells)))
        if np.sum(genes == total_genes) > 0:
            r = np.logical_not(genes == total_genes)
            self.exp_mat = self.exp_mat.loc[self.exp_mat.index[r]]
            if verbose:
                log_info('%s empty genes will be removed' % (np.sum(genes == total_genes)))
    
    def remove_mito(self, mito_pattern='^mt-', verbose=False):
        """ Remove mitochondrial genes. """
        mt_count = self.exp_mat.index.str.contains(mito_pattern, regex=True, case=False)
        if np.sum(mt_count) > 0:
            self.exp_mat = self.exp_mat.loc[self.exp_mat.index[np.logical_not(mt_count)]]
        if verbose:
            print('%s mitochondrial genes detected and removed' % np.sum(mt_count))
            
    def detect_ERCC_spikes(self, ERCC_pattern='^ERCC[_-]\S+$', verbose=False):
        """ Moves ERCC (if present) to a separate container. """
        s = self.exp_mat.index.str.contains(ERCC_pattern)
        self.exp_ERCC = self.exp_mat[s]
        self.exp_mat = self.exp_mat[np.logical_not(s)]
        if verbose:
            print('%s ERCC spikes detected' % np.sum(s))

    def load_from_file(self, filename, sep='\t', header=0, column_id=True, verbose=False):
        """
        Load a gene expression matrix consisting of raw read counts
    
        Args:
            filename    Path to the file
            sep         Character used to separate fields
            header      If the data file has a header (set to 0 if it has
            
        Remarks:
            Gene expression matrix should not be normalized.
        """
        if not os.path.exists(filename):
            raise Exception('%s not found' % filename)
        self.exp_mat = pd.read_csv(filename,
                                   delimiter=sep,
                                   header=header)
        if column_id:
            self.exp_mat.index = self.exp_mat[self.exp_mat.columns[0]]
            self.exp_mat = self.exp_mat.drop(self.exp_mat.columns[0], axis=1)

        # remove duplicate genes
        dups = self.exp_mat.index.duplicated(False)
        if np.any(dups):
            self.exp_mat = self.exp_mat.iloc[np.logical_not(dups)]
            if verbose:
                print('%s duplicated genes detected and removed.' % np.sum(dups))
        if np.any(self.exp_mat.dtypes != 'int64'):
            raise Exception('Non-count values detected in data matrix.')
        if verbose:
            print(self._print_raw_dimensions() + ' were loaded')
