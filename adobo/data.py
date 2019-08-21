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

# Suppress warnings from sklearn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class data:
    def __init__(self):
        self.exp_mito = None
        self.exp_ERCC = None
        
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
    
    def detect_mito(self, mito_pattern='^mt-', verbose=False):
        """ Remove mitochondrial genes. """
        mt_count = self.exp_mat.index.str.contains(mito_pattern, regex=True, case=False)
        if np.sum(mt_count) > 0:
            self.exp_mito = self.exp_mat.loc[self.exp_mat.index[mt_count]]
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
        
    def auto_clean(self, rRNA_genes, sd_thres=3, seed=42):
        """
        Finds low quality cells using five metrics:

            1. log-transformed number of molecules detected
            2. the number of genes detected
            3. the percentage of reads mapping to ribosomal
            4. mitochondrial genes
            5. ERCC recovery (if available)
            
        Arguments:
            rRNA_genes      List of rRNA genes.
            sd_thres        Number of standard deviations to consider significant, i.e.
                            cells are low quality if this. Set to higher to remove
                            fewer cells. Default is 3.
            seed            For the random number generator.

        Remarks:
        This function computes Mahalanobis distances from the quality metrics. A robust
        estimate of covariance is used in the Mahalanobis function. Cells with
        Mahalanobis distances of three standard deviations from the mean are considered
        outliers.
        """
        
        data = self.exp_mat
        data_mt = self.exp_mito
        data_ERCC = self.exp_ERCC
        
        if type(data_ERCC) == None:
            raise Exception('auto_clean() needs ERCC spikes')
        if type(data_mt) == None:
            raise Exception('No mitochondrial genes found. Run detect_mito() first.')

        reads_per_cell = data.sum(axis=0) # no. reads/cell
        no_genes_det = np.sum(data > 0, axis=0)
        data_rRNA = data.loc[data.index.intersection(rRNA_genes)]
        
        perc_rRNA = data_rRNA.sum(axis=0)/reads_per_cell*100
        perc_mt = data_mt.sum(axis=0)/reads_per_cell*100
        perc_ERCC = data_ERCC.sum(axis=0)/reads_per_cell*100

        qc_mat = pd.DataFrame({'reads_per_cell' : np.log(reads_per_cell),
                               'no_genes_det' : no_genes_det,
                               'perc_rRNA' : perc_rRNA,
                               'perc_mt' : perc_mt,
                               'perc_ERCC' : perc_ERCC})
        robust_cov = MinCovDet(random_state=seed).fit(qc_mat)
        mahal_dists = robust_cov.mahalanobis(qc_mat)

        MD_mean = np.mean(mahal_dists)
        MD_sd = np.std(mahal_dists)

        thres_lower = MD_mean - MD_sd * sd_thres
        thres_upper = MD_mean + MD_sd * sd_thres

        res = (mahal_dists < thres_lower) | (mahal_dists > thres_upper)

        self.low_quality_cells = data.columns[res].values
        print(self.low_quality_cells)

    def barplot_reads_per_cell(self, barcolor='#E69F00', filename=None,
                               title='sequencing reads'):
        """ Generates a bar plot of read counts per cell. """
        cell_counts = self.exp_mat.sum(axis=0)
        plt.clf()
        colors = [barcolor]*(len(cell_counts))

        plt.bar(np.arange(len(cell_counts)), sorted(cell_counts, reverse=True),
                color=colors)
        plt.ylabel('raw read counts')
        plt.xlabel('cells (sorted on highest to lowest)')
        plt.title(title)
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
    def barplot_genes_per_cell(self, barcolor='#E69F00', filename=None,
                               title='expressed genes'):
        """ Generates a bar plot of number of expressed genes per cell. """
        genes_expressed = self.exp_mat.apply(lambda x: sum(x > 0), axis=0)

        plt.clf()
        plt.bar(np.arange(len(genes_expressed)), sorted(genes_expressed, reverse=True),
                color=[barcolor]*len(genes_expressed))
        plt.ylabel('number of genes')
        plt.xlabel('cells (sorted on highest to lowest)')
        plt.title(title)
        if filename:
            plt.savefig(filename, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def simple_filters(self, minreads=1000, minexpgenes=0.001, verbose=False):
        """
        Removes cells with too few reads and genes with very low expression.

        Arguments:
            minreads        Minimum number of reads per cell required to keep the cell
            minexpgenes     If this value is a float, then at least that fraction of
                            cells must express the gene. If integer, then it denotes the
                            minimum that number of cells must express the gene.
        """
        cell_counts = self.exp_mat.sum(axis=0)
        r = cell_counts > minreads
        self.exp_mat = self.exp_mat[self.exp_mat.columns[r]]
        if verbose:
            print('%s cells removed' % np.sum(np.logical_not(r)))
        if minexpgenes > 0:
            if type(minexpgenes) == int:
                genes_expressed = self.exp_mat.apply(lambda x: sum(x > 0), axis=1)
                target_genes = genes_expressed[genes_expressed>minexpgenes].index
                d = '{0:,g}'.format(np.sum(genes_expressed <= minexpgenes))
                self.exp_mat = self.exp_mat[self.exp_mat.index.isin(target_genes)]
                if verbose:
                    print('Removed %s genes.' % d)
            else:
                genes_expressed = self.exp_mat.apply(lambda x: sum(x > 0)/len(x), axis=1)
                d = '{0:,g}'.format(np.sum(genes_expressed <= minexpgenes))
                self.exp_mat = self.exp_mat[genes_expressed > minexpgenes]
                if verbose:
                    print('Removed %s genes.' % d)

    def load_from_file(self, filename, sep='\t', header=0, column_id=True, verbose=False):
        """
        Load a gene expression matrix consisting of raw read counts
    
        Arguments:
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
