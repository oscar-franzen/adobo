# Suppress warnings from sklearn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def simple_filters(obj, minreads=1000, minexpgenes=0.001, verbose=False):
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

def auto_clean(self, rRNA_genes, sd_thres=3, seed=42, verbose=False):
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
        This function computes Mahalanobis distances from the quality metrics. A
        robust estimate of covariance is used in the Mahalanobis function. Cells with
        Mahalanobis distances of three standard deviations from the mean are
        considered outliers.
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
    
    if verbose:
        print('%s low quality cell(s) identified' % len(self.low_quality_cells))
