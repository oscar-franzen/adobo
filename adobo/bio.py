# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzén <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions related to biology.
"""

import re
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import scale as sklearn_scale

import adobo._log
import adobo.IO
import adobo.preproc
import adobo.dr

def cell_cycle_train(verbose=False):
    """Trains a cell cycle classifier using Stochastic Gradient Descent with data from
    Buettner et al.

    Parameters
    ----------
    verbose : `bool`
        Be verbose or not. Default: False
    
    References
    ----------
    Buettner et al. (2015) Computational analysis of cell-to-cell heterogeneity in
        single-cell RNA-sequencing data reveals hidden subpopulations of cells. Nat
        Biotech.

    Returns
    -------
    `sklearn.linear_model.SGDClassifier`
        A trained classifier.
    `list`
        Containing training features.
    """
    path_pkg = re.sub('/_log.py', '', adobo._log.__file__)
    path_data = path_pkg + '/data/Buettner_2015.mat'
    path_gene_lengths = path_pkg + '/data/Buettner_2015.mat.lengths'
    path_cc_genes = path_pkg + '/data/GO_0007049.txt' # cell cycle genes
    desc = 'Buettner et al. (2015) doi:10.1038/nbt.3102'
    B = adobo.IO.load_from_file(path_data, desc=desc)
    adobo.preproc.detect_ercc_spikes(B, ercc_pattern='NA_ERCC-[0-9]+')
    adobo.normalize.norm(B, method='rpkm', gene_lengths=path_gene_lengths)
    cc_genes = pd.read_csv(path_cc_genes, sep='\t', header=None)
    symb = pd.Series([ i[0] for i in B.norm.index.str.split('_') ])
    norm_cc_mat = B.norm[symb.isin(cc_genes[1]).values]
    X = norm_cc_mat.transpose() # cells as rows and genes as columns
    X = sklearn_scale(X,
                      axis=0,            # over genes, i.e. features (columns)
                      with_mean=True,    # subtracting the column means
                      with_std=True)     # scale the data to unit variance    
    Y = [ i[0] for i in norm_cc_mat.columns.str.split('_') ]
    
    clf = SGDClassifier(loss='hinge', penalty='l2', max_iter=5, shuffle=True,
                        verbose=verbose)
    clf.fit(X, Y)
    # np.sum(clf.predict(X) != Y)
    return clf, norm_cc_mat.index
    
def cell_cycle_predict(obj, clf, tr_features, retx=False):
    """Predicts cell cycle phase
    
    Notes
    ---------
    The classifier is trained on mouse data, so it should _only_ be used on mouse data
    unless it is trained on something else. Gene identifiers must use ensembl identifiers
    (prefixed with 'ENSMUSG'); pure gene symbols are not enough. Results are returned as
    a column in the data frame `meta_cells` of the passed object. Does not return
    probability scores.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    clf : `sklearn.linear_model.SGDClassifier`
        The classifier.
    tr_features : `list`
        Training features.
    retx : `bool`
        Returns list of results. Default: False
    
    Returns
    -------
    Modifies the passed object. If `retx=True` a list is returned with predictions.
    """
    X = obj.norm
    if X.index[0].rfind('ENSMUSG') < 0:
        raise Exception('Gene identifiers must use ENSG format.')
    X_g = X.index
    if re.search('ENSMUSG\d+\.\d+', X_g[0]):
        X_g = X_g.str.extract('^(.*)\.[0-9]+$', expand=False)
    if re.search('_ENSMUSG', X_g[0]):
        X_g = X_g.str.extract('^\S+?_(\S+)$', expand=False)
    symb = [ i[1] for i in tr_features.str.split('_') ]
    X_found = X[X_g.isin(symb)]
    X_g = X_found.index
    if re.search('ENSMUSG\d+\.\d+', X_g[0]):
        X_g = X_g.str.extract('^(.*)\.[0-9]+$', expand=False)
    if re.search('_ENSMUSG', X_g[0]):
        X_g = X_g.str.extract('^\S+?_(\S+)$', expand=False)
    if len(X_found) == 0:
        raise Exception('No genes found.')
    X_found.index = X_g

    symb = pd.Series(symb)
    missing = symb[np.logical_not(symb.isin(X_g))]
    X_empty = pd.DataFrame(np.zeros((len(missing), X_found.shape[1])))
    
    X_empty.index = missing
    X_empty.columns = X_found.columns
    X = pd.concat([X_found, X_empty])
    X = X.reindex(symb)
    # scale
    X = X.transpose() # cells as rows and genes as columns
    X = sklearn_scale(X,
                      axis=0,            # over genes, i.e. features (columns)
                      with_mean=True,    # subtracting the column means
                      with_std=True)     # scale the data to unit variance    
    pred = clf.predict(X)
    obj.meta_cells['cell_cycle'] = None
    obj.meta_cells['cell_cycle'][obj.meta_cells.index.isin(obj.norm.columns)] = pred
    if retx:
        return pred
