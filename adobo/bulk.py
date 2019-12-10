# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://oscar-franzen.github.io/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for bulk RNA-seq integration
"""

from .IO import reader

def music(obj, bulk, normalization=None, clust_alg=None, verbose=False):
    """Generates a set of marker genes for every cluster by combining tests from
    pairwise analyses.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    bulk : `str` or `pandas.DataFrame`
        Either a string, which should be the path to the expression matrix or a pandas
        data frame containing the bulk gene express. Genes as rows and samples as columns.
    normalization : `str`
        The name of the normalization to operate on. If this is empty or None
        then the function will be applied on the last normalization that was applied.
    clust_alg : `str`
        Name of the clustering strategy. If empty or None, the last one will be used.
    verbose : `bool`
        Be verbose or not. Default: False
    
    Example
    -------
    # Add example

    References
    ----------
    .. [1] Wang et al., Nature Communications (2019) Bulk tissue cell type deconvolution
           with multi-subject single-cell expression reference
    .. [2] https://github.com/xuranw/MuSiC

    Returns
    -------
    Nothing. Modifies the passed object.
    """
    if isinstance(bulk, str):
        exp_bulk = reader('/home/rand/Temp/GSE50244_bulk.mat')
