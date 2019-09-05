# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
This module contains functions to cluster data.
"""

from sklearn.neighbors import NearestNeighbors

def knn(obj, k=10, target='irlb'):
    """
    Nearest Neighbour Search. Finds the k number of near neighbours for each cell.
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    k : `int`
        Number of nearest neighbors. Default: 10
    target : `{'irlb', 'pca'}`
        The dimensionality reduction result to run the NN search on. Default: irlb
        
    Returns
    -------
    Nothing. Modifies the passed object.
    """
    X = obj.dr[target]
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    nbrs.fit(X)
    indices = nbrs.kneighbors(X)[1]
    obj.nn_idx = indices+1

def generate(obj, k=10):
    """
    A wrapper function for generating single cell clusters
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A dataset class object.
    k : `int`
        Number of nearest neighbors. Default: 10
    
    References
    ----------
    None.
    
    Returns
    -------
    Nothing. Modifies the passed object.
    """
    
    knn(obj, k)
