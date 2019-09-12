# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzén <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for plotting scRNA-seq data.
"""
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from ._constants import CLUSTER_COLORS_DEFAULT, YLW_CURRY
from ._colors import unique_colors

def reads_per_cell(obj, barcolor='#E69F00', title='sequencing reads', filename=None):
    """Generates a bar plot of read counts per cell

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    barcolor : `str`
        Color of the bars. Default: "#E69F00"
    title : `str`
        Title of the plot. Default: "sequencing reads"
    filename : `str`, optional
        Write plot to file instead of showing it on the screen.

    Returns
    -------
    None
    """
    exp_mat = obj.exp_mat
    cell_counts = exp_mat.sum(axis=0)
    plt.clf()
    colors = [barcolor]*(len(cell_counts))
    plt.gca().get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    plt.bar(np.arange(len(cell_counts)), sorted(cell_counts, reverse=True),
            color=colors)
    plt.ylabel('raw read counts')
    plt.xlabel('cells (sorted on highest to lowest)')
    plt.title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()
    
def genes_per_cell(obj, barcolor='#E69F00', title='expressed genes', filename=None):
    """Generates a bar plot of number of expressed genes per cell

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    barcolor : `str`
        Color of the bars. Default: "#E69F00"
    title : `str`
        Title of the plot. Default: "sequencing reads"
    filename : `str`, optional
        Write plot to file instead of showing it on the screen.

    Returns
    -------
    None
    """
    exp_mat = obj.exp_mat
    genes_expressed = exp_mat.apply(lambda x: sum(x > 0), axis=0)
    plt.clf()
    plt.gca().get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
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

def pca_contributors(obj, target='irlb', dim=range(0,2), top=10, color='#fcc603',
                     fontsize=6, fig_size=(10, 5), filename=None, **args):
    """Examine the top contributing genes for each PCA component
    
    Note
    ----
    Genes are ranked by their absolute value. Additional parameters are passed into
    :py:func:`matplotlib.pyplot.savefig`.
    
    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    target : `{'irlb', 'svd'}`
        The dimensional reduction to use. Default: irlb
    dim : :py:class:`range`
        Specifies the components to plot. For example: range(0,5) specifies the first five.
    top : `int`
        Specifies the number of top scoring genes to include. Default: 10
    color : `str`
        Color of the bars. As a string or hex code. Default: "#fcc603"
    fontsize : `int`
        Specifies font size. Default: 8
    fig_size : `tuple`
        Figure size in inches. Default: (10, 10)
    filename : `str`, optional
        Write to a file instead of showing the plot on screen.
        
    Returns
    -------
    None
    """
    if not target in obj.dr_gene_contr:
        raise Exception('Target %s not found' % target)
    if dim.stop > obj.dr_gene_contr[target].shape[1]:
        raise Exception('Number of requested dimensions cannot be higher than the number \
of generated PCA components.')
    contr = obj.dr_gene_contr[target][dim]
    
    plt.rcdefaults()
    f, ax = plt.subplots(1, contr.shape[1], figsize=fig_size)
    f.subplots_adjust(wspace=1)
    
    #f.suptitle('%s' % target)
    for k, d in contr.iteritems():
        d = d.sort_values(ascending=False)
        d = d.head(top)
        y_pos = np.arange(len(d))
        ax[k].barh(y_pos, d.values, color=YLW_CURRY)
        ax[k].set_yticks(y_pos)
        ax[k].set_yticklabels(d.index.values, fontsize=fontsize)
        ax[k].set_xlabel('abs(PCA score)', fontsize=fontsize)
        ax[k].set_title('comp. %s' % (k+1), fontsize=fontsize)
        ax[k].invert_yaxis() # labels read top-to-bottom
    #f.subplots_adjust(left=0.1, bottom=0.1)
    plt.tight_layout()
    if filename != None:
        plt.savefig(filename, **args)
    else:
        plt.show()

def cell_viz(obj, reduction='tsne', clustering=('leiden',), metadata=(),
             genes=(), ncols=2, filename=None, marker_size=0.8, font_size=8,
             colors='adobo', title=None, legend=True, min_cluster_size=0,
             fig_size=(10, 10), verbose=False):
    """Generates a 2d scatter plot from an embedding

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    reduction : `{'tsne', 'umap', 'irlb', 'svd'}`
        The dimensional reduction to use. Default: tsne
    clustering : `tuple`, optional
        Specifies the clustering outcomes to plot.
    metadata : `tuple`, optional
        Specifies the metadata variables to plot.
    genes : `tuple`, optional
        Specifies genes to plot.
    ncols : `int`
        Number of columns in the plotting grid. Default: 2
    filename : `str`, optional
        Name of an output file instead of showing on screen.
    marker_size : `float`
        The size of the markers.
    font_size : `float`
        Font size. Default: 8
    colors : `{'default', 'random'}` or `list`
        Can be: (i) "adobo" or "random"; or (ii) a `list` of colors with the same
        length as the number of factors. If colors is set to "adobo", then colors are
        retrieved from :py:attr:`adobo._constants.CLUSTER_COLORS_DEFAULT` (but if the
        number of clusters exceed 50, then random colors will be used). Default: adobo
    title : `str`
        Title of the plot. By default the title is set to the reduction technique.
    legend : `bool`
        Add legend or not. Default: True
    min_cluster_size : `int`
        Can be used to prevent clusters below a certain number of cells to be plotted.
        Only applicable if `what_to_color` is set to 'clusters'. Default: 0
    fig_size : `tuple`
        Figure size in inches. Default: (10, 10)
    verbose : `bool`
        Be verbose or not. Default: True

    Returns
    -------
    None
    """
    available_reductions = ('tsne', 'umap', 'irlb', 'svd')
    if not reduction in available_reductions:
        raise Exception('`reduction` must be one of %s.' % ', '.join(available_reductions))
    if not reduction in obj.dr:
        if len(obj.dr.keys()) > 0:
            t = (reduction, ', '.join(obj.dr.keys()))
            q = 'Reduction "%s" was not found, the following have been generated: %s' % t
        else:
            q = 'Reduction "%s" was not found. Run `ad.dr.tsne(...)` or \
`ad.dr.umap(...)` first.' % reduction
    if ncols < 1 or ncols > 50:
        raise Exception('`ncols` has an invalid value.')
    if marker_size<0:
        raise Exception('`marker_size` cannot be negative.')
    if title == None:
        title = reduction
    # cast to tuple if necessary
    if type(clustering) == str:
        clustering = (clustering,)
    if type(metadata) == str:
        metadata = (metadata,)
    if type(genes) == str:
        genes = (genes,)
    for k in clustering:
        if not k in obj.clusters:
            raise Exception('Clustering "%s" not found.' % k)
    for k in metadata:
        if not k in obj.meta_cells.columns:
            raise Exception('Meta data variable "%s" not found.' % k)
    for gene in genes:
        if not gene in obj.norm.index:
            raise Exception('%s was not found in the gene expression matrix' % gene)
    # setup colors
    if colors == 'adobo':
        colors = CLUSTER_COLORS_DEFAULT
        #if what_to_color == 'nothing':
        #    colors = ['black']
    elif colors == 'random':
        colors = unique_colors(len(groups))
        if verbose:
            print('Using random colors: %s' % colors)
    else:
        colors = colors
    n_plots = len(clustering) + len(metadata) + len(genes)
    # setup plotting grid
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    fig, aa = plt.subplots(nrows=int(np.ceil(n_plots/ncols)),
                           ncols=n_plots if n_plots < ncols else ncols,
                           figsize=fig_size)
    if not type(aa) is np.ndarray:
        aa = np.array([aa])
    aa = aa.flatten()
    # turn off unused axes
    i = len(aa)-n_plots
    for p in np.arange(i):
        aa[len(aa)-p-1].axis('off')
    # the embedding
    E = obj.dr[reduction]
    # plot index
    pl_idx = 0
    # plot clusterings
    for cl_algo in clustering:
        cl = obj.clusters[cl_algo]
        groups = np.unique(cl)
        
        if min_cluster_size > 0:
            z = pd.Series(dict(Counter(cl)))
            remove = z[z<min_cluster_size].index.values
            groups = groups[np.logical_not(pd.Series(groups).isin(remove))]
        
        for i, k in enumerate(groups):
            idx = np.array(cl) == k
            e = E[idx]
            col = colors[i]
            aa[pl_idx].scatter(e.iloc[:, 0], e.iloc[:, 1], s=marker_size, color=col)
        aa[pl_idx].set_title(cl_algo, size=font_size)
        if legend:
            aa[pl_idx].legend(list(groups), loc='upper left', markerscale=5,
                              bbox_to_anchor=(1, 1),
                              prop={'size': 5})
        pl_idx += 1
    # plot meta data variables
    for meta_var in metadata:
        m_d = obj.meta_cells.loc[obj.meta_cells.status=='OK', meta_var]
        if m_d.dtype.name == 'category':
            groups = np.unique(m_d)
            for i, k in enumerate(groups):
                idx = np.array(m_d) == k
                e = E[idx]
                col = colors[i]
                aa[pl_idx].scatter(e.iloc[:, 0], e.iloc[:, 1], s=marker_size, color=col)
            if legend:
                aa[pl_idx].legend(list(groups), loc='upper left', markerscale=5,
                                  bbox_to_anchor=(1, 1),
                                  prop={'size': 7})
        else:
            # If data are continuous
            cmap = sns.cubehelix_palette(as_cmap=True)
            po = aa[pl_idx].scatter(E.iloc[:, 0], E.iloc[:, 1], s=marker_size,
                                    c=m_d.values, cmap=cmap)
            cbar = fig.colorbar(po, ax=aa[pl_idx])
            #cbar.set_label('foobar')
        aa[pl_idx].set_title(meta_var, size=font_size)
        pl_idx += 1
    # plot genes
    for gene in genes:
        ge = obj.norm.loc[gene, :]
        cmap = sns.cubehelix_palette(as_cmap=True)
        po = aa[pl_idx].scatter(E.iloc[:, 0], E.iloc[:, 1], s=marker_size,
                                c=ge.values, cmap=cmap)
        cbar = fig.colorbar(po, ax=aa[pl_idx])
        aa[pl_idx].set_title(gene, size=font_size)
        pl_idx += 1
    for ax in aa:
        ax.set_ylabel('%s component 2' % reduction, size=font_size)
        ax.set_xlabel('%s component 1' % reduction, size=font_size)
    #plt.title(title)
    #fig.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    if filename != None:
        plt.savefig(filename, **args)
    else:
        plt.show()
