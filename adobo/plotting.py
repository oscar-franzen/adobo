# adobo.
#
# Description: An analysis framework for scRNA-seq data.
#  How to use: https://github.com/oscar-franzen/adobo/
#     Contact: Oscar Franzen <p.oscar.franzen@gmail.com>
"""
Summary
-------
Functions for plotting scRNA-seq data.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from ._constants import CLUSTER_COLORS_DEFAULT, YLW_CURRY
from ._colors import unique_colors

def reads_per_cell(obj, barcolor='#E69F00', filename=None,
                           title='sequencing reads'):
    """Generates a bar plot of read counts per cell

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    barcolor : `str`, optional
        Color of the bars (default: "#E69F00").
    filename : `str`, optional
        Write plot to file.
    title : `str`, optional
        Title of the plot (default: "sequencing reads").

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
    
def genes_per_cell(obj, barcolor='#E69F00', filename=None,
                           title='expressed genes'):
    """Generates a bar plot of number of expressed genes per cell

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    barcolor : `str`, optional
        Color of the bars (default: "#E69F00").
    filename : `str`, optional
        Write plot to file.
    title : `str`, optional
        Title of the plot (default: "sequencing reads").

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

def cell_viz(obj, reduction='tsne', what_to_color='clusters', filename=None,
             marker_size=0.8, font_size=8, cluster_colors='adobo', title='', legend=True,
             verbose=True):
    """Generates a 2d scatter plot from an embedding

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    reduction : `{'tsne', 'umap', 'irlb', 'svd'}`
        The dimensional reduction to use. Default: tsne
    what_to_color : `{'clusters', 'nothing'}`
        Specifies what to color. If this is 'clusters', then clustering results will be
        colored. 'nothing' indicates not to color anything. Default: 'clusters'
    filename : `str`, optional
        Name of an output file instead of showing on screen.
    marker_size : `float`
        The size of the markers.
    font_size : `float`
        Font size. Default: 8
    cluster_colors : `{'default', 'random'}` or `list`
        Can be: (i) "adobo" or "random"; (ii) a list of colors with the same
        length as the number of cells (same order as cells occur in the normalized
        matrix). If cluster_colors is set to "adobo", then colors are retrieved from
        :py:attr:`adobo._constants.CLUSTER_COLORS_DEFAULT` (but if the number of clusters
        exceed 50, then random colors will be used). Default: adobo
    title : `str`
        Title of the plot.
    legend : `bool`
        Add legend or not. Default: True
    verbose : `bool`
        Be verbose or not. Default: True

    Returns
    -------
    None
    """
    if not what_to_color in ('clusters', 'nothing'):
        raise Exception('`what_to_color` can be "clusters" or "nothing".')
    available_reductions = ('tsne', 'umap', 'irlb', 'svd')
    if not reduction in available_reductions:
        raise Exception('`reduction` must be one of %s' % ', '.join(available_reductions))
    if not reduction in obj.dr:
        t = (reduction, ', '.join(obj.dr.keys()))
        q = 'Reduction "%s" was not found, the following have been generated: %s' % t
        raise Exception(q)
    if marker_size<0:
        raise Exception('Marker size cannot be negative.')
    E = obj.dr[reduction]
    if what_to_color == 'nothing' or len(obj.clusters) == 0:
        cl = [0]*E.shape[0]
        if verbose:
            print('Clustering has not been performed. Plotting anyway.')
    elif what_to_color == 'clusters':
        cl = obj.clusters[len(obj.clusters)-1]['cl']
    n_clusters = len(np.unique(cl))
    if cluster_colors == 'adobo':
        colors = CLUSTER_COLORS_DEFAULT
        if what_to_color == 'nothing':
            colors = ['black']
    elif cluster_colors == 'random':
        colors = unique_colors(n_clusters)
    else:
        colors = cluster_colors
    plt.clf()
    f, ax = plt.subplots(1, 1)
    for i in range(n_clusters):
        idx = np.array(cl) == i
        e = E[idx]
        col = colors[i]
        ax.scatter(e.iloc[:, 0], e.iloc[:, 1], s=marker_size, color=col)
    if legend:
        ax.legend(list(range(n_clusters)), loc='upper left', markerscale=5,
                   bbox_to_anchor=(1, 1))
    ax.set_ylabel('Component 2', size=font_size)
    ax.set_xlabel('Component 1', size=font_size)
    f.tight_layout()
    if filename != None:
        plt.savefig(filename, **args)
    else:
        plt.show()

def pca_contributors(obj, target='irlb', dim=range(0,5), top=10, filename=None,
                     fontsize=8, **args):
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
    fontsize : `int`
        Specifies font size. Default: 8
    color : `str`
        Color of the bars. As a string or hex code. Default: "#fcc603"
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
    f, ax = plt.subplots(1, contr.shape[1])
    f.tight_layout()
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
    f.subplots_adjust(left=0.1, bottom=0.1)
    if filename != None:
        plt.savefig(filename, **args)
    else:
        plt.show()
