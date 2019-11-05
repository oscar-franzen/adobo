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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from sklearn.preprocessing import scale as sklearn_scale
import networkx as nx
import igraph as ig

import adobo
from .dr import svd
from ._constants import CLUSTER_COLORS_DEFAULT, YLW_CURRY
from ._colors import unique_colors

def overall_scatter(obj, color='#E69F00', title=None, filename=None):
    """Generates a scatter plot showing the total number of reads on one axis
       and the number of detected genes on the other axis

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    color : `str`
        Color of the plot. Default: '#E69F00'
    title : `str`
        Title of the plot. Default: None
    filename : `str`, optional
        Write plot to file instead of showing it on the screen. Default: None

    Returns
    -------
    None
    """
    plt.clf()
    plt.close(fig='all')
    count_data = obj.count_data
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ff = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(ff)
    ax.get_xaxis().set_major_formatter(ff)
    # summary statistics per cell
    reads = count_data.sum(axis=0)
    genes = [np.sum(r[1] > 0) for r in count_data.transpose().iterrows()]
    ax.scatter(x=reads, y=genes, color=color, s=2)
    ax.set_ylabel('detected genes')
    ax.set_xlabel('total reads')
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def overall(obj, what='reads', how='histogram', bin_size=100, color='#E69F00',
            title=None, filename=None):
    """Generates a plot of read counts per cell or expressed genes per cell

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
        A data class object.
    what : `{'reads', 'genes'}`
        If 'reads' then plots the number of reads per cell. If 'genes', then
        plots the number of expressed genes per cell. Default: 'reads'
    how : `{'histogram', 'boxplot', 'barplot', 'violin'}`
        Type of plot to generate. Default: 'histogram'
    bin_size : `int`
        If `how` is a histogram, then this is the bin size. Default: 100
    color : `str`
        Color of the plot. Default: '#E69F00'
    title : `str`
        Change the default title of the plot. Default: None
    filename : `str`, optional
        Write plot to file instead of showing it on the screen. Default: None

    Returns
    -------
    None
    """
    if not what in ('reads', 'genes'):
        raise Exception('"what" can only be "reads" or "genes".')
    if not how in ('histogram', 'boxplot', 'barplot', 'violin'):
        raise Exception('"how" can only be "histogram", "boxplot", "barplot" or "violin".')
    plt.clf()
    plt.close(fig='all')
    count_data = obj.count_data
    if what == 'reads':
        summary = count_data.sum(axis=0)
        ylab = 'raw read counts'
        xlab = 'cells'
    elif what == 'genes':
        summary = [np.sum(r[1] > 0) for r in count_data.transpose().iterrows()]
        ylab = 'detected genes'
        xlab = 'cells'
    colors = [color]*(len(summary))
    plt.close(fig='all')
    f, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    ff = matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))
    ax.get_yaxis().set_major_formatter(ff)
    ax.get_xaxis().set_major_formatter(ff)
    if how == 'barplot':
        ax.bar(np.arange(len(summary)), sorted(summary, reverse=True),
               color=colors)
        ax.set_xlabel('cells (sorted on highest to lowest)')
    elif how == 'boxplot':
        ax.boxplot(summary)
        ax.set_xlabel('cells')
        ax.set_xticklabels([])
        ax.set_xticks([])
    elif how == 'histogram':
        ax.hist(summary, bins=bin_size, color=color)
        ax.set_xlabel('cells')
        ylab = 'frequency'
        xlab = '%s bin' % what
    elif how == 'violin':
        parts = ax.violinplot(summary, showmedians=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
        xlab = ''
        ax.set_xticklabels([])
        ax.set_xticks([])
    else:
        raise Exception('The `how` parameter has an invalid value.')
    ax.set_ylabel(ylab)
    ax.set_xlabel(xlab)
    if not title and what == 'reads':
        title = 'total number of reads per cell'
    elif not title and what == 'genes':
        title = 'total number of expressed genes per cell'
    ax.set_title(title)
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

def pca_contributors(obj, name=None, dim=[0, 1, 2], top=10, color='#fcc603',
                     fontsize=6, fig_size=(10, 5), filename=None, verbose=False,
                     **args):
    """Examine the top contributing genes to each PCA component

    Note
    ----
    Genes are ranked by their absolute value. Additional parameters are passed
    into :py:func:`matplotlib.pyplot.savefig`.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    name : `str`
        The name of the normalization to operate on. If this is empty or None
        then the function will be applied on all normalizations available.
    dim : `list`
        A list of indices specifying components to plot. First component has
        index zero.
    top : `int`
        Specifies the number of top scoring genes to include. Default: 10
    color : `str`
        Color of the bars. As a string or hex code. Default: "#fcc603"
    fontsize : `int`
        Specifies font size. Default: 6
    fig_size : `tuple`
        Figure size in inches. Default: (10, 10)
    filename : `str`, optional
        Write to a file instead of showing the plot on screen.
    verbose : `bool`
        Be verbose or not. Default: False

    Returns
    -------
    Nothing
    """
    targets = {}
    if name is None or name == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    if len(targets) == 0:
        raise Exception('Nothing found to work on.')
    plt.clf()
    plt.close(fig='all')
    f, ax = plt.subplots(nrows=len(targets), ncols=len(dim), figsize=fig_size)
    if ax.ndim == 1:
        ax = [ax]
    f.subplots_adjust(wspace=1)
    count = 0
    for row, k in enumerate(targets):
        item = targets[k]
        if verbose:
            print('Plotting the %s normalization' % k)
        if not 'pca' in item['dr']:
            print('pca decomposition not found for the %s normalization' % k)
            for d in dim:
                ax[row][d].axis('off')
            continue
        count += 1
        contr = item['dr']['pca']['contr'][dim]
        idx = 0
        for i, d in contr.iteritems():
            d = d.sort_values(ascending=False)
            d = d.head(top)
            y_pos = np.arange(len(d))
            ax[row][idx].barh(y_pos, d.values, color=YLW_CURRY)
            ax[row][idx].set_yticks(y_pos)
            ax[row][idx].set_yticklabels(d.index.values, fontsize=fontsize)
            ax[row][idx].set_xlabel('abs(PCA score)', fontsize=fontsize)
            ax[row][idx].set_title('comp. %s' % (i+1), fontsize=fontsize)
            ax[row][idx].invert_yaxis() # labels read top-to-bottom
            if idx == 0:
                ax[row][idx].set_ylabel(k)
            idx += 1
    plt.tight_layout()
    if filename != None:
        plt.savefig(filename, **args)
    elif count:
        plt.show()

def cell_viz(obj, reduction='tsne', name=(), clustering=(), metadata=(),
             genes=(), edges=False, trajectory=None, filename=None, marker_size=0.8,
             font_size=8, colors='adobo', title=None, legend=True,
             min_cluster_size=10, fig_size=(10, 10), margins = None,
             verbose=False, **args):
    """Generates a 2d scatter plot from an embedding

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    reduction : `{'tsne', 'umap', 'pca', 'force_graph'}`
        The dimensional reduction to use. Default: tsne
    name : `tuple`
        A tuple of normalization to use. If it has the length zero, then all
        available normalizations will be used.
    clustering : `tuple`, optional
        Specifies the clustering outcomes to plot.
    metadata : `tuple`, optional
        Specifies the metadata variables to plot.
    genes : `tuple`, optional
        Specifies genes to plot.
    edges : `bool`
        Draw edges (only applicable if reduction='force_graph'). Default: False
    trajectory : `str`, optional
       The trajectory to plot. For example 'slingshot'. Default: None
    filename : `str`, optional
        Name of an output file instead of showing on screen.
    marker_size : `float`
        The size of the markers.
    font_size : `float`
        Font size. Default: 8
    colors : `{'default', 'random'}` or `list`
        Can be: (i) "adobo" or "random"; or (ii) a `list` of colors with the
        same length as the number of factors. If colors is set to "adobo", then
        colors are retrieved from :py:attr:`adobo._constants.CLUSTER_COLORS_DEFAULT`
        (but if the number of clusters exceed 50, then random colors will be
        used). Default: adobo
    title : `str`
        Title of the plot. By default the title is set to the reduction technique.
    legend : `bool`
        Add legend or not. Default: True
    min_cluster_size : `int`
        Can be used to prevent clusters below a certain number of cells to be
        plotted. Default: 10
    fig_size : `tuple`
        Figure size in inches. Default: (10, 10)
    margins : `dict`
        Can be used to adjust margins. Should be a dict with one or more of the
        keys: 'left', 'bottom', 'right', 'top', 'wspace', 'hspace'. Set
        verbose=True to figure out the present values. Default: None
    verbose : `bool`
        Be verbose or not. Default: True

    Returns
    -------
    None
    """
    avail_reductions = ('tsne', 'umap', 'pca', 'force_graph')
    D = obj.norm_data
    if not reduction in avail_reductions:
        raise Exception('`reduction` must be one of %s.' % ', '.join(avail_reductions))
    if marker_size < 0:
        raise Exception('`marker_size` cannot be negative.')
    # cast to tuple if necessary
    if type(clustering) == str:
        clustering = (clustering,)
    if type(metadata) == str:
        metadata = (metadata,)
    if type(genes) == str:
        genes = (genes,)
    if type(name) == str:
        name = (name,)
    if len(clustering) == 0:
        clustering = tuple({'q' : list(D[item]['clusters'].keys()) for item in D}['q'])
    if len(clustering) == 0:
        raise Exception('No clusterings found. Run `adobo.clustering.generate` first.')
    # setup colors
    if colors == 'adobo':
        colors = CLUSTER_COLORS_DEFAULT
    elif colors == 'random':
        colors = unique_colors(len(groups))
        if verbose:
            print('Using random colors: %s' % colors)
    else:
        colors = colors
    n_plots = len(clustering) + len(metadata) + len(genes) # per row
    #if n_plots == 1:
    #    n_plots = 2
    plt.rc('xtick', labelsize=font_size)
    plt.rc('ytick', labelsize=font_size)
    targets = {}
    if len(name) == 0:
        targets = D
    else:
        targets[name] = D[name]
    # setup plotting grid
    plt.clf()
    plt.close(fig='all')
    fig, aa = plt.subplots(nrows=len(targets), ncols=n_plots, figsize=fig_size,
                           constrained_layout=True)
    if not isinstance(aa, np.ndarray):
        aa = np.array([aa])
    if reduction == 'pca':
        red_key = 'comp'
    elif reduction == 'force_graph':
        red_key = 'coords'
    else:
        red_key = 'embedding'
    ttl = fig.suptitle(reduction)
    if len(targets) == 2 and n_plots == 1:
        aa = np.vstack(aa)
    for row, l in enumerate(targets):
        item = targets[l]
        # the embedding
        if not reduction in item['dr']:
            q = 'Reduction "%s" was not found. Run `ad.dr.tsne(...)` or \
`ad.dr.umap(...)` first.' % reduction
            raise Exception(q)
        if reduction == 'force_graph' and edges and not 'graph' in item:
            raise Exception('Graph has not been generated. Run \
`adobo.clustering.generate(...)` first.')
        E = item['dr'][reduction][red_key]
        pl_idx = 0 # plot index
        markerscale = 5
        if marker_size > 5: markerscale = 5/2
        # plot clusterings
        for cl_algo in clustering:
            if not cl_algo in item['clusters']:
                raise Exception('Clustering "%s" not found.' % cl_algo)
            cl = item['clusters'][cl_algo]['membership']
            groups = np.unique(cl)
            if min_cluster_size > 0:
                z = pd.Series(dict(Counter(cl)))
                remove = z[z < min_cluster_size].index.values
                groups = groups[np.logical_not(pd.Series(groups).isin(remove))]
            for i, k in enumerate(groups):
                idx = np.array(cl) == k
                e = E[idx]
                col = colors[i]
                aa[row][pl_idx].scatter(e.iloc[:, 0], e.iloc[:, 1], s=marker_size,
                                        color=col)
            aa[row][pl_idx].set_title(cl_algo, size=font_size)
            aa[row][pl_idx].set_aspect('equal')
            if pl_idx == 0:
                aa[row][pl_idx].set_ylabel(l)
            if legend:
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                aa[row][pl_idx].legend(list(groups), loc='upper left',
                                       markerscale=markerscale,
                                       bbox_to_anchor=(1, 1),
                                       prop={'size': font_size})
            if trajectory == 'slingshot':
                # cluster weights matrix
                l = np.array([(cl == clID).astype(int) for clID in groups])
                l = l.transpose()
                centers = []
                for clID in groups:
                    w = l[:, clID]
                    centers.append(np.average(E, axis=0, weights=w))
                centers = np.array(centers)
                aa[row][pl_idx].plot(centers[:, 0], centers[:, 1], 'bo', color='blue')
                adj = obj.norm_data['standard']['slingshot'][cl_algo]['adjacency']
                for i in np.arange(0, max(groups)+1):
                    for j in np.arange(i, max(groups))+1:
                        if adj.iloc[i, j] or adj.iloc[j, i]:
                            xy = centers[(i,j),:]
                            aa[row][pl_idx].plot(xy[:, 0], xy[:, 1], color='black')
            if edges and reduction == 'force_graph':
                d = {}
                for k, i in E.iterrows():
                    d[k] = i.values
                snn_graph = item['graph']
                nn = set(snn_graph[snn_graph.columns[0]])
                g = ig.Graph()
                g.add_vertices(len(nn))
                g.vs['name'] = list(range(1, len(nn)+1))
                ll = []
                for i in snn_graph.itertuples(index=False):
                    ll.append(tuple(i))
                g.add_edges(ll)
                A = g.get_edgelist()
                GG = nx.Graph(A)
                edge_collection = nx.draw_networkx_edges(GG, d, ax=aa[row][pl_idx],
                                                         width=0.3, edge_color='grey',
                                                         alpha=0.5)
                edge_collection.set_zorder(-2)
            pl_idx += 1
        # plot meta data variables
        for meta_var in metadata:
            if not meta_var in obj.meta_cells.columns:
                raise Exception('Meta data variable "%s" not found.' % k)
            m_d = obj.meta_cells.loc[obj.meta_cells.status == 'OK', meta_var]
            if m_d.dtype.name == 'category':
                groups = np.unique(m_d)
                for i, k in enumerate(groups):
                    idx = np.array(m_d) == k
                    e = E[idx]
                    col = colors[i]
                    aa[row][pl_idx].scatter(e.iloc[:, 0], e.iloc[:, 1],
                                            s=marker_size, color=col)
                if legend:
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    aa[row][pl_idx].legend(list(groups), loc='upper left',
                                           markerscale=markerscale,
                                           bbox_to_anchor=(1, 1),
                                           prop={'size': font_size})
            else:
                # If data are continuous
                cmap = sns.cubehelix_palette(as_cmap=True)
                po = aa[row][pl_idx].scatter(E.iloc[:, 0], E.iloc[:, 1],
                                             s=marker_size, c=m_d.values,
                                             cmap=cmap)
                cbar = fig.colorbar(po, ax=aa[row][pl_idx])
                #cbar.set_label('foobar')
            aa[row][pl_idx].set_title(meta_var, size=font_size)
            aa[row][pl_idx].set_aspect('equal')
            pl_idx += 1
        # plot genes
        for gene in genes:
            if not gene in item['data'].index:
                m = '"%s" was not found in the gene expression matrix' % gene
                raise Exception(m)
            ge = item['data'].loc[gene, :]
            cmap = sns.cubehelix_palette(as_cmap=True)
            po = aa[row][pl_idx].scatter(E.iloc[:, 0], E.iloc[:, 1],
                                         s=marker_size, c=ge.values, cmap=cmap)
            divider = make_axes_locatable(aa[row][pl_idx])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(po, cax=cax)
            aa[row][pl_idx].set_title(gene, size=font_size)
            aa[row][pl_idx].set_aspect('equal')
            pl_idx += 1
        # turn off unused axes
        #if (len(clustering) + len(metadata) + len(genes)) == 1:
        #    aa[row][1].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if verbose:
        print('fig.subplotpars.top: %s' % fig.subplotpars.top)
        print('fig.subplotpars.bottom: %s' % fig.subplotpars.bottom)
        print('fig.subplotpars.left: %s' % fig.subplotpars.left)
        print('fig.subplotpars.right: %s' % fig.subplotpars.right)
        print('fig.subplotpars.hspace: %s' % fig.subplotpars.hspace)
        print('fig.subplotpars.wspace: %s' % fig.subplotpars.wspace)
    if margins:
        fig.subplots_adjust(**margins)
    if filename != None:
        plt.savefig(filename, **args)
    else:
        plt.show()

def pca_elbow(obj, name=(), comp_max=200, all_genes=False, filename=None, font_size=8,
              fig_size=(10, 10), verbose=True, **args):
    """Generates a PCA elbow plot

    Notes
    -----
    Can be used for determining the number of principal components to include. PCA is
    based on singular value decomposition.

    Parameters
    ----------
    obj : :class:`adobo.data.dataset`
          A data class object
    name : `tuple`
        A tuple of normalization to use. If it has the length zero, then all available
        normalizations will be used.
    comp_max : `int`
        Maximum number of components to include.
    all_genes : `bool`
        Run on all genes, i.e. not only highly variable genes. Default: False
    filename : `str`, optional
        Name of an output file instead of showing on screen.
    font_size : `float`
        Font size. Default: 8
    fig_size : `tuple`
        Figure size in inches. Default: (10, 10)
    verbose : `bool`
        Be verbose or not. Default: True

    Returns
    -------
    Nothing.
    """
    targets = {}
    if len(name) == 0 or name == '':
        targets = obj.norm_data
    else:
        targets[name] = obj.norm_data[name]
    # setup plotting grid
    plt.clf()
    plt.close(fig='all')
    fig, aa = plt.subplots(nrows=1,
                           ncols=len(targets),
                           figsize=fig_size)
    if len(targets) == 1:
        aa = [aa]
    else:
        aa = aa.flatten()
    for i, k in enumerate(targets):
        if verbose:
            print('Running %s' % k)
        item = targets[k]
        if not 'hvg' in item and not all_genes:
            raise Exception('Run adobo.dr.find_hvg() first.')
        X = item['data']
        if not all_genes:
            hvg = item['hvg']['genes']
            X = X[X.index.isin(hvg)]
        else:
            if verbose:
                print('Using all genes')
        d_scaled = sklearn_scale(X.transpose(),     # cells as rows and genes as columns
                                 axis=0,            # over genes, i.e. features (columns)
                                 with_mean=True,    # subtracting the column means
                                 with_std=True)     # scale the data to unit variance
        d_scaled = pd.DataFrame(d_scaled.transpose(), index=X.index)
        sdev = svd(d_scaled, ncomp=None, only_sdev=True)
        var = sdev**2
        pvar = var/sum(var)
        cs = np.cumsum(pvar)[0:comp_max]
        aa[i].plot(cs)
        aa[i].set_ylabel('cumulative variance (percent of total)')
        aa[i].set_xlabel('components')
        aa[i].set_title(k)
    if filename != None:
        plt.savefig(filename, **args)
    else:
        plt.show()
