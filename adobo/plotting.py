import numpy as np
import matplotlib.pyplot as plt

def barplot_reads_per_cell(obj, barcolor='#E69F00', filename=None,
                           title='sequencing reads'):
    """Generates a bar plot of read counts per cell

    Parameters
    ----------
        obj : data
              A data class object
        barcolor : str, optional
                   Color of the bars (default is orange)
        filename : str, optional
                   Write plot to file
        title : str, optional
                Title of the plot

    Returns
    -------
    None   
    """
    exp_mat = obj.exp_mat
    cell_counts = exp_mat.sum(axis=0)
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
    
def barplot_genes_per_cell(obj, barcolor='#E69F00', filename=None,
                           title='expressed genes'):
    """Generates a bar plot of number of expressed genes per cell

    Parameters
    ----------
        obj : data
              A data class object
        barcolor : str, optional
                   Color of the bars (default is orange)
        filename : str, optional
                   Write plot to file
        title : str, optional
                Title of the plot

    Returns
    -------
    None   
    """
    exp_mat = obj.exp_mat
    genes_expressed = exp_mat.apply(lambda x: sum(x > 0), axis=0)

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
