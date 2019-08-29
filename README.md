# Description
adobo for scRNA-seq

# Tutorials and API Documentation
Classes, functions and arguments are documented here:
* https://oscar-franzen.github.io/adobo/

# Install
```bash
# pip3
```

# Quick guide to get started
### Launch Python
```bash
python3
```

### Load the adobo package
```python
import adobo as ad
```

### Basic usage - loading a dataset of raw read counts
Create a new dataset object. This will be a new object containing your single cell data. All downstream operations and analyses are performed on this object and stored in it. The input file should be a gene expression matrix (rows as genes and cells as columns). Fields can be separated by any character (default is tab) and it can be changed with the `sep` parameter. The data matrix file can have a header or not (`header=0` indicates a header is present, otherwise use `header=None`). Most adobo functions also have a `verbose` parameter, which when set to `True` makes the function more noisy.
```python
data = ad.IO.load_from_file('input_single_cell_rnaseq_read_counts.mat',
                             verbose=True,
                             column_id=False)
```

Your gene expression data is stored in the attribute `data.exp_mat`, and after loading it is good to examine that the data were loaded properly:

```
data.exp_mat
```

Remove empty cells and genes:
```python
ad.preproc.remove_empty(data, verbose=True)

# detect and remove mitochondrial genes
ad.preproc.detect_mito(data, verbose=True)

# detect ERCC spikes
ad.preproc.detect_ERCC_spikes(data, verbose=True)
```

### Quality control and filtering
```python
# barplot of number of reads per cell
ad.plotting.barplot_reads_per_cell(data)

# barplot of number of expressed genes per cell
ad.plotting.barplot_genes_per_cell(data)

# apply a simple filter, requiring a minimum number of reads per cell and a minimum number
# of cells expressing a gene. For SMART-seq2 data, bump up the minreads option.
ad.preproc.simple_filter(data, minreads=1000, minexpgenes=0.001, verbose=True)
```

### Automatic detection of low quality cells
```python
# Load list of rRNA genes
import pandas as pd
rRNA = pd.read_csv('examples/rRNA_genes.txt', header=None)
rRNA = rRNA.iloc[:,0].values

ad.preproc.find_low_quality_cells(data, rRNA_genes=rRNA, verbose=True)
```

### Normalize
```python
# adobo supports several normalization strategies, the default scales by total read depth
# per cell
ad.normalize.norm(data, method='standard')

### Detect highly variable genes
```

### Display content of a dataset
To display the canonical string representation:
```python
repr(data)
```
To display a summary:
```python
data.info()
```

# Contact, bugs, etc
* Oscar Franzén, <p.oscar.franzen@gmail.com>

# Developer notes
* docstrings are in numpy format.
* API docs are generated with Sphinx
* traceback is suppressed by default, set `ad.debug=0` to enable full traceback
