adobo for scRNA-seq

# Launch Python
```bash
python3
```

# Basic usage - loading a dataset of raw read counts
```python
import adobo
# create a new data object
d = adobo.data()
d.load_from_file('input.mat', verbose=True, column_id=False)

# remove empty cells and genes
d.remove_empty(verbose=True)

# detect and remove mitochondrial genes
d.detect_mito(verbose=True)

# detect ERCC spikes
d.detect_ERCC_spikes(verbose=True)
```

# Quality control and filtering
```python
# barplot of number of reads per cell
d.barplot_reads_per_cell()

# barplot of number of expressed genes per cell
d.barplot_genes_per_cell()

# apply a simple filter, requiring minimum number of reads per cell and a minimum
# number of cells expressing a gene
d.simple_filters(minreads=1000, minexpgenes=0.001, verbose=True)
```

# Automatic detection of low quality cells
```python
# Load list of rRNA genes
import pandas as pd
rRNA = pd.read_csv('examples/rRNA_genes.txt', header=None)
rRNA = rRNA.iloc[:,0].values

d.auto_clean(rRNA_genes=rRNA)
```
