# VarPheno

VarPheno integrates graph convolutional network and GNNExplainer model interpreter to embed somatic mutations into single cell regulatory landscapes and reveals their roles in shaping cell states and lineage dynamics.
![](https://github.com/ImmunoInformatics-dev/VarPheno/blob/main/framework/VarPheno.png)

## Install

You can choose to install the CPU or GPU version.
```bash
conda env create -f environment.gpu.yml #conda env create -f environment.cpu.yml
conda activate VarPheno
```
You can also install VarPheno following the steps:

```bash
conda create -n VarPheno python=3.10 -y
conda activate VarPheno

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch-geometric
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm scanpy anndata
```

## Data Preparation

### (1) Feature Matrix

Three input files should be prepared: 

$M^{(1)}$: 1-SNP_SampleCellType_Matrix.txt

|  | Sample1_Celltype1 | Sample1_Celltype2 |   ...   | Sample10_Celltype9 | Sample10_Celltype10 |
|---|:---:|:---:|:---:|:---:|:---:|
| chr1-838667-G-A | 1 | 0 | ... | 0 | 0 |
| chr1-890636-T-C | 0 | 0 | ... | 0 | 1 |
| chr1-991241-A-C | 0 | 1 | ... | 0 | 0 |

$M^{(2)}$: 2-SampleCelltype_Cell.txt 

|  | Cell1 | Cell2 | Cell3 | Cell4 |... | Cell98 | Cell99 | Cell100 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Sample1_Celltype1 | 1 | 1 | 0 | 1 | ... | 0 | 1 | 0 |
| Sample1_Celltype2 | 0 | 0 | 1 | 1 | ... | 0 | 0 | 1 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Sample10_Celltype9 | 0 | 0 | 0 | 0 | ... | 1 | 1 | 1 |
| Sample10_Celltype10 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 |

$M^{(3)}$: 3-Peaks_Cell.pkl

|   | Cell1 | Cell2 | Cell3 | Cell4 |... | Cell98 | Cell99 | Cell100 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| chr1-838400-838899 | 0.12 | 0.00 | 0.35 | 0.00 | ... | 0.08 | 0.00 | 0.00 |
| chr1-890400-890899 | 0.00 | 0.21 | 0.00 | 0.00 | ... | 0.00 | 0.14 | 0.00 |
| chr1-991000-991499 | 0.43 | 0.00 | 0.18 | 0.00 | ... | 0.00 | 0.00 | 0.00 |



```math
X_{\mathrm{SNP-Cell}}=\left(M^{\left(1\right)}M^{\left(2\right)}\right)\odot M^{\left(3\right)}
```
where $M^{(1)}$ denote an SNP-sample binary matrix indicating whether each SNP was detected in samples, $M^{(2)}$ is a sample-cell matrix used to record the sample source of each cell, $M^{(3)}$ denote the peak-cell normalized matrix obtained from scATAC-seq data. Only peaks that exited SNPs would be retained.



```bash
from Varpheno import prepare_feature_matrix

prepare_feature_matrix(
    snp_sample_file="ExampleData/1-SNP_SampleCellType_Matrix.txt",
    sample_cell_file="ExampleData/2-SampleCelltype_Cell.txt",
    peaks_cell_file="ExampleData/3-Peaks_Cell.pkl",
    out_npz="ExampleData/Feature.npz"
)
```

### (2) Edge Matrix

Snn_connectivities.pkl should be prepared:

|   | Cell1 | Cell2 | Cell3 | Cell4 |... | Cell98 | Cell99 | Cell100 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Cell1 | 1 | 0 | 0 | 0 | ... | 0 | 1 | 0 |
| Cell2 | 0 | 1 | 0 | 1 | ... | 0 | 0 | 0 |
| Cell3 | 0 | 0 | 1 | 0 | ... | 0 | 0 | 0 |
| Cell4 | 0 | 0 | 0 | 1 | ... | 0 | 0 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| Cell98 | 0 | 0 | 0 | 0 | ... | 1 | 0 | 0 |
| Cell99 | 0 | 1 | 0 | 0 | ... | 0 | 1 | 0 |
| Cell100 | 0 | 0 | 0 | 0 | ... | 0 | 0 | 1 |


The edge matrix is obtained by calculating the cosine similarity of the Feature matrix, and impose restrictions based on the SNN structure. Formally, the adjacency matrix $A\in\mathbb{R}^{N\times N}$ was defined as:
```math
A_{ij} =
\begin{cases}
\mathrm{sim}(x_i, x_j), & S_{ij} > 0 \\
0, & S_{ij} = 0
\end{cases}
```
where $S_{ij}$ denotes the SNN connectivity between cells $i$ and $j$. $N$ denotes the number of cells.

```bash
from Varpheno import prepare_edge_matrix

prepare_edge_matrix(
    snp_cell_npz="ExampleData/Feature.npz",
    snn_file="ExampleData/Snn_connectivities.pkl",
    out_npy="ExampleData/Edge.npy"
)
```

### (3) Cell label

The cell type to which the cells belong in the 2-SampleCelltype_Cell.txt file.

### (4) SNP list

SNP list should be prepared, and the order of the SNP list is consistent with the column names in 1-SNP_SampleCellType_Matrix.txt.

## Run 

```bash
from Varpheno import run_snp_celltype_score

run_snp_celltype_score(
    matrix_npz = "ExampleData/Feature.npz",
    Edge_npy = "ExampleData/Edge.npy",
    label_csv = "ExampleData/Cell_label.csv",
    snp_name_file = "ExampleData/SNP.txt",
    outdir="Output"
)
```

## Output

- Cell_Celltype_probability.csv
- GCN_evaluation.csv
- GCN_per_celltype_PRAUC.csv
- SNP_in_Celltype_score.csv
- GNNExplainer_evaluation.csv

## Cite
If you have used VarPheno, please refer to the following article: 
