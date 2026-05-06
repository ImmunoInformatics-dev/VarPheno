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

You need to prepare four input files.

(1) Feature Matrix
```math
X_{\mathrm{SNP-Cell}}=\left(M^{\left(1\right)}M^{\left(2\right)}\right)\odot M^{\left(3\right)}
```
where $M^{(1)}\in\mathbb{R}^{S\times K}$ denote an SNP-sample binary matrix indicating whether each SNP was detected in samples, $M^{(2)}\in\mathbb{R}^{K\times N}$ is a sample-cell matrix used to record the sample source of each cell, $M^{(3)}\in\mathbb{R}^{S\times N}$ denote the peak-cell normalized matrix obtained from Signac pipeline, only peaks that exited SNPs would be retained. $K$ is the number of samples.

```bash
from Varpheno import prepare_snp_cell_matrix

C = prepare_snp_cell_matrix(
    snp_sample_file="ExampleData/1-SNP_SampleCellType_Matrix.txt",
    sample_cell_file="ExampleData/2-SampleCelltype_Cell.txt",
    peaks_cell_file="ExampleData/3-Peaks_Cell.pkl",
    out_npz="ExampleData/Feature.npz"
)
```

(2) Edge Matrix

The edge matrix is obtained by calculating the cosine similarity of the Feature matrix, and impose restrictions based on the SNN structure. Formally, the adjacency matrix $A\in\mathbb{R}^{N\times N}$ was defined as:
```math
A_{ij} =
\begin{cases}
\mathrm{sim}(x_i, x_j), & S_{ij} > 0 \\
0, & S_{ij} = 0
\end{cases}
```
where $S_{ij}$ denotes the SNN connectivity between cells $i$ and $j$.

(3) Cell label

Cell type labels of cells.

(4) SNP list


## Run 

```bash
from Varpheno import run_snp_celltype_score

res = run_snp_celltype_score(
    matrix_npz = "ExampleData/Feature.npz",
    Edge_npy = "ExampleData/Edge.npy",
    label_csv = "ExampleData/Cell_label.csv",
    snp_name_file = "ExampleData/SNP.txt",
    outdir="OutPut"
)
```

## Output

- Cell_Celltype_probability.csv
- GCN_evaluation.csv
- GCN_per_celltype_PRAUC.csv
- SNP_in_Celltype_score.csv
- GNNExplainer_evaluation.csv

