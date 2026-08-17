# VarPheno

VarPheno is a computational framework for interpreting the functional impact of somatic mutations in single-cell regulatory landscapes and characterizing their associations with cell states and lineage dynamics.

## Install

You can install VarPheno following the steps:

```bash
conda create -n VarPheno python=3.10 -y
conda activate VarPheno

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install torch-geometric
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm scanpy anndata
```

## Run 

```bash
from VarPheno import run_snv_celltype_score

run_snv_celltype_score(
    matrix_npz = "ExampleData/Feature.npz",
    Edge_npy = "ExampleData/Edge.npy",
    label_csv = "ExampleData/Cell_label.csv",
    snv_name_file = "ExampleData/SNV.txt",
    outdir="Output"
)
```

## Output

- Cell_Celltype_probability.csv
- GCN_evaluation.csv
- GCN_per_celltype_PRAUC.csv
- SNV_in_Celltype_score.csv
- GNNExplainer_evaluation.csv

## Cite
If you have used VarPheno, please refer to the following article: 
