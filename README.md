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

## Run

```bash
from Varpheno import run_snp_celltype_score

res = run_snp_celltype_score(
    matrix_npz = "ExampleData/Feature.npz",
    label_csv = "ExampleData/Cell_label.csv",
    similarity_npy = "ExampleData/Similarity_Snn.npy",
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

