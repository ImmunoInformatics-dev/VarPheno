# VarPheno

VarPheno integrates graph convolutional network and GNNExplainer model interpreter to embed somatic mutations into single cell regulatory landscapes and reveals their roles in shaping cell states and lineage dynamics.
![](https://github.com/ImmunoInformatics-dev/VarPheno/blob/main/framework/VarPheno.png)

## Install

```bash
conda env create -f environment.yml
conda activate VarPheno
```


## Run

python run_example.py

## Output

- Cell_Celltype_probability.csv
- GCN_evaluation.csv
- GCN_per_celltype_PRAUC.csv
- SNP_in_Celltype_score.csv
- GNNExplainer_evaluation.csv

