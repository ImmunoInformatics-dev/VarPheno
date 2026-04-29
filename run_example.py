from varpheno import run_snp_celltype_score

res = run_snp_celltype_score(
    matrix_npz = "ExampleData/Feature.npz",
    label_csv = "ExampleData/Cell_label.csv",
    similarity_npy = "ExampleData/Similarity_Snn.npy",
    snp_name_file = "ExampleData/SNP.txt",
    outdir="OutPut"
)
