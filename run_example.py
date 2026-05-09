from VarPheno import prepare_feature_matrix

prepare_feature_matrix(
    snv_sample_file="ExampleData/1-SNV_SampleCellType_Matrix.txt",
    sample_cell_file="ExampleData/2-SampleCelltype_Cell.txt",
    peaks_cell_file="ExampleData/3-Peaks_Cell.pkl",
    out_npz="ExampleData/Feature.npz"
)

from VarPheno import prepare_edge_matrix

prepare_edge_matrix(
    snv_cell_npz="ExampleData/Feature.npz",
    snn_file="ExampleData/Snn_connectivities.npz",
    out_npy="ExampleData/Edge.npy"
)

from VarPheno import run_snv_celltype_score

run_snv_celltype_score(
    matrix_npz = "ExampleData/Feature.npz",
    Edge_npy = "ExampleData/Edge.npy",
    label_csv = "ExampleData/Cell_label.csv",
    snv_name_file = "ExampleData/SNV.txt",
    outdir="Output"
)
