import os
import gc
import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

from torch_geometric.nn import GCNConv
from torch_geometric.explain import Explainer, unfaithfulness
from torch_geometric.explain.algorithm import GNNExplainer


def prepare_feature_matrix(
    snv_sample_file,
    sample_cell_file,
    peaks_cell_file,
    out_npz="SNV_Cell_matrix.npz"
):
    print("[1/3] Loading SNV x Sample_Celltype, Sample_Celltype x Cell, peak activity matrix...")
    SNV_Sample = pd.read_csv(
        snv_sample_file,
        header=0,
        sep="\t",
        index_col=0
    ).values

    Sample_Cell = pd.read_csv(
        sample_cell_file,
        header=0,
        sep="\t",
        index_col=0
    ).values

    Peaks_Cell = sp.load_npz(peaks_cell_file).toarray()
    Peaks_Cell = sp.csc_matrix(Peaks_Cell.astype("float64"))
    
    print("[2/3] Calculating SNV x Cell matrix...")
    SNV_Sample = sp.csr_matrix(SNV_Sample.astype("float64"))
    Sample_Cell = sp.csr_matrix(Sample_Cell.astype("float64"))

    AB = SNV_Sample.dot(Sample_Cell).tocsr()

    print("[3/3] Calculating Feature matrix...")
    C = AB.multiply(Peaks_Cell)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True) if os.path.dirname(out_npz) else None
    sp.save_npz(out_npz, C)

    print("[OK] SNV x Cell matrix generated.")
    print(f"[OK] Saved to: {out_npz}")
    print(f"[INFO] Output shape: {C.shape}")

    return C

def prepare_edge_matrix(
    snv_cell_npz,
    snn_file,
    out_npy="Edge.npy"
):
    print("[1/4] Loading Feature matrix...")
    aa = sp.load_npz(snv_cell_npz)

    print("[2/4] Calculating Cell x Cell cosine similarity...")
    result_features = torch.tensor(
        aa.T.toarray(),
        dtype=torch.float32
    )

    similarity_matrix = cosine_similarity(result_features)

    print("[3/4] Loading SNN connectivity matrix...")
    snn = sp.load_npz(snn_file).toarray()

    if hasattr(snn, "values"):
        snn = snn.values

    print("[4/4] Filtering Edge matrix by SNN...")
    similarity_matrix[snn == 0] = 0

    os.makedirs(os.path.dirname(out_npy), exist_ok=True) if os.path.dirname(out_npy) else None
    np.save(out_npy, similarity_matrix)

    print("[OK] SNN-filtered similarity matrix generated.")
    print(f"[OK] Saved to: {out_npy}")
    print(f"[INFO] Output shape: {similarity_matrix.shape}")

    return similarity_matrix
    
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, output_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def plot_training(train_loss, val_loss, train_acc, val_acc, out):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train")
    plt.plot(val_acc, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def evaluate_model(prob, labels, val_idx, label_names):
    y_true = labels[val_idx].cpu().numpy()
    y_prob = prob[val_idx].detach().cpu().numpy()
    y_pred = y_prob.argmax(axis=1)

    y_onehot = F.one_hot(
        labels[val_idx],
        num_classes=len(label_names)
    ).cpu().numpy()

    res = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_macro": roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="macro"),
        "prauc_macro": average_precision_score(y_onehot, y_prob, average="macro"),
        "auc_micro": roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="micro"),
        "prauc_micro": average_precision_score(y_onehot, y_prob, average="micro"),
        "auc_weighted": roc_auc_score(y_onehot, y_prob, multi_class="ovr", average="weighted"),
        "prauc_weighted": average_precision_score(y_onehot, y_prob, average="weighted"),
    }

    per_class = []
    for i, ct in enumerate(label_names):
        per_class.append({
            "Celltype": ct,
            "PR_AUC": average_precision_score(y_onehot[:, i], y_prob[:, i])
        })

    return pd.DataFrame([res]), pd.DataFrame(per_class)


def aggregate_snv_by_celltype(node_mask, labels, label_names, snv_names=None):
    node_mask = node_mask.detach().cpu()
    labels_cpu = labels.cpu()

    out = []
    for i, ct in enumerate(label_names):
        idx = labels_cpu == i
        score = node_mask[idx].mean(dim=0).numpy()
        out.append(score)

    df = pd.DataFrame(out, index=label_names)

    if snv_names is not None:
        df.columns = snv_names
    else:
        df.columns = [f"SNV_{i}" for i in range(df.shape[1])]

    return df


def run_snv_celltype_score(
    matrix_npz,
    label_csv,
    Edge_npy,
    outdir="./OutPut",
    snv_name_file=None,
    hidden_dim1=256,
    hidden_dim2=125,
    dropout=0.3,
    lr=1e-4,
    epochs=250,
    test_size=0.3,
    random_state=42,
    explainer_epochs=300
):
    os.makedirs(outdir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    print("[1/7] Loading input files...")

    data = sp.load_npz(matrix_npz)
    labels_raw = pd.read_csv(label_csv, header=None)[0].astype(str)
    similarity = np.load(Edge_npy)

    le = LabelEncoder()
    labels_np = le.fit_transform(labels_raw)
    label_names = list(le.classes_)

    if snv_name_file is not None:
        snv_names = pd.read_csv(snv_name_file, header=None)[0].astype(str).tolist()
    else:
        snv_names = None

    print("[OK] Input files loaded.")
    print(f"[INFO] SNV x Cell matrix shape: {data.shape}")
    print(f"[INFO] Number of cells: {len(labels_np)}")
    print(f"[INFO] Number of cell types: {len(label_names)}")

    print("[2/7] Building graph data...")

    features = torch.tensor(data.T.toarray(), dtype=torch.float32).to(device)

    adj = coo_matrix(similarity)
    edge_index = torch.tensor(
        np.vstack((adj.row, adj.col)),
        dtype=torch.long
    ).to(device)

    labels = torch.tensor(labels_np, dtype=torch.long).to(device)

    print("[OK] Graph data prepared.")
    print(f"[INFO] Feature matrix shape: {features.shape}")
    print(f"[INFO] Edge index shape: {edge_index.shape}")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    train_idx, val_idx = next(sss.split(np.arange(len(labels_np)), labels_np))
    train_idx = torch.tensor(train_idx, dtype=torch.long).to(device)
    val_idx = torch.tensor(val_idx, dtype=torch.long).to(device)

    print(f"[INFO] Training cells: {len(train_idx)}")
    print(f"[INFO] Validation cells: {len(val_idx)}")

    print("[3/7] Training GCN model...")

    model = GCN(
        input_dim=features.shape[1],
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        output_dim=len(label_names),
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    best_val_loss = float("inf")
    best_model_path = os.path.join(outdir, "best_model.pth")

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(features, edge_index)
        loss = F.nll_loss(out[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        pred_train = out[train_idx].argmax(dim=1)
        acc_train = (pred_train == labels[train_idx]).float().mean().item()

        model.eval()
        with torch.no_grad():
            out_val = model(features, edge_index)
            loss_val = F.nll_loss(out_val[val_idx], labels[val_idx])
            pred_val = out_val[val_idx].argmax(dim=1)
            acc_val = (pred_val == labels[val_idx]).float().mean().item()

        train_loss.append(loss.item())
        val_loss.append(loss_val.item())
        train_acc.append(acc_train)
        val_acc.append(acc_val)

        if loss_val.item() < best_val_loss:
            best_val_loss = loss_val.item()
            torch.save(model.state_dict(), best_model_path)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch + 1}: "
                f"Train Loss={loss.item():.4f}, "
                f"Train Acc={acc_train:.4f}, "
                f"Val Loss={loss_val.item():.4f}, "
                f"Val Acc={acc_val:.4f}"
            )

        if torch.cuda.is_available() and epoch % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print("[OK] GCN training completed.")
    print(f"[OK] Best model saved to: {best_model_path}")

    training_curve_path = os.path.join(outdir, "training_curve.pdf")
    plot_training(
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        training_curve_path
    )

    print(f"[OK] Training curve saved to: {training_curve_path}")

    print("[4/7] Predicting cell types and evaluating GCN model...")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        logits = model(features, edge_index)
        prob = torch.softmax(logits, dim=1)

    eval_df, prauc_df = evaluate_model(prob, labels, val_idx, label_names)

    gcn_eval_path = os.path.join(outdir, "GCN_evaluation.csv")
    prauc_path = os.path.join(outdir, "GCN_per_celltype_PRAUC.csv")
    prob_path = os.path.join(outdir, "Cell_Celltype_probability.csv")

    eval_df.to_csv(gcn_eval_path, index=False)
    prauc_df.to_csv(prauc_path, index=False)

    prob_df = pd.DataFrame(prob.cpu().numpy(), columns=label_names)
    prob_df.to_csv(prob_path, index=False)

    print("[OK] Cell type prediction completed.")
    print(f"[OK] Cell prediction probability saved to: {prob_path}")
    print(f"[OK] GCN evaluation saved to: {gcn_eval_path}")
    print(f"[OK] Per-celltype PR-AUC saved to: {prauc_path}")

    print("[5/7] Running GNNExplainer...")

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=explainer_epochs),
        explanation_type="model",
        node_mask_type="attributes",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs"
        )
    )

    explanation = explainer(features, edge_index)

    explanation_path = os.path.join(outdir, "explanation.pt")
    torch.save(explanation.to_dict(), explanation_path)

    print("[OK] GNNExplainer completed.")
    print(f"[OK] Explanation object saved to: {explanation_path}")

    print("[6/7] Evaluating GNNExplainer explanation...")

    unfaith = unfaithfulness(explainer, explanation)

    gnn_eval_path = os.path.join(outdir, "GNNExplainer_evaluation.csv")

    pd.DataFrame({
        "metric": ["unfaithfulness"],
        "value": [float(unfaith)]
    }).to_csv(
        gnn_eval_path,
        index=False
    )

    print("[OK] GNNExplainer evaluation completed.")
    print(f"[OK] GNNExplainer unfaithfulness: {float(unfaith):.4f}")
    print(f"[OK] GNNExplainer evaluation saved to: {gnn_eval_path}")

    print("[7/7] Calculating SNV importance score in each Celltype...")

    snv_celltype_score = aggregate_snv_by_celltype(
        explanation.node_mask,
        labels,
        label_names,
        snv_names=snv_names
    )

    snv_score_path = os.path.join(outdir, "SNV_in_Celltype_score.csv")
    snv_celltype_score.to_csv(snv_score_path)

    print("[OK] SNV importance scoring completed.")
    print(f"[OK] SNV in Celltype score saved to: {snv_score_path}")
    print("[DONE] All analysis completed.")

    return {
        "model": model,
        "evaluation": eval_df,
        "per_celltype_prauc": prauc_df,
        "cell_celltype_probability": prob_df,
        "snv_celltype_score": snv_celltype_score,
        "unfaithfulness": float(unfaith)
    }
