# ========== 依赖 ==========
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from rdkit import Chem
from typing import List, Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
# -----------------------------
# Scatter fallback
# -----------------------------
try:
    from torch_scatter import scatter_add
    _HAS_SCATTER = True
except Exception:
    _HAS_SCATTER = False
    def scatter_add(src, index, dim=0, dim_size=None):
        if dim != 0:
            raise NotImplementedError("scatter_add fallback only supports dim=0")
        if dim_size is None:
            dim_size = int(index.max().item()) + 1
        out = src.new_zeros((dim_size, src.size(1)))
        return out.index_add_(0, index, src)
import torch
import torch.nn.functional as F

def contrastive_loss(z_ae, z_as, tau=0.1):
    """
    z_ae : [M, d]
    z_as : [M, d]
    """

    # cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        z_ae.unsqueeze(1),
        z_as.unsqueeze(0),
        dim=2
    ) / tau

    # AE -> AS
    log_prob_ae = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    loss_ae = -log_prob_ae.diag()

    # AS -> AE
    sim_matrix_T = sim_matrix.T
    log_prob_as = sim_matrix_T - torch.logsumexp(sim_matrix_T, dim=1, keepdim=True)
    loss_as = -log_prob_as.diag()

    loss = 0.5 * (loss_ae + loss_as)

    return loss.mean()
# -----------------------------
# MyGCNConv
# -----------------------------
class TransChemGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, att_scale=1.0):
        super().__init__()
        self.lin_bond = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_vin = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_vconn = nn.Linear(in_channels, out_channels, bias=False)
        self.att_scale = att_scale
        self.att_proj = nn.Parameter(torch.randn(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x,
                edge_index_bond=None,
                edge_index_virtual_intra=None,
                edge_index_virtual_conn=None):
        device = x.device
        N = x.size(0)
        x_bond = self.lin_bond(x)
        x_vin = self.lin_vin(x)
        x_vconn = self.lin_vconn(x)
        out = torch.zeros_like(x_bond, device=device)

        # bond edges
        if edge_index_bond is not None and edge_index_bond.size(1) > 0:
            edge_bond, _ = add_self_loops(edge_index_bond, num_nodes=N)
            row, col = edge_bond
            deg = degree(row, N, dtype=x_bond.dtype)
            deg = deg.to(device)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            messages = x_bond[col] * norm.unsqueeze(-1)
            out_bond = scatter_add(messages, row, dim=0, dim_size=N) if _HAS_SCATTER else out.index_add_(0, row, messages)
            out = out + out_bond

        # virtual_intra
        if edge_index_virtual_intra is not None and edge_index_virtual_intra.size(1) > 0:
            row_v, col_v = edge_index_virtual_intra
            messages_vin = x_vin[col_v]
            out_vin_sum = scatter_add(messages_vin, row_v, dim=0, dim_size=N) if _HAS_SCATTER else out.index_add_(0, row_v, messages_vin)
            deg_vin = degree(row_v, N, dtype=x_vin.dtype).unsqueeze(-1).clamp(min=1.0)
            deg_vin = deg_vin.to(device)
            out = out + out_vin_sum / deg_vin

        # virtual_conn
        if edge_index_virtual_conn is not None and edge_index_virtual_conn.size(1) > 0:
            row_c, col_c = edge_index_virtual_conn
            proj_i = x_vconn[row_c] * self.att_proj.unsqueeze(0)
            proj_j = x_vconn[col_c] * self.att_proj.unsqueeze(0)
            e_ij = (proj_i * proj_j).sum(dim=-1) * self.att_scale
            alpha = torch.sigmoid(e_ij).unsqueeze(-1)
            messages_vc = alpha * x_vconn[col_c]
            out_vc = scatter_add(messages_vc, row_c, dim=0, dim_size=N) if _HAS_SCATTER else out.index_add_(0, row_c, messages_vc)
            out = out + out_vc

        out = out + self.bias
        return out

# -----------------------------
# SMILES -> Data
# -----------------------------
def atom_features(atom: Chem.rdchem.Atom):
    return [
        atom.GetAtomicNum(),
        int(atom.GetChiralTag()),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
        1 if atom.GetIsAromatic() else 0,
    ]

import random, math
from typing import Optional
import torch
from torch_geometric.data import Data
from rdkit import Chem

def sample_virtual_edges(candidate_pairs, sigma=0.7, keep_minimum=1):
    sampled = []
    for (a, b) in candidate_pairs:
        # Gaussian random variable N(0, σ)
        x = random.gauss(0, sigma)
        # Convert to probability score
        p = math.exp(-(x ** 2) / (2 * sigma * sigma))
        # Bernoulli sample
        if random.random() < p:
            sampled.append((a, b))

    # ensure at least one edge if available
    if len(sampled) < keep_minimum and len(candidate_pairs) > 0:
        sampled.append(random.choice(candidate_pairs))

    return sampled


def smiles_to_data(smiles: str, y: Optional[float] = None, pool_type="mean") -> Data:
    mol = Chem.MolFromSmiles(smiles)

    # ----- Existing Atom feature extraction -----
    N = mol.GetNumAtoms()
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    # Normal bond edges
    row, col, bond_set = [], [], set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [i, j]
        col += [j, i]
        bond_set.update([(i, j), (j, i)])

    edge_index_bond = torch.tensor([row, col], dtype=torch.long) if len(row) > 0 else torch.empty((2, 0), dtype=torch.long)

    # ----------- Virtual Intra-Fragment Edges ----------
    vin_candidates = []
    for frag in Chem.GetMolFrags(mol):
        frag = list(frag)
        for i_idx in range(len(frag)):
            for j_idx in range(i_idx + 1, len(frag)):
                a, b = frag[i_idx], frag[j_idx]
                if (a, b) in bond_set:
                    continue
                vin_candidates.append((a, b))

    sampled_edges = sample_virtual_edges(vin_candidates, sigma=0.7)

    vin_row, vin_col = [], []
    for a, b in sampled_edges:
        vin_row += [a, b]
        vin_col += [b, a]

    edge_index_virtual_intra = torch.tensor([vin_row, vin_col], dtype=torch.long) if len(vin_row) > 0 else torch.empty((2, 0), dtype=torch.long)

    # ----------- Virtual connection edges -------------
    conn_atoms = list({nbr.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0 for nbr in atom.GetNeighbors()})
    vcr_row, vcr_col = [], []
    for i in range(len(conn_atoms)):
        for j in range(i + 1, len(conn_atoms)):
            a, b = conn_atoms[i], conn_atoms[j]
            vcr_row += [a, b, b, a]
            vcr_col += [b, a, a, b]

    edge_index_virtual_conn = torch.tensor([vcr_row, vcr_col], dtype=torch.long) if len(vcr_row) > 0 else torch.empty((2, 0), dtype=torch.long)

    # Construct Data
    data = Data(
        x=x,
        edge_index_bond=edge_index_bond,
        edge_index_virtual_intra=edge_index_virtual_intra,
        edge_index_virtual_conn=edge_index_virtual_conn,
    )

    if y is not None:
        data.y = torch.tensor(y, dtype=torch.float)

    data.smiles = smiles
    return data


def load_smiles_and_targets(
        csv_path,
        smiles_column="Smiles",
        target_column="TmValue",
):
    import pandas as pd
    import torch
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # ---------- 读取 CSV ----------
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_column, target_column]).reset_index(drop=True)

    filtered_smiles = []
    filtered_targets = []

    # ---------- Filtering ----------
    for smi, tgt in zip(df[smiles_column], df[target_column]):
        if "[R" in smi:
            continue
        smi = smi.replace("*", "[*]")
        filtered_smiles.append(smi)
        filtered_targets.append(tgt)


    # ---------- Step 2: 计算 charge 并池化 ----------
    padded_charge_vectors = []     # 每原子 padded vector
    pooled_charge_targets = []     # 分子级自监督伪标签

    for raw_smi in filtered_smiles:

        smi_for_charge = raw_smi.replace("[*]", "C")
        mol = Chem.MolFromSmiles(smi_for_charge)

        try:
            AllChem.ComputeGasteigerCharges(mol)

            charges = []
            for atom in mol.GetAtoms():
                val = atom.GetProp("_GasteigerCharge")
                try:
                    val = float(val)
                except:
                    val = 0.0
                charges.append(val)

            charge_tensor = torch.tensor(charges, dtype=torch.float)

        except:
            charge_tensor = torch.zeros(mol.GetNumAtoms(), dtype=torch.float)


        # --------  分子级池化生成自监督标签  --------
        charge_pool = charge_tensor.mean().item()  # 可换成 max / sum / abs().mean()
        pooled_charge_targets.append(charge_pool)

    return (
        filtered_smiles,
        filtered_targets,
        pooled_charge_targets
    )




# -----------------------------
# Dataset
# -----------------------------
from torch.utils.data import Dataset


class SMILESDataset(Dataset):
    def __init__(self, smiles_list, targets=None):
        self.nan_count = 0  # 统计删除数量

        if targets is not None:
            clean_smiles = []
            clean_targets = []

            for s, y in zip(smiles_list, targets):
                y_tensor = torch.tensor(y, dtype=torch.float32)

                if torch.isnan(y_tensor).any():
                    self.nan_count += 1
                    continue  # 略过此样本

                clean_smiles.append(s)
                clean_targets.append(y)

            self.smiles = clean_smiles
            self.targets = clean_targets

            if self.nan_count > 0:
                print(f"⚠ Removed {self.nan_count} samples with NaN targets")
            else:
                print("✓ No NaN values found in target")

        else:
            self.smiles = smiles_list
            self.targets = None
            print("✓ Dataset without targets (inference mode)")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        y = None if self.targets is None else self.targets[idx]
        return smiles_to_data(self.smiles[idx], y)


# -----------------------------
# SimpleGNN
# -----------------------------
class SimpleGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int=128, num_layers:int=3, return_embedding=False):
        super().__init__()
        self.return_embedding = return_embedding
        self.convs = nn.ModuleList([TransChemGCNConv(in_channels if i==0 else hidden_channels, hidden_channels) for i in range(num_layers)])
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels//2),
            nn.ReLU(),
            nn.Linear(hidden_channels//2, 1)
        )
    def forward(self, x, batch, edge_index_bond, edge_index_virtual_intra, edge_index_virtual_conn):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index_bond, edge_index_virtual_intra, edge_index_virtual_conn))
        x = global_mean_pool(x, batch)
        if self.return_embedding:
            return x  # graph embedding
        else:
            return self.lin(x).squeeze(-1)  # prediction

# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.x, batch.batch,
                      batch.edge_index_bond,
                      batch.edge_index_virtual_intra,
                      batch.edge_index_virtual_conn)
        preds = preds.view(-1)
        target = batch.y.view(-1)# flatten to [32000]
        loss = contrastive_loss(preds, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    preds_list, y_list = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.batch,
                          batch.edge_index_bond,
                          batch.edge_index_virtual_intra,
                          batch.edge_index_virtual_conn)
            preds_list.append(preds.cpu())
            y_list.append(batch.y.view(-1).cpu())
            preds = preds.view(-1)
            total_loss += F.mse_loss(preds, batch.y.view(-1), reduction='sum').item()
    preds_all = torch.cat(preds_list).numpy()
    y_all = torch.cat(y_list).numpy()
    return total_loss/len(loader.dataset), preds_all, y_all

# -----------------------------
# Main
# -----------------------------
#model_path = "/root/autodl-tmp/GCN/best_gnn_model_pretrain7.pth"
csv_path = ""
smiles_list, _, targets = load_smiles_and_targets(csv_path, smiles_column="SMILES", target_column="label")
train_smi, test_smi, train_y, test_y= train_test_split(
    smiles_list, targets, test_size=0.2)

train_smi, val_smi, train_y, val_y= train_test_split(
    train_smi, train_y, test_size=0.1)

train_loader = DataLoader(SMILESDataset(train_smi, train_y), batch_size=64, shuffle=True)
val_loader   = DataLoader(SMILESDataset(val_smi, val_y), batch_size=64)
test_loader  = DataLoader(SMILESDataset(test_smi, test_y), batch_size=64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_dim = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))
model = SimpleGNN(in_dim, hidden_channels=512, num_layers=16, return_embedding=False).to(device)
#model.load_state_dict(torch.load(model_path))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# training
best_val = float("inf")
best_state = None
save_path = ""

# ---------- Training ----------
for epoch in range(1, 11):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_mse, _, _ = evaluate(model, val_loader, device)

    if val_mse < best_val:
        best_val = val_mse
        best_state = model.state_dict()
        torch.save(best_state, save_path)   # 保存最佳模型

    print(f"Epoch {epoch:03d} | Train Loss={train_loss:.4f} | Val MSE={val_mse:.4f}")
# ---------- Load Best Model ----------
embedding_model = SimpleGNN(in_dim, hidden_channels=512, num_layers=16, return_embedding=False).to(device)
embedding_model.load_state_dict(torch.load(save_path))
embedding_model.to(device)
embedding_model.eval()

# ---------- Evaluate on Test ----------
test_mse, preds, ys = evaluate(embedding_model, test_loader, device)

preds_np = preds if isinstance(preds, np.ndarray) else preds.detach().cpu().numpy()
ys_np = ys if isinstance(ys, np.ndarray) else ys.detach().cpu().numpy()

r2 = r2_score(ys_np, preds_np)

print("====== Final Performance (Best Model) ======")
print("Test MSE:", test_mse)
print("Test R2 :", r2)