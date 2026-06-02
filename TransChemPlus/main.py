# ========== 依赖 ==========
import math
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.EState import EStateIndices
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, degree

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


def contrastive_loss(z_ae, z_as, tau=0.1):
    """
    z_ae : [B, d]
    z_as : [B, d]
    """
    sim_matrix = F.cosine_similarity(
        z_ae.unsqueeze(1),
        z_as.unsqueeze(0),
        dim=2,
    ) / tau

    log_prob_ae = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    loss_ae = -log_prob_ae.diag()

    sim_matrix_t = sim_matrix.T
    log_prob_as = sim_matrix_t - torch.logsumexp(sim_matrix_t, dim=1, keepdim=True)
    loss_as = -log_prob_as.diag()

    return 0.5 * (loss_ae + loss_as).mean()


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

    def forward(
        self,
        x,
        edge_index_bond=None,
        edge_index_virtual_intra=None,
        edge_index_virtual_conn=None,
    ):
        device = x.device
        n_nodes = x.size(0)

        x_bond = self.lin_bond(x)
        x_vin = self.lin_vin(x)
        x_vconn = self.lin_vconn(x)
        out = torch.zeros_like(x_bond, device=device)

        if edge_index_bond is not None and edge_index_bond.size(1) > 0:
            edge_bond, _ = add_self_loops(edge_index_bond, num_nodes=n_nodes)
            row, col = edge_bond
            deg = degree(row, n_nodes, dtype=x_bond.dtype).to(device)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            messages = x_bond[col] * norm.unsqueeze(-1)
            out_bond = (
                scatter_add(messages, row, dim=0, dim_size=n_nodes)
                if _HAS_SCATTER
                else torch.zeros_like(out).index_add_(0, row, messages)
            )
            out = out + out_bond

        if edge_index_virtual_intra is not None and edge_index_virtual_intra.size(1) > 0:
            row_v, col_v = edge_index_virtual_intra
            messages_vin = x_vin[col_v]
            out_vin_sum = (
                scatter_add(messages_vin, row_v, dim=0, dim_size=n_nodes)
                if _HAS_SCATTER
                else torch.zeros_like(out).index_add_(0, row_v, messages_vin)
            )
            deg_vin = degree(row_v, n_nodes, dtype=x_vin.dtype).unsqueeze(-1).clamp(min=1.0).to(device)
            out = out + out_vin_sum / deg_vin

        if edge_index_virtual_conn is not None and edge_index_virtual_conn.size(1) > 0:
            row_c, col_c = edge_index_virtual_conn
            proj_i = x_vconn[row_c] * self.att_proj.unsqueeze(0)
            proj_j = x_vconn[col_c] * self.att_proj.unsqueeze(0)
            e_ij = (proj_i * proj_j).sum(dim=-1) * self.att_scale
            alpha = torch.sigmoid(e_ij).unsqueeze(-1)
            messages_vc = alpha * x_vconn[col_c]
            out_vc = (
                scatter_add(messages_vc, row_c, dim=0, dim_size=n_nodes)
                if _HAS_SCATTER
                else torch.zeros_like(out).index_add_(0, row_c, messages_vc)
            )
            out = out + out_vc

        out = out + self.bias
        return out


# -----------------------------
# SMILES -> Data
# -----------------------------
def atom_features(atom: Chem.rdchem.Atom):
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    total_h = atom.GetTotalNumHs()
    explicit_valence = atom.GetValence(Chem.ValenceType.EXPLICIT)
    implicit_valence = atom.GetValence(Chem.ValenceType.IMPLICIT)
    total_valence = atom.GetTotalValence()
    atom_degree = atom.GetDegree()
    hybridization = int(atom.GetHybridization())
    radical_e = atom.GetNumRadicalElectrons()
    aromatic = int(atom.GetIsAromatic())

    ring_member = int(atom.IsInRing())
    ring_count = sum(atom.IsInRingSize(i) for i in range(3, 8))
    ring_3 = int(atom.IsInRingSize(3))
    ring_4 = int(atom.IsInRingSize(4))
    ring_5 = int(atom.IsInRingSize(5))
    ring_6 = int(atom.IsInRingSize(6))
    ring_7 = int(atom.IsInRingSize(7))
    chiral = int(atom.GetChiralTag())
    neighbors = len(atom.GetNeighbors())
    heavy_neighbors = sum(1 for a in atom.GetNeighbors() if a.GetAtomicNum() > 1)

    try:
        g_charge = atom.GetDoubleProp("_GasteigerCharge")
        g_charge = int(g_charge * 100)
    except Exception:
        g_charge = 0

    pt = Chem.GetPeriodicTable()

    try:
        electronegativity = pt.GetElectronegativity(atomic_num)
        if electronegativity is None:
            electronegativity = 0.0
    except Exception:
        electronegativity = 0.0

    try:
        rc = pt.GetRcovalent(atomic_num)
        volume = rc * rc * rc
    except Exception:
        volume = 0.0

    h_donor = int(atomic_num in [7, 8] and total_h > 0)
    h_acceptor = int(atomic_num in [7, 8])

    return [
        atomic_num,
        formal_charge,
        total_h,
        implicit_valence,
        explicit_valence,
        total_valence,
        atom_degree,
        hybridization,
        radical_e,
        aromatic,
        ring_member,
        ring_count,
        ring_3,
        ring_4,
        ring_5,
        ring_6,
        ring_7,
        chiral,
        neighbors,
        heavy_neighbors,
        g_charge,
        electronegativity,
        h_donor,
        h_acceptor,
        volume,
    ]


def sample_virtual_edges(candidate_pairs, sigma=0.7, keep_minimum=1):
    sampled = []
    for a, b in candidate_pairs:
        x = random.gauss(0, sigma)
        p = math.exp(-(x ** 2) / (2 * sigma * sigma))
        if random.random() < p:
            sampled.append((a, b))

    if len(sampled) < keep_minimum and len(candidate_pairs) > 0:
        sampled.append(random.choice(candidate_pairs))

    return sampled


def smiles_to_data(smiles: str, y: Optional[list[float]] = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid smiles: {smiles}")

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    row, col, bond_set = [], [], set()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [i, j]
        col += [j, i]
        bond_set.update([(i, j), (j, i)])

    edge_index_bond = (
        torch.tensor([row, col], dtype=torch.long)
        if len(row) > 0
        else torch.empty((2, 0), dtype=torch.long)
    )

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

    edge_index_virtual_intra = (
        torch.tensor([vin_row, vin_col], dtype=torch.long)
        if len(vin_row) > 0
        else torch.empty((2, 0), dtype=torch.long)
    )

    conn_atoms = list(
        {
            nbr.GetIdx()
            for atom in mol.GetAtoms()
            if atom.GetAtomicNum() == 0
            for nbr in atom.GetNeighbors()
        }
    )
    vcr_row, vcr_col = [], []
    for i in range(len(conn_atoms)):
        for j in range(i + 1, len(conn_atoms)):
            a, b = conn_atoms[i], conn_atoms[j]
            vcr_row += [a, b, b, a]
            vcr_col += [b, a, a, b]

    edge_index_virtual_conn = (
        torch.tensor([vcr_row, vcr_col], dtype=torch.long)
        if len(vcr_row) > 0
        else torch.empty((2, 0), dtype=torch.long)
    )

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
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_column, target_column]).reset_index(drop=True)

    filtered_smiles = []
    filtered_targets = []
    pooled_targets = []

    invalid_smiles_count = 0
    charge_fail_count = 0
    estate_fail_count = 0

    for smi, tgt in zip(df[smiles_column], df[target_column]):
        smi = str(smi).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles_count += 1
            continue

        charge_vals = [0.0] * mol.GetNumAtoms()
        try:
            AllChem.ComputeGasteigerCharges(mol)
            tmp = []
            for atom in mol.GetAtoms():
                try:
                    val = float(atom.GetProp("_GasteigerCharge"))
                    if not np.isfinite(val):
                        val = 0.0
                except Exception:
                    val = 0.0
                tmp.append(val)
            if len(tmp) == mol.GetNumAtoms():
                charge_vals = tmp
        except Exception:
            charge_fail_count += 1

        estate_vals = [0.0] * mol.GetNumAtoms()
        try:
            tmp = list(EStateIndices(mol))
            if len(tmp) == mol.GetNumAtoms():
                estate_vals = [float(v) if np.isfinite(v) else 0.0 for v in tmp]
        except Exception:
            estate_fail_count += 1

        hyb_vals = []
        for atom in mol.GetAtoms():
            hyb = atom.GetHybridization()
            if hyb.name == "SP":
                hyb_vals.append(1.0)
            elif hyb.name == "SP2":
                hyb_vals.append(2.0)
            elif hyb.name == "SP3":
                hyb_vals.append(3.0)
            else:
                hyb_vals.append(0.0)

        charge_tensor = torch.tensor(charge_vals, dtype=torch.float)
        estate_tensor = torch.tensor(estate_vals, dtype=torch.float)
        hyb_tensor = torch.tensor(hyb_vals, dtype=torch.float)

        charge_pool = charge_tensor.abs().mean().item() if charge_tensor.numel() > 0 else 0.0
        estate_pool = estate_tensor.abs().mean().item() if estate_tensor.numel() > 0 else 0.0
        hyb_pool = hyb_tensor.mean().item() if hyb_tensor.numel() > 0 else 0.0

        pooled_val = [charge_pool, estate_pool, hyb_pool]

        filtered_smiles.append(smi)
        filtered_targets.append(float(tgt))
        pooled_targets.append(pooled_val)

    print(f"raw rows after dropna: {len(df)}")
    print(f"kept rows: {len(filtered_smiles)}")
    print(f"invalid smiles dropped: {invalid_smiles_count}")
    print(f"charge fallback count: {charge_fail_count}")
    print(f"estate fallback count: {estate_fail_count}")

    return filtered_smiles, filtered_targets, pooled_targets


# -----------------------------
# Dataset
# -----------------------------
class SMILESDataset(Dataset):
    def __init__(self, smiles_list, targets=None):
        self.nan_count = 0

        if targets is not None:
            clean_smiles = []
            clean_targets = []

            for s, y in zip(smiles_list, targets):
                y_tensor = torch.tensor(y, dtype=torch.float32)
                if torch.isnan(y_tensor).any():
                    self.nan_count += 1
                    continue
                clean_smiles.append(s)
                clean_targets.append(y)

            self.smiles = clean_smiles
            self.targets = clean_targets

            if self.nan_count > 0:
                print(f"Removed {self.nan_count} samples with NaN targets")
            else:
                print("No NaN values found in target")
        else:
            self.smiles = smiles_list
            self.targets = None
            print("Dataset without targets")

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        y = None if self.targets is None else self.targets[idx]
        return smiles_to_data(self.smiles[idx], y)


# -----------------------------
# SimpleGNN
# -----------------------------
class SimpleGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 128, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                TransChemGCNConv(in_channels if i == 0 else hidden_channels, hidden_channels)
                for i in range(num_layers)
            ]
        )
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 16),
        )

    def forward(self, x, batch, edge_index_bond, edge_index_virtual_intra, edge_index_virtual_conn):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index_bond, edge_index_virtual_intra, edge_index_virtual_conn))
        x = global_mean_pool(x, batch)
        return self.lin(x)  # [B, 16]


# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, optimizer, device, proj_target):
    model.train()
    proj_target.train()
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        preds = model(
            batch.x,
            batch.batch,
            batch.edge_index_bond,
            batch.edge_index_virtual_intra,
            batch.edge_index_virtual_conn,
        )  # [B, 16]

        target = batch.y.view(-1, 3)      # [B, 3]
        target_proj = proj_target(target) # [B, 16]

        loss = contrastive_loss(preds, target_proj)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device, proj_target):
    model.eval()
    proj_target.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            preds = model(
                batch.x,
                batch.batch,
                batch.edge_index_bond,
                batch.edge_index_virtual_intra,
                batch.edge_index_virtual_conn,
            )

            target = batch.y.view(-1, 3)
            target_proj = proj_target(target)

            loss = contrastive_loss(preds, target_proj)
            total_loss += loss.item() * batch.num_graphs

    return total_loss / len(loader.dataset)


# -----------------------------
# Main
# -----------------------------
model_path = "/home/devuser/cgy/GCN/GCN/best_gnn_model_pretrain.pth"
csv_path = "/home/devuser/cgy/GCN/GCN/predicted_tg.csv"
save_path = "/home/devuser/cgy/GCN/GCN/pretrain_contrastive_3targets.pth"

smiles_list, _, targets = load_smiles_and_targets(
    csv_path,
    smiles_column="New_SMILES",
    target_column="Tg_predicted",
)

assert len(smiles_list) > 0, "No valid samples after preprocessing"
assert len(smiles_list) == len(targets), "smiles_list and targets size mismatch"

train_smi, test_smi, train_y, test_y = train_test_split(
    smiles_list,
    targets,
    test_size=0.2,
    random_state=42,
)

train_smi, val_smi, train_y, val_y = train_test_split(
    train_smi,
    train_y,
    test_size=0.1,
    random_state=42,
)

print("train size:", len(train_smi))
print("val size:", len(val_smi))
print("test size:", len(test_smi))

assert len(train_smi) > 0, "train_smi is empty"
assert len(val_smi) > 0, "val_smi is empty"
assert len(test_smi) > 0, "test_smi is empty"

train_loader = DataLoader(SMILESDataset(train_smi, train_y), batch_size=64, shuffle=True)
val_loader = DataLoader(SMILESDataset(val_smi, val_y), batch_size=64, shuffle=False)
test_loader = DataLoader(SMILESDataset(test_smi, test_y), batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
in_dim = len(atom_features(Chem.MolFromSmiles("C").GetAtomWithIdx(0)))

model = SimpleGNN(in_dim, hidden_channels=512, num_layers=16).to(device)
proj_target = nn.Linear(3, 16).to(device)

if model_path:
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    filtered_checkpoint = {
        k: v for k, v in checkpoint.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model_dict.update(filtered_checkpoint)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded {len(filtered_checkpoint)} matching parameters from checkpoint")

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(proj_target.parameters()),
    lr=1e-3,
)

best_val = float("inf")

for epoch in range(1, 11):
    train_loss = train_epoch(model, train_loader, optimizer, device, proj_target)
    val_loss = evaluate(model, val_loader, device, proj_target)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(
            {
                "model": model.state_dict(),
                "proj_target": proj_target.state_dict(),
            },
            save_path,
        )

    print(f"Epoch {epoch:03d} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f}")

print("====== Final Performance (Best Model) ======")
ckpt = torch.load(save_path, map_location=device)
model.load_state_dict(ckpt["model"])
proj_target.load_state_dict(ckpt["proj_target"])
test_loss = evaluate(model, test_loader, device, proj_target)
print(f"Test Loss={test_loss:.4f}")
