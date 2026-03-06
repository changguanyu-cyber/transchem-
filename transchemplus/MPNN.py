import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from rdkit import Chem
from rdkit import RDLogger
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import add_self_loops
from torch.nn import GRUCell

RDLogger.DisableLog('rdApp.*')

# -----------------------------
# Atom feature (7-dim)
# -----------------------------
class MPNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')  # sum aggregation
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gru = GRUCell(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        m = self.propagate(edge_index, x=x)
        x = self.gru(m, x)
        return x

    def message(self, x_j):
        return self.msg_mlp(x_j)

class SimpleMPNN(nn.Module):
    def __init__(self, in_channels=24, hidden_channels=512, num_layers=6):
        super().__init__()

        self.node_encoder = nn.Linear(in_channels, hidden_channels)

        self.layers = nn.ModuleList([
            MPNNLayer(hidden_channels) for _ in range(num_layers)
        ])

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        x = self.node_encoder(x)

        for layer in self.layers:
            x = layer(x, edge_index)

        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)

def atom_features(atom: Chem.rdchem.Atom):
    # ---------- 1. Basic chemical properties ----------
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    total_h = atom.GetTotalNumHs()
    explicit_valence = atom.GetExplicitValence()
    total_valence = atom.GetTotalValence()
    degree = atom.GetDegree()
    hybridization = int(atom.GetHybridization())
    radical_e = atom.GetNumRadicalElectrons()
    aromatic = int(atom.GetIsAromatic())

    # ---------- 2. Ring information ----------
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

    # ---------- 3. Charge / volume / electronegativity ----------
    try:
        g_charge = atom.GetDoubleProp("_GasteigerCharge")
        g_charge = int(g_charge * 100)
    except:
        g_charge = 0
    pt = Chem.GetPeriodicTable()

    # Real element electronegativity (Pauling scale)
    try:
        electronegativity = pt.GetElectronegativity(atomic_num)
        if electronegativity is None:
            electronegativity = 0.0
    except:
        electronegativity = 0.0

    # Approximate volume using covalent radius
    try:
        rc = pt.GetRcovalent(atomic_num)
        volume = rc * rc * rc
    except:
        volume = 0.0

    h_donor = int(atom.GetAtomicNum() in [7,8] and total_h > 0)
    h_acceptor = int(atomic_num in [7,8])

    return [
        atomic_num, formal_charge, total_h, explicit_valence,
        total_valence, degree, hybridization, radical_e, aromatic,

        ring_member, ring_count, ring_3, ring_4, ring_5, ring_6, ring_7,
        chiral, neighbors, heavy_neighbors,

        g_charge, electronegativity, h_donor, h_acceptor, volume
    ]


# -----------------------------
# SMILES -> PyG Data
# -----------------------------
def smiles_to_data(smiles: str, y: float):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    x = torch.tensor(
        [atom_features(a) for a in mol.GetAtoms()],
        dtype=torch.float
    )

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([y], dtype=torch.float)
    )


# -----------------------------
# Dataset
# -----------------------------
class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, targets):
        self.data_list = []
        removed = 0

        for s, y in zip(smiles_list, targets):
            if y is None or np.isnan(y):
                removed += 1
                continue

            data = smiles_to_data(s, y)
            if data is None:
                removed += 1
                continue

            self.data_list.append(data)

        print(f"✓ Dataset size: {len(self.data_list)} | Removed: {removed}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# -----------------------------
# Simple GCN
# -----------------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=128, num_layers=6):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.head(x).squeeze(-1)


# -----------------------------
# Train / Eval
# -----------------------------
def train_epoch(model, loader, optimizer, device):
    model.train()
    total = 0.0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs

    return total / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    preds, ys = [], []
    total = 0.0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            preds.append(pred.cpu())
            ys.append(batch.y.view(-1).cpu())
            total += F.mse_loss(pred, batch.y.view(-1), reduction="sum").item()

    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    mse = total / len(ys)
    return mse, preds, ys


# -----------------------------
# Main
# -----------------------------
def main():
    csv_path = "/home/devuser/cgy/GCN/Uncertianty_quantification_Polymer_informatics-main/data/Gas_permeability_solubility_diffusivity_wide_final_filled.csv"
    result_csv = "/home/devuser/cgy/GCN/Uncertianty_quantification_Polymer_informatics-main/data/MPNN_results_all_properties_51216.csv"

    df = pd.read_csv(csv_path)
    smiles_all = df.iloc[:, 0].values.tolist()
    target_cols = df.columns[1:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = []

    for target in target_cols:
        print(f"\n========== {target} ==========")

        y_all = df[target].values.astype(np.float32)

        train_smi, test_smi, train_y, test_y = train_test_split(
            smiles_all, y_all, test_size=0.2, random_state=42
        )

        train_smi, val_smi, train_y, val_y = train_test_split(
            train_smi, train_y, test_size=0.1, random_state=42
        )

        train_loader = DataLoader(
            SmilesDataset(train_smi, train_y), batch_size=64, shuffle=True
        )
        val_loader = DataLoader(
            SmilesDataset(val_smi, val_y), batch_size=64
        )
        test_loader = DataLoader(
            SmilesDataset(test_smi, test_y), batch_size=64
        )

        model = SimpleMPNN(
            in_channels=24,
            hidden_channels=128,
            num_layers=6
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        best_val = float("inf")
        best_state = None

        for epoch in range(1, 301):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            val_mse, _, _ = evaluate(model, val_loader, device)

            if val_mse < best_val:
                best_val = val_mse
                best_state = model.state_dict()

            print(
                f"[{target}] Epoch {epoch:03d} | "
                f"Train={train_loss:.4f} | Val MSE={val_mse:.4f}"
            )

        model.load_state_dict(best_state)

        _, preds, ys = evaluate(model, test_loader, device)

        rmse = np.sqrt(mean_squared_error(ys, preds))
        r2 = r2_score(ys, preds)

        print(f"[{target}] RMSE={rmse:.4f} | R2={r2:.4f}")

        results.append({
            "property": target,
            "RMSE": rmse,
            "R2": r2
        })

    results_df = pd.DataFrame(results).set_index("property").T
    results_df.to_csv(result_csv)

    print("\n===== Final Results =====")
    print(results_df)
    print(f"\nSaved to: {result_csv}")


if __name__ == "__main__":
    main()