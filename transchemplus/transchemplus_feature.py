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
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

# -----------------------------
# Atom feature
# -----------------------------
def atom_features(atom: Chem.rdchem.Atom):
    atomic_num = atom.GetAtomicNum()
    formal_charge = atom.GetFormalCharge()
    total_h = atom.GetTotalNumHs()
    explicit_valence = atom.GetExplicitValence()
    total_valence = atom.GetTotalValence()
    degree = atom.GetDegree()
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
        g_charge = int(atom.GetDoubleProp("_GasteigerCharge") * 100)
    except:
        g_charge = 0

    pt = Chem.GetPeriodicTable()
    try:
        electronegativity = pt.GetElectronegativity(atomic_num) or 0.0
    except:
        electronegativity = 0.0

    try:
        rc = pt.GetRcovalent(atomic_num)
        volume = rc ** 3
    except:
        volume = 0.0

    h_donor = int(atomic_num in [7, 8] and total_h > 0)
    h_acceptor = int(atomic_num in [7, 8])

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

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return Data(
        x=x,
        edge_index=edge_index,
        y=torch.tensor([y], dtype=torch.float),
        smiles=smiles
    )


# -----------------------------
# Dataset
# -----------------------------
class SmilesDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, targets, atom_features_list=None):

        self.data_list = []
        for i, (s, y) in enumerate(zip(smiles_list, targets)):
            if y is None or np.isnan(y):
                continue

            data = smiles_to_data(s, y)
            if data is None:
                continue


            if atom_features_list is not None:
                atom_feat = atom_features_list[i]
                data.x_extra = torch.tensor(atom_feat, dtype=torch.float32)

            self.data_list.append(data)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# -----------------------------
# Simple GCN
# -----------------------------
class SimpleGCN(nn.Module):
    def __init__(self, in_channels=24, hidden_channels=512, num_layers=16):
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

    def forward(self, x, edge_index, batch, return_node_emb=False):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))

        node_emb = x
        graph_emb = global_mean_pool(x, batch)
        out = self.head(graph_emb).squeeze(-1)

        if return_node_emb:
            return out, node_emb
        return out


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
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.batch)
            preds.append(pred.cpu())
            ys.append(batch.y.view(-1).cpu())
    preds = torch.cat(preds).numpy()
    ys = torch.cat(ys).numpy()
    return preds, ys


# -----------------------------
# Save atom embeddings + SMILES
# -----------------------------
def save_atom_embeddings(model, loader, device, save_path):
    model.eval()
    all_data = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, node_emb = model(
                batch.x, batch.edge_index, batch.batch, return_node_emb=True
            )

            node_emb = node_emb.cpu()
            batch_idx = batch.batch.cpu()

            for i in range(batch.num_graphs):
                mask = batch_idx == i
                all_data.append({
                    "smiles": batch.smiles[i],
                    "atom_features": node_emb[mask]
                })

    torch.save(all_data, save_path)


# -----------------------------
# Main
# -----------------------------
def main():
    csv_path = "/home/devuser/cgy/GCN/Uncertianty_quantification_Polymer_informatics-main/data/Tm.csv"
    npz_path = "/home/devuser/cgy/chemprop-Mob/data/smiles_atom_features_Tm.npz"
    result_csv = "/home/devuser/cgy/GCN/Uncertianty_quantification_Polymer_informatics-main/data/GNN_results.csv"
    atom_feat_path = "/home/devuser/cgy/GCN/Uncertianty_quantification_Polymer_informatics-main/data/atom_features_with_smiles.pt"


    df = pd.read_csv(csv_path)
    target = df.columns[1]
    y_all = df[target].values.astype(np.float32)

    data_npz = np.load(npz_path, allow_pickle=True)
    smiles_all = data_npz["smiles"].tolist()
    atom_features_all = data_npz["atom_features"]

    train_smi, test_smi, train_y, test_y = train_test_split(
        smiles_all, y_all, test_size=0.2, random_state=42
    )

    train_loader = DataLoader(
        SmilesDataset(train_smi, train_y, atom_features_all), batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        SmilesDataset(test_smi, test_y, atom_features_all), batch_size=64
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, 301):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch:03d} | Train MSE={train_loss:.4f}")

    preds, ys = evaluate(model, test_loader, device)
    rmse = np.sqrt(mean_squared_error(ys, preds))
    r2 = r2_score(ys, preds)
    print(r2)
    print(rmse)
    pd.DataFrame({"RMSE": [rmse], "R2": [r2]}).to_csv(result_csv, index=False)





if __name__ == "__main__":
    main()
