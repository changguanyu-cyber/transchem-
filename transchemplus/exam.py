from rdkit import Chem
from rdkit.Chem import AllChem
import torch
smiles = "*OC(F)(F)OC(*)(F)F"
smiles = smiles.replace('[*]', 'C')

mol = Chem.MolFromSmiles(smiles)

AllChem.ComputeGasteigerCharges(mol)

charges = []
for atom in mol.GetAtoms():
    charge = atom.GetProp("_GasteigerCharge")
    charges.append(float(charge))
pooled = torch.mean(torch.tensor(charges))
print(pooled)
print("SMILES:", smiles)
for i, atom in enumerate(mol.GetAtoms()):
    print(f"Atom {i:2d} ({atom.GetSymbol():2}) : {charges[i]: .4f}")
