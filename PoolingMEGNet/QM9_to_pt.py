from megnet.data.molecule import MolecularGraph
from openbabel.pybel import readstring
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch
import numpy as np
import tqdm

QM9_path = r"C:\NMarom\dataset\QM9"
property_list = ["A", "B", "C", "mu", "alpha", "EHOMO", "ELUMO", "gap", "R^2", "zpve", "U0", "U", "H", "G", "C_v"]

def read_QM9_xyz(graph_converter = MolecularGraph(), global_features = torch.zeros(1, 2), prop = "gap"):
    data_list = []
    for idx in tqdm.trange(1, 133886):
        mol = f"dsgdb9nsd_{idx:06}.xyz"
        with open(os.path.join(QM9_path, mol), 'r') as f:
            mol = f.readlines()
        y = float(mol[1].split()[2:][property_list.index(prop)])
        mol = ''.join(mol)
        try:
            MEGNet_feature = graph_converter.convert(readstring("xyz", mol))
        except Exception as e:
            print(f"An error occured: {e}")
            continue
        data = Data(
            x = torch.tensor(MEGNet_feature["atom"], dtype = torch.float), 
            edge_index = torch.tensor((MEGNet_feature["index1"], MEGNet_feature["index2"]), dtype = torch.long), 
            edge_attr = torch.tensor(MEGNet_feature["bond"]),
            global_features = global_features,
            y = torch.tensor(y))
        if np.isnan(data.x).any() or np.isinf(data.x).any():
            print("x conversion issue:", data.x)
            continue
        if np.isnan(data.edge_index).any() or np.isinf(data.edge_index).any():
            print("x conversion issue:", data.edge_index)
            continue
        if np.isnan(data.edge_attr).any() or np.isinf(data.edge_attr).any():
            print("x conversion issue:", data.edge_attr)
            continue
        if np.isnan(data.global_features).any() or np.isinf(data.global_features).any():
            print("x conversion issue:", data.global_features)
            continue
        if np.isnan(data.y).any() or np.isinf(data.y).any():
            print("x conversion issue:", data.y)
            continue
        data_list.append(data)
    torch.save(data_list, "QM9MEGNet.pt")
    return DataLoader(data_list)

if __name__ == "__main__":
    read_QM9_xyz()


