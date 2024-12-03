# Yiqun Luo (luo2@andrew.cmu.edu)
 
import numpy as np
import pandas as pd
from openbabel.pybel import readstring
from megnet.data.molecule import MolecularGraph
from megnet.data.crystal import CrystalGraph
from torch_geometric.data import Data
import torch

OE62_PATH = "../data/df_62k.json"
split_file = ""

def OE62_to_pt(molecules: np.ndarray, gaps: np.ndarray, file_name: str = "OE62MEGNet.pt"):
    assert len(molecules) == len(gaps)

    data_list = []
    for mol, gap in zip(molecules, gaps):
        try:
            MEGNet_feature = graph_converter.convert(mol)
        except Exception as e:
            print(e)
            continue
        data = Data(
            x = torch.tensor(MEGNet_feature["atom"], dtype = torch.float), 
            edge_index = torch.tensor((MEGNet_feature["index1"], MEGNet_feature["index2"]), dtype = torch.long), 
            edge_attr = torch.tensor(MEGNet_feature["bond"]),
            global_features = global_features,
            y = torch.tensor(gap))
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
    torch.save(data_list, file_name)

    return


if __name__ == "__main__":
    # graph_converter = MolecularGraph()
    graph_converter = CrystalGraph()
    global_features = torch.zeros(1, 2)

    # Load data
    df_62k = pd.read_json(OE62_PATH, orient = "split")

    # Create pymatgen Molecules
    molecules = np.array(df_62k["xyz_pbe_relaxed"].apply(lambda x: readstring("xyz", x)))

    # Calculate Gap
    occupied_levels = df_62k['energies_occ_pbe']
    unoccupied_levels = df_62k['energies_unocc_pbe']
    homos = occupied_levels.apply(lambda x: x[-1])
    lumos = unoccupied_levels.apply(lambda x: x[0])
    gaps = np.array(lumos - homos)

    OE62_to_pt(molecules, gaps)