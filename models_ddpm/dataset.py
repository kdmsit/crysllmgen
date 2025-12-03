import hydra
import torch
import omegaconf
import numpy as np
import pandas as pd
from omegaconf import ValueNode
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Batch
from .data_utils import get_scaler_from_data_list
from .data_utils import (preprocess, add_scaled_lattice_prop)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MaterialDispDataset(Dataset):
    def __init__(self, path, num_samples=1):
        super().__init__()
        self.df = pd.read_csv(path)
        self.rows = self.df.shape[0]
        self.n_samples = num_samples

    def __len__(self):
        return self.rows * self.n_samples

    def __getitem__(self, index):
        row = index // self.n_samples
        sample = index % self.n_samples

        df_row = self.df.iloc[row]
        # print(df_row)
        num_atoms = df_row[1]
        mat_id = df_row[0]
        text = df_row[2]
        data = Data(
            mat_id=mat_id,
            sam_id=sample,
            num_atoms=torch.tensor(num_atoms),
            text=text,
            num_nodes=num_atoms,
        )
        return data
class MaterialDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.df = pd.read_csv(path)
        self.prop = 'heat_ref'
        self.niggli = True
        self.primitive = False
        self.graph_method = 'crystalnn'
        self.lattice_scale_method = 'scale_length'
        self.preprocess_workers = 30
        self.cached_data = preprocess(self.path,self.preprocess_workers,niggli=self.niggli, primitive=self.primitive, graph_method=self.graph_method, prop_list=[self.prop])
        add_scaled_lattice_prop(self.cached_data, self.lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        (frac_coords, atom_types, lengths, angles, edge_indices,to_jimages, num_atoms) = data_dict['graph_arrays']

        one_hot = np.zeros((len(atom_types), 100))
        for i in range(len(atom_types)):
            one_hot[i][atom_types[i]-1]=1

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            atom_types_one_hot=torch.Tensor(one_hot),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometricy=prop.view(1, -1),
        )
        return data

class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


def main(cfg: omegaconf.DictConfig):
    dataset: CrystDataset = hydra.utils.instantiate(cfg.data.datamodule.datasets.train, _recursive_=False)
    lattice_scaler = get_scaler_from_data_list(dataset.cached_data,key='scaled_lattice')
    scaler = get_scaler_from_data_list(dataset.cached_data,key=dataset.prop)
    dataset.lattice_scaler = lattice_scaler
    dataset.scaler = scaler
    data_list = [dataset[i] for i in range(len(dataset))]
    batch = Batch.from_data_list(data_list)
    return batch


if __name__ == "__main__":
    main()