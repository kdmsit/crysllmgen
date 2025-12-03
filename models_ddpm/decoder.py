import torch
import torch.nn as nn
import torch.nn.functional as F
from models_ddpm.cspnet import CSPNet
from torch_scatter import scatter

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

MAX_ATOMIC_NUM = 100


class GemNetTDecoder(nn.Module):
    """Decoder with GemNetT."""

    def __init__(self):
        super(GemNetTDecoder, self).__init__()
        self.net = CSPNet(hidden_dim = 512,latent_dim = 0, max_atoms = 100, num_layers = 6, act_fn = 'silu', dis_emb = 'sin',
                              num_freqs = 128, edge_style = 'fc',max_neighbors= 20, cutoff = 7, ln = True, ip = True, smooth = True)
        self.fc_atom = nn.Linear(512, MAX_ATOMIC_NUM)

    def forward(self, time_emb, input_atom_types, input_frac_coords, input_lattice, batch):
        lattice_out, coord_out, h = self.net(time_emb, input_atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        atom_types_out = self.fc_atom(h)
        return lattice_out, coord_out, atom_types_out


