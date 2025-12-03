import math
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from models_ddpm.decoder import GemNetTDecoder
from models_ddpm.cspnet import CSPNet
from models_ddpm.diff_utils import BetaScheduler,SigmaScheduler
from models_ddpm.diff_utils import d_log_p_wrapped_normal
from models_ddpm.data_utils import lattice_params_to_matrix_torch
MAX_ATOMIC_NUM=100


### Model definition

class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CSPDiffusion(nn.Module):
    def __init__(self,timesteps, run_type) -> None:
        super(CSPDiffusion, self).__init__()
        self.decoder =CSPNet(hidden_dim = 512,max_atoms = 100,
                             num_layers = 6, act_fn = 'silu', dis_emb = 'sin',
                              num_freqs = 128, edge_style = 'fc',max_neighbors= 20,
                             cutoff = 7, run_type = run_type, ln = True, ip = True, smooth = False,pred_type = False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta_scheduler = BetaScheduler(timesteps=timesteps,scheduler_mode = 'cosine')
        self.sigma_scheduler = SigmaScheduler(timesteps=timesteps,sigma_begin = 0.005, sigma_end = 0.5)
        self.time_dim = 256
        self.time_embedding = SinusoidalTimeEmbeddings(self.time_dim)
        self.keep_lattice = False
        self.keep_coords = False
        self.optim = optim.Adam(self.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode='min', factor=0.6, patience=30, threshold=0.0001)

    def forward(self, batch):

        batch_size = batch.num_graphs
        times = self.beta_scheduler.uniform_sample_t(batch_size, self.device)
        time_emb = self.time_embedding(times)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        sigmas = self.sigma_scheduler.sigmas[times]
        sigmas_norm = self.sigma_scheduler.sigmas_norm[times]

        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        frac_coords = batch.frac_coords

        rand_l, rand_x = torch.randn_like(lattices), torch.randn_like(frac_coords)

        input_lattice = c0[:, None, None] * lattices + c1[:, None, None] * rand_l
        sigmas_per_atom = sigmas.repeat_interleave(batch.num_atoms)[:, None]
        sigmas_norm_per_atom = sigmas_norm.repeat_interleave(batch.num_atoms)[:, None]
        input_frac_coords = (frac_coords + sigmas_per_atom * rand_x) % 1.

        if self.keep_coords:
            input_frac_coords = frac_coords

        if self.keep_lattice:
            input_lattice = lattices

        pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch.num_atoms, batch.batch)
        # pred_l, pred_x = self.decoder(time_emb, batch.atom_types, input_frac_coords, input_lattice, batch)
        tar_x = d_log_p_wrapped_normal(sigmas_per_atom * rand_x, sigmas_per_atom) / torch.sqrt(sigmas_norm_per_atom)

        loss_lattice = F.mse_loss(pred_l, rand_l)
        loss_coord = F.mse_loss(pred_x, tar_x)

        loss = loss_lattice +loss_coord

        return loss, loss_lattice, loss_coord

    @torch.no_grad()
    def sample(self, batch, step_lr = 1e-5,diff_steps = 1000):
        batch_size = batch.num_graphs

        # CrysLLMGen
        x_T= batch.frac_coords
        l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        # CSP Task
        # l_T, x_T = torch.randn([batch_size, 3, 3]).to(self.device), torch.rand([batch.num_nodes, 3]).to(self.device)

        if self.keep_coords:
            x_T = batch.frac_coords

        if self.keep_lattice:
            l_T = lattice_params_to_matrix_torch(batch.lengths, batch.angles)

        # time_start = self.beta_scheduler.timesteps
        time_start = diff_steps

        traj = {time_start : {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'frac_coords' : x_T % 1.,
            'lattices' : l_T
        }}


        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device = self.device)

            time_emb = self.time_embedding(times)
            
            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]
            sigma_x = self.sigma_scheduler.sigmas[t]
            sigma_norm = self.sigma_scheduler.sigmas_norm[t]


            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            x_t = traj[t]['frac_coords']
            l_t = traj[t]['lattices']

            if self.keep_coords:
                x_t = x_T

            if self.keep_lattice:
                l_t = l_T

            # PC-sampling refers to "Score-Based Generative Modeling through Stochastic Differential Equations"
            # Origin code : https://github.com/yang-song/score_sde/blob/main/sampling.py

            # Corrector
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            step_size = step_lr * (sigma_x / self.sigma_scheduler.sigma_begin) ** 2
            std_x = torch.sqrt(2 * step_size)

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t, l_t, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_05 = l_t if not self.keep_lattice else l_t

            # Predictor
            rand_l = torch.randn_like(l_T) if t > 1 else torch.zeros_like(l_T)
            rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

            adjacent_sigma_x = self.sigma_scheduler.sigmas[t-1] 
            step_size = (sigma_x ** 2 - adjacent_sigma_x ** 2)
            std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))   

            pred_l, pred_x = self.decoder(time_emb, batch.atom_types, x_t_minus_05, l_t_minus_05, batch.num_atoms, batch.batch)

            pred_x = pred_x * torch.sqrt(sigma_norm)

            x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x if not self.keep_coords else x_t

            l_t_minus_1 = c0 * (l_t_minus_05 - c1 * pred_l) + sigmas * rand_l if not self.keep_lattice else l_t


            traj[t - 1] = {
                'num_atoms' : batch.num_atoms,
                'atom_types' : batch.atom_types,
                'frac_coords' : x_t_minus_1 % 1.,
                'lattices' : l_t_minus_1              
            }

        traj_stack = {
            'num_atoms' : batch.num_atoms,
            'atom_types' : batch.atom_types,
            'all_frac_coords' : torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
            'all_lattices' : torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)])
        }

        return traj[0], traj_stack

    