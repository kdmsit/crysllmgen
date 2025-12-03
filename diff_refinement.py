import os
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from config import config
from torch_geometric.data import Batch
from models_ddpm.dataset import MaterialDataset,MaterialDispDataset
from torch_geometric.data import DataLoader
from models_ddpm.decoder import GemNetTDecoder
from models_ddpm.diffusion import CSPDiffusion
from torch.utils.data import Dataset
from torch_geometric.data import Data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

recommand_step_lr = {
    'csp':{
        "perov_5": 5e-7,
        "carbon_24": 5e-6,
        "mp_20": 1e-5,
        "mpts_52": 1e-5
    },
    'csp_multi':{
        "perov_5": 5e-7,
        "carbon_24": 5e-7,
        "mp_20": 1e-5,
        "mpts_52": 1e-5
    },
    'gen':{
        "perov_5": 1e-6,
        "carbon_24": 1e-5,
        "mp_20": 5e-6
    },
}

def lattices_to_params_shape(lattices):

    lengths = torch.sqrt(torch.sum(lattices ** 2, dim=-1))
    angles = torch.zeros_like(lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[...,i] = torch.clamp(torch.sum(lattices[...,j,:] * lattices[...,k,:], dim = -1) / (lengths[...,j] * lengths[...,k]), -1., 1.)
    angles = torch.arccos(angles) * 180.0 / np.pi

    return lengths, angles



def reconstructon(loader, model, num_evals, step_lr = 1e-5):
    """
    reconstruct the crystals in <loader>.
    """
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):
            print(f'batch {idx+1} / {len(loader)}, sample {eval_idx+1} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr=step_lr)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch)

def generation(loader, model, num_evals, step_lr = 1e-5, diff_steps = 1000):
    """
    reconstruct the crystals in <loader>.
    """
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            batch.cuda()
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):
            print(f'batch {idx+1} / {len(loader)}, sample {eval_idx+1} / {num_evals}')
            outputs, traj = model.sample(batch, step_lr=step_lr, diff_steps = diff_steps)
            batch_frac_coords.append(outputs['frac_coords'].detach().cpu())
            batch_num_atoms.append(outputs['num_atoms'].detach().cpu())
            batch_atom_types.append(outputs['atom_types'].detach().cpu())
            batch_lattices.append(outputs['lattices'].detach().cpu())

        frac_coords.append(torch.stack(batch_frac_coords, dim=0))
        num_atoms.append(torch.stack(batch_num_atoms, dim=0))
        atom_types.append(torch.stack(batch_atom_types, dim=0))
        lattices.append(torch.stack(batch_lattices, dim=0))

        # batch = batch.to(device)
        # input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=1)
    num_atoms = torch.cat(num_atoms, dim=1)
    atom_types = torch.cat(atom_types, dim=1)
    lattices = torch.cat(lattices, dim=1)
    lengths, angles = lattices_to_params_shape(lattices)
    # input_data_batch = Batch.from_data_list(input_data_list)

    # return (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch)
    return (frac_coords, atom_types, lattices, lengths, angles, num_atoms)


class SampleDataset(Dataset):

    def __init__(self, dataset,llm_file_name):
        super().__init__()
        # data = torch.load(f"gen/{dataset}/{llm_file_name}.pt")

        # data = torch.load(f"../gen/basic/new_llm_1_1_un_mp_1.0_0.7_1K.pt")
        # data = torch.load(f"../gen/basic/new_llm_1_1_un_mp_0.7_1.0_1K.pt")
        # data = torch.load(f"../gen/basic/new_llm_1_1_un_mp_0.7_0.7_1K.pt")

        # data = torch.load(f"../new_llm_1_1_un_perov_5_1.0_0.7_10000.pt")

        # data = torch.load(f"../gen/perov_5/new_llm_1_1_un_perov_0.7_1.0_10000.pt")
        # data = torch.load(f"../gen/perov_5/new_llm_1_1_un_perov_0.7_0.7_10000.pt")
        # data = torch.load(f"../gen/perov_5/new_llm_1_1_un_perov_1.0_0.7_10000.pt")
        # data = torch.load(f"../gen/perov_5/new_llm_1_1_un_perov_0.9_0.99_10000.pt")
        # data = torch.load(f"../gen/perov_5/new_llm_1_1_un_perov_0.9_0.9_10000.pt")

        # data = torch.load(f"../new_llm_1_1_un_mp_1.0_0.7_10000.pt")

        # data = torch.load(f"../gen/mpts_52/new_llm_1_1_un_mpts_0.7_0.7_10000.pt")
        # data = torch.load(f"../gen/mpts_52/new_llm_1_1_un_mpts_1.0_0.7_10000.pt")
        # data = torch.load(f"../gen/mpts_52/new_llm_1_1_un_mpts_0.9_0.9_10000.pt")
        # data = torch.load(f"../gen/mpts_52/new_llm_1_1_un_mpts_0.7_1.0_10000.pt")
        # data = torch.load(f"../gen/mpts_52/new_llm_1_1_un_mpts_0.9_0.99_10000.pt")

        data = torch.load(llm_file_name)
        self.frac_coords = data['frac_coords'][0]
        self.atom_types = data['atom_types'][0]
        self.lengths = data['lengths'][0]
        self.angles = data['angles'][0]
        self.num_atoms = data['num_atoms'][0]
        self.data_dict = data['data_dict']

        # print("frac_cooord=> ",self.frac_coords.size())
        # print("atom_types=> ",self.atom_types.size())
        # print("lengths=> ",self.lengths.size())
        # print("angles=> ",self.angles.size())
        # print("num_atoms=> ",self.num_atoms.size())
        # print("data_dict=> ", len(self.data_dict))

    def __len__(self) -> int:
        return self.lengths.size(0)

    def __getitem__(self, index):
        structure = self.data_dict[index]
        data = Data(
            num_atoms=torch.LongTensor([structure["n_atom"]]),
            num_nodes=structure["n_atom"],
            num_bonds=structure["edge_indices"].shape[0],
            lengths=structure["length"],
            angles=structure["angle"],
            frac_coords=torch.Tensor(structure["x_coord"]),
            atom_types=torch.LongTensor(structure["a_type"]),
            edge_index=torch.LongTensor(structure["edge_indices"].T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(structure["to_jimages"]),
        )
        return data

def main(args):
    model_path = Path(args.model_path,args.dataset)
    print("Tasks: ",args.tasks)

    test_set = SampleDataset(args.dataset, args.llm_file_name)
    test_dataloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    device = config.device
    if config.device is None or not torch.cuda.is_available():
        device = "cpu"
    chkpt_name = args.chkpt_name
    model = CSPDiffusion(args.timesteps,args.run_type).to(device)
    chkpt = torch.load(chkpt_name, map_location=device)
    model.load_state_dict(chkpt["model"])


    step_lr = args.step_lr if args.step_lr >= 0 else recommand_step_lr['csp' if args.num_evals == 1 else 'csp_multi']['perov_5']
    if torch.cuda.is_available():
        model.to('cuda')

    if 'recon' in args.tasks:
        print('Evaluate model on the reconstruction task.')
        start_time = time.time()
        (frac_coords,  atom_types,_, lengths, angles,num_atoms, input_data_batch) = (
            reconstructon(test_dataloader,model, args.num_evals, step_lr))
        print('Reconstruction Time :', time.time() - start_time)
        recon_out_name = 'eval_recon.pt'
        print('Saving Pt File..')
        print(model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save({'input_data_batch': input_data_batch,
            'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'time': time.time() - start_time
        }, model_path / recon_out_name)
        print('Saving Pt File..Done')

    if 'gen' in args.tasks:
        print('Evaluate model on the generation task.')
        start_time = time.time()
        (frac_coords,  atom_types,_, lengths, angles, num_atoms) = generation(test_dataloader, model, args.num_evals, step_lr, args.diff_steps)
        print('Generation Time :',time.time() - start_time)
        gen_out_name = 'eval_gen.pt'
        print('Saving Pt File..')
        print(model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save({'frac_coords': frac_coords,
            'num_atoms': num_atoms,
            'atom_types': atom_types,
            'lengths': lengths,
            'angles': angles,
            'time': time.time() - start_time
        }, model_path / gen_out_name)
        print('Saving Pt File..Done')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--tasks', nargs='+', default=['recon', 'gen', 'opt', 'disp'])
    parser.add_argument('--chkpt_name', required=True, type=str)
    parser.add_argument('--num_batches_to_samples', default=20, type=int)
    parser.add_argument('--num_to_samples', default=1, type=int)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--diff_steps', default=500, type=int)
    parser.add_argument('--step_lr', default=-1, type=float)
    parser.add_argument('--num_evals', default=1, type=int)
    parser.add_argument('--dataset', required=True, type=str, default='perov_5')
    parser.add_argument('--llm_file_name',  type=str, default='llm_7b') #required=True,
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--run-type', type=str, default='train')
    args = parser.parse_args()
    main(args)
    # main('gen/',"30112023","001402",'recon',8,4)