import torch
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import os
from config import config

def configure_save_path(dataset):
    now_time = datetime.now()
    date_str = now_time.strftime("%d%m%Y")
    time_str = now_time.strftime("%H%M%S")

    save_dir = "out"
    save_dir_dataset = os.path.join(save_dir, dataset)
    save_dir_date = os.path.join(save_dir_dataset, date_str)
    save_dir_time = os.path.join(save_dir_date, time_str)

    if not os.path.exists(save_dir): os.mkdir(save_dir)
    if not os.path.exists(save_dir_dataset): os.mkdir(save_dir_dataset)
    if not os.path.exists(save_dir_date): os.mkdir(save_dir_date)
    if not os.path.exists(save_dir_time): os.mkdir(save_dir_time)

    return save_dir_time

def plot_losses(path, coord_losses, type_losses, L_losses, kld_losses, num_atom_losses):
    np.savez_compressed(path+".npz", coord_losses=coord_losses, type_losses=type_losses,
                        kld_losses=kld_losses, num_atom_losses=num_atom_losses)
    
    plt.clf()
    plt.figure(figsize=(12, 8), dpi=90)
    plt.subplot(3, 3, 1)
    plt.plot(L_losses[1:])
    plt.xlabel("Epoch")
    plt.ylabel("Lattice loss")
    plt.grid(True)

    plt.subplot(3, 3, 2)
    plt.plot(kld_losses[1:])
    plt.xlabel("Epoch")
    plt.ylabel("KL divergence")
    plt.grid(True)
    # plt.title("VAE KL divergence loss")

    plt.subplot(3, 3, 4)
    plt.plot(num_atom_losses[1:])
    plt.xlabel("Epoch")
    plt.ylabel("Num Atom loss")
    plt.grid(True)
    # plt.title("Diffusion neg log likelihood loss")

    plt.subplot(3, 3, 5)
    plt.plot(coord_losses[1:])
    plt.xlabel("Epoch")
    plt.ylabel("Coordinate loss")
    plt.grid(True)

    plt.subplot(3, 3, 7)
    plt.plot(type_losses[1:])
    plt.xlabel("Epoch")
    plt.ylabel("Atom Type loss")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path+".png")


def save_model(model, path):
    torch.save({
        "model": model.state_dict(),
        "model_opt": model.optim.state_dict(),
        }, path)

def delete_model(path):
    os.remove(path)

def load_model(model, vae_model, path):
    ckpt = torch.load(path)
    model.load_state_dict(ckpt["dif"])
    model.optim.load_state_dict(ckpt["dif_opt"])
    vae_model.load_state_dict(ckpt["vae"])
    vae_model.optim.load_state_dict(ckpt["vae_opt"])
