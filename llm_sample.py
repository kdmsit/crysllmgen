"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
from pymatgen.core import Structure
from pymatgen.core.lattice import Lattice
from llm_finetune import get_crystal_string, MAX_LENGTH
from templating import make_swap_table
from data_utils import process_one
from tqdm import tqdm
from pymatgen.io.cif import CifWriter

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

from transformers import modeling_utils
if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none","colwise",'rowwise']

def parse_fn(gen_str):
    lines = [x for x in gen_str.split("\n") if len(x) > 0]
    lengths = [float(x) for x in lines[0].split(" ")]
    angles = [float(x) for x in lines[1].split(" ")]
    species = [x for x in lines[2::2]]
    coords = [[float(y) for y in x.split(" ")] for x in lines[3::2]]

    structure = Structure(
        lattice=Lattice.from_parameters(
            *(lengths + angles)),
        species=species,
        coords=coords,
        coords_are_cartesian=False,
    )

    return structure.to(fmt="cif")

def prepare_model_and_tokenizer(args):
    llama_options = args.model_name.split("-")
    is_chat = len(llama_options) == 2
    model_size = llama_options[0]

    def llama2_model_string(model_size, chat):
        chat = "chat-" if chat else ""
        return f"meta-llama/Llama-2-{model_size.lower()}-{chat}hf"

    model_string = llama2_model_string(model_size, is_chat)
    print("model string=>",model_string)
    model = LlamaForCausalLM.from_pretrained(
        model_string,
        load_in_8bit=True,
        device_map="auto",
    )

    tokenizer = LlamaTokenizer.from_pretrained(
        model_string,
        model_max_length=MAX_LENGTH,
        padding_side="right",
        use_fast=False,
    )

    model.eval()

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        llama_tokenizer=tokenizer,
        model=model,
    )

    model = PeftModel.from_pretrained(model, args.model_path, device_map="auto")

    return model, tokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    llama_tokenizer,
    model,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = llama_tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(llama_tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def unconditional_sample(args, model, tokenizer,prompts):
    batch_prompts = prompts[0:1]
    # print("Batch Prompt=>",batch_prompts)
    batch = tokenizer(list(batch_prompts), return_tensors="pt")
    batch = {k: v.cuda() for k, v in batch.items()}
    generate_ids = model.generate(**batch, do_sample=True, max_new_tokens=500, temperature=args.temperature, top_p=args.top_p)
    gen_strs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for gen_str, prompt in zip(gen_strs, batch_prompts):
        material_str = gen_str.replace(prompt, "")

        try:
            cif_str = parse_fn(material_str)
            frac_coords, atom_types, lengths, angles, num_atoms, edge_indices, to_jimages,data_dict = (
                process_one(cif_str, True, False, 'crystalnn', False, 0.01))
        except:
            return None, None, None, None, None, None

        num_atoms = torch.tensor([num_atoms])
        frac_coords = torch.tensor(frac_coords)
        lengths = torch.tensor(lengths)
        angles = torch.tensor(angles)
        atom_types = torch.tensor(atom_types)
        all_valid = ((atom_types >= 0) & (atom_types < 100)).all().item()
        if not all_valid:
            return None,None,None,None,None,None
        else:
            return frac_coords, atom_types,lengths,angles,num_atoms,data_dict




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--out_path", type=str, default="llm_samples.csv")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--format_instruction_prompt", type=int, default=0)
    parser.add_argument("--format_response_format", type=int, default=0)
    parser.add_argument("--conditions", type=str, default="pretty_formula")
    parser.add_argument("--conditions_file", type=str, default="") #"data/with_tags/test.csv"
    parser.add_argument("--infill_file", type=str, default="") #"data/with_tags/test.csv"
    parser.add_argument("--infill_do_constraint", type=int, default=0)
    parser.add_argument("--infill_constraint_tolerance", type=float, default=0.1)
    args = parser.parse_args()

    model, tokenizer = prepare_model_and_tokenizer(args)

    condition_templates = {
        "pretty_formula": "The chemical formula is {pretty_formula}. ",
        "e_above_hull": "The energy above the convex hull is {e_above_hull}. ",
        "spacegroup.number": "The spacegroup number is {spacegroup.number}. ",
    }

    n_atom, x_coord, a_type, length, angle = [], [], [], [], []
    all_data = []

    pbar = tqdm(total=args.num_samples, desc="Generating samples")
    sample_count = 0

    while sample_count < args.num_samples:
    # for sample_count in tqdm(range(args.num_samples), desc="Generating samples..."):
        prompts = []
        prompt = "Below is a description of a bulk material. "
        prompt += (
            "Generate a description of the lengths and angles of the lattice vectors "
            "and then the element type and coordinates for each atom within the lattice:\n"
        )
        prompts.append(prompt)

        frac_coords, atom_types, lengths, angles, num_atoms, data_dict = unconditional_sample(args, model, tokenizer,prompts)
        if atom_types is None:
            continue
        else:
            n_atom.append(num_atoms)
            x_coord.append(frac_coords)
            a_type.append(atom_types)
            length.append(lengths.view(1, 3))
            angle.append(angles.view(1, 3))
            all_data.append(data_dict)
            sample_count = sample_count + 1
            structure = Structure(lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
                                  species=atom_types,
                                  coords=frac_coords,
                                  coords_are_cartesian=False)
            # writer = CifWriter(structure)
            # writer.write_file("cif_files/"+str(sample_count)+".cif")
            pbar.update(1)

    print("Sampled Data size:", len(n_atom))

    n_atom = torch.cat(n_atom, dim=0)
    x_coord = torch.cat(x_coord, dim=0)
    a_type = torch.cat(a_type, dim=0)
    length = torch.cat(length, dim=0)
    angle = torch.cat(angle, dim=0)

    n_atom = n_atom.unsqueeze(0)
    x_coord = x_coord.unsqueeze(0)
    a_type = a_type.unsqueeze(0)
    length = length.unsqueeze(0)
    angle = angle.unsqueeze(0)

    path = os.path.join("llm_sample_" + args.dataset + "_" + str(args.num_samples) + ".pt")
    torch.save({
        "frac_coords": x_coord,
        "num_atoms": n_atom,
        "atom_types": a_type,
        "lengths": length,
        "angles": angle,
        "data_dict": all_data,
    }, path)
    print("Saved to file")



    # if args.conditions_file:
    #     conditional_sample(args)
    # else:
    #     unconditional_sample(args)
