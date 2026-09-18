#
# For licensing see accompanying LICENSE file.
# Copyright (c) 2025 Apple Inc. Licensed under MIT License.
#

import os
import torch
import hydra
import omegaconf
import argparse
import numpy as np
from copy import deepcopy
from pathlib import Path
from itertools import starmap
import lightning.pytorch as pl

from model.flow import LinearPath
from model.torch.sampler import EMSampler

from processor.protein_processor import ProteinDataProcessor
from utils.datamodule_utils import process_one_inference_structure
from utils.esm_utils import _af2_to_esm, esm_registry
from utils.boltz_utils import process_structure, save_structure
from utils.fasta_utils import process_fastas, download_fasta_utilities, check_fasta_inputs
from boltz_data_pipeline.feature.featurizer import BoltzFeaturizer
from boltz_data_pipeline.tokenize.boltz_protein import BoltzTokenizer
from utils.device_utils import get_torch_device  # also pins MLX to exact fp32 (MLX_ENABLE_TF32=0)
from utils.io_utils import get_config_path, download_file

try: 
    import mlx.core as mx
    from mlx.utils import tree_unflatten, tree_flatten
    from model.mlx.sampler import EMSampler as EMSamplerMLX
    from model.mlx.esm_network import ESM2 as ESM2MLX
    from utils.mlx_utils import map_torch_to_mlx, map_plddt_torch_to_mlx, load_mlx_state_dict
    MLX_AVAILABLE = True
except Exception as e:  # noqa: BLE001 - report why, instead of silently falling back to torch
    MLX_AVAILABLE = False
    print(f"MLX not installed, skip importing MLX related packages. ({type(e).__name__}: {e})")


ckpt_url_dict = {
    "simplefold_100M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_100M.ckpt",
    "simplefold_360M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_360M.ckpt",
    "simplefold_700M": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_700M.ckpt",
    "simplefold_1.1B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.1B.ckpt",
    "simplefold_1.6B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_1.6B.ckpt",
    "simplefold_3B": "https://ml-site.cdn-apple.com/models/simplefold/simplefold_3B.ckpt",
}

plddt_ckpt_url = "https://ml-site.cdn-apple.com/models/simplefold/plddt_module_1.6B.ckpt"


def initialize_folding_model(args):
    # define folding model
    simplefold_model = args.simplefold_model

    # create checkpoint directory
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")

    # create folding model
    ckpt_path = os.path.join(ckpt_dir, f"{simplefold_model}.ckpt")
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_dir, exist_ok=True)
        download_file(ckpt_url_dict[simplefold_model], ckpt_path)
    cfg_path = get_config_path(f"configs/model/architecture/foldingdit_{simplefold_model[11:]}.yaml")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # load model checkpoint
    if args.backend == 'torch':
        device = get_torch_device(getattr(args, "device", "auto"))
        model_config = omegaconf.OmegaConf.load(cfg_path)
        model = hydra.utils.instantiate(model_config)
        model.load_state_dict(checkpoint, strict=True)
        model = model.to(device)
    elif args.backend == 'mlx':
        device = "cpu"
        # replace torch implementations with mlx
        with open(cfg_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        model_config = omegaconf.OmegaConf.create(yaml_str)
        model = hydra.utils.instantiate(model_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, checkpoint.items()) if k is not None}
        load_mlx_state_dict(model, mlx_state_dict)
    print(f"Folding model {simplefold_model} loaded.")
    print(f"Using device: {device}.")

    model.eval()
    return model, device


def initialize_plddt_module(args, device):
    if not args.plddt:
        return None, None

    # load pLDDT module if specified
    plddt_ckpt_path = os.path.join(args.ckpt_dir, "plddt.ckpt")
    if not os.path.exists(plddt_ckpt_path):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        download_file(plddt_ckpt_url, plddt_ckpt_path)

    plddt_module_path = get_config_path("configs/model/architecture/plddt_module.yaml")
    plddt_checkpoint = torch.load(plddt_ckpt_path, map_location="cpu", weights_only=True)

    if args.backend == "torch":
        plddt_config = omegaconf.OmegaConf.load(plddt_module_path)
        plddt_out_module = hydra.utils.instantiate(plddt_config)
        plddt_out_module.load_state_dict(plddt_checkpoint, strict=True)
        plddt_out_module = plddt_out_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_module_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_out_module = hydra.utils.instantiate(plddt_config)

        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_plddt_torch_to_mlx, plddt_checkpoint.items()) if k is not None}
        load_mlx_state_dict(plddt_out_module, mlx_state_dict)

    plddt_out_module.eval()
    print(f"pLDDT output module loaded with {args.backend} backend.")

    plddt_latent_ckpt_path = os.path.join(args.ckpt_dir, "simplefold_1.6B.ckpt")
    if not os.path.exists(plddt_latent_ckpt_path):
        os.makedirs(args.ckpt_dir, exist_ok=True)
        download_file(ckpt_url_dict["simplefold_1.6B"], plddt_latent_ckpt_path)

    plddt_latent_config_path = get_config_path("configs/model/architecture/foldingdit_1.6B.yaml")
    plddt_latent_checkpoint = torch.load(plddt_latent_ckpt_path, map_location="cpu", weights_only=True)

    if args.backend == "torch":
        plddt_latent_config = omegaconf.OmegaConf.load(plddt_latent_config_path)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        plddt_latent_module.load_state_dict(plddt_latent_checkpoint, strict=True)
        plddt_latent_module = plddt_latent_module.to(device)
    elif args.backend == "mlx":
        # replace torch implementations with mlx
        with open(plddt_latent_config_path, "r") as f:
            yaml_str = f.read()
        yaml_str = yaml_str.replace('torch', 'mlx')

        plddt_latent_config = omegaconf.OmegaConf.create(yaml_str)
        plddt_latent_module = hydra.utils.instantiate(plddt_latent_config)
        mlx_state_dict = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, plddt_latent_checkpoint.items()) if k is not None}
        load_mlx_state_dict(plddt_latent_module, mlx_state_dict)

    plddt_latent_module.eval()
    print(f"pLDDT latent module loaded with {args.backend} backend.")

    return plddt_latent_module, plddt_out_module


def initialize_esm_model(args, device):
    # load ESM2 model
    esm_model, esm_dict = esm_registry["esm2_3B"]()
    af2_to_esm = _af2_to_esm(esm_dict)

    if args.backend == 'torch':
        esm_model = esm_model.to(device)
        af2_to_esm = af2_to_esm.to(device)
    elif args.backend == 'mlx':
        esm_model_mlx = ESM2MLX(num_layers=36, embed_dim=2560, attention_heads=40)
        esm_state_dict_torch = esm_model.cpu().state_dict()

        esm_state_dict_torch = {k: mx.array(v) for k, v in starmap(map_torch_to_mlx, esm_state_dict_torch.items()) if k is not None}
        load_mlx_state_dict(esm_model_mlx, esm_state_dict_torch)
        esm_model = esm_model_mlx
    print(f"pLM ESM-3B loaded with {args.backend} backend.")

    esm_model.eval()
    return esm_model, esm_dict, af2_to_esm


def initialize_others(args, device):
    # prepare data tokenizer, featurizer, and processor
    tokenizer = BoltzTokenizer()
    featurizer = BoltzFeaturizer()
    processor = ProteinDataProcessor(
        device=device,
        scale=16.0, 
        ref_scale=5.0, 
        multiplicity=1,
        inference_multiplicity=args.nsample_per_protein,
        backend=args.backend,
    )

    # define flow process and sampler
    flow = LinearPath()

    if args.backend == "torch":
        sampler_cls = EMSampler
    elif args.backend == "mlx":
        sampler_cls = EMSamplerMLX

    sampler = sampler_cls(
        num_timesteps=args.num_steps,
        t_start=1e-4,
        tau=args.tau,
        log_timesteps=True,
        w_cutoff=0.99,
    )
    return tokenizer, featurizer, processor, flow, sampler


def generate_structure(
    args, batch, sampler, flow, processor,
    model, plddt_latent_module, plddt_out_module, device
):
    # run inference for target protein
    if args.backend == "torch":
        # draw on the CPU generator so a given seed yields the same noise on cpu/mps/cuda
        noise = torch.randn(batch['coords'].shape, dtype=batch['coords'].dtype).to(device)
    elif args.backend == "mlx":
        noise = mx.random.normal(batch['coords'].shape)
    out_dict = sampler.sample(model, flow, noise, batch)

    if args.plddt:
        if args.backend == "torch":
            t = torch.ones(batch['coords'].shape[0], device=device)
            # use unscaled coords to extract latent for pLDDT prediction
            out_feat = plddt_latent_module(
                out_dict["denoised_coords"].detach(), t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"].detach(),
                batch,
            )
        elif args.backend == "mlx":
            t = mx.ones(batch['coords'].shape[0])
            # use unscaled coords to extract latent for pLDDT prediction
            out_feat = plddt_latent_module(
                out_dict["denoised_coords"], t, batch)
            plddt_out_dict = plddt_out_module(
                out_feat["latent"],
                batch,
            )
        # scale pLDDT to [0, 100]
        plddts = plddt_out_dict["plddt"] * 100.0
        if args.backend == "torch":
            # the writers call .item() per residue; one transfer instead of one sync per residue
            plddts = plddts.detach().cpu()
        else:
            # same for MLX: materialize once instead of one lazy slice eval per residue
            plddts = np.array(plddts)
    else:
        plddts = None

    out_dict = processor.postprocess(out_dict, batch)
    # sampled_coord = out_dict['denoised_coords'].detach()
    if args.backend == "torch":
        sampled_coord = out_dict['denoised_coords'].detach()
    else:
        sampled_coord = out_dict['denoised_coords']

    pad_mask = batch['atom_pad_mask']
    return sampled_coord, pad_mask, plddts


def predict_structures_from_fastas(args):
    # create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir = output_dir / f"predictions_{args.simplefold_model}"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    cache = output_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    # set random seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    if args.backend == "mlx" and not MLX_AVAILABLE:
        args.backend = "torch"
        print("MLX not available, switch to torch backend.")

    # initialize models
    model, device = initialize_folding_model(args)
    plddt_latent_module, plddt_out_module = initialize_plddt_module(args, device)
    esm_model, esm_dict, af2_to_esm = initialize_esm_model(args, device)

    # initialize other components
    tokenizer, featurizer, processor, flow, sampler = initialize_others(args, device)

    if args.backend == "mlx":
        # pl.seed_everything seeds python/numpy/torch only; MLX has its own generator. Seed it after all
        # models are built so the seed -> noise mapping does not depend on which models were loaded.
        mx.random.seed(args.seed)

    # process fasta files to input format
    download_fasta_utilities(cache)
    data = check_fasta_inputs(Path(args.fasta_path))
    if not data:
        raise ValueError("No valid input files found. Please check the input directory.")
    process_fastas(
        data=data,
        out_dir=output_dir,
        ccd_path=cache / "ccd.pkl",
    )

    for struct_file in output_dir.glob("structures/*.npz"):
        record_file = output_dir / "records" / f"{struct_file.stem}.json"

        # prepare the target protein data for inference
        batch, structure, record = process_one_inference_structure(
            struct_file, record_file,
            tokenizer, featurizer, processor,
            esm_model, esm_dict, af2_to_esm,
        )

        sampled_coord, pad_mask, plddts = generate_structure(
            args, batch, sampler, flow, processor,
            model, plddt_latent_module, plddt_out_module, device
        )

        for i in range(args.nsample_per_protein):
            sampled_coord_i = sampled_coord[i]
            pad_mask_i = pad_mask[i]

            # save the generated structure
            structure_save = process_structure(
                deepcopy(structure), sampled_coord_i, pad_mask_i, record, backend=args.backend
            )
            outname = f"{record.id}_sampled_{i}"
            save_structure(
                structure_save, prediction_dir, outname,
                output_format=args.output_format,
                plddts=plddts[i] if plddts is not None else None
            )
