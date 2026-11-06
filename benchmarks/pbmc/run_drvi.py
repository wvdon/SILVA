#!/usr/bin/env python
"""DRVI with the settings of the official tutorial (theislab/DRVI_tutorials, general_pipeline.ipynb):
raw counts, 2000 batch-aware highly variable genes (seurat_v3, selected beforehand in the tutorial), `batch_key` = donor,
n_latent=128, encoder_dims=decoder_dims=[128,128], 400 epochs, early_stopping=False, n_epochs_kl_warmup=400.
  python benchmarks/pbmc/run_drvi.py --adata data/kang.h5ad --batch-key batch --out runs/drvi_s0 --seed 0
Embedding = DRVI posterior mean (128-d)."""
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import scanpy as sc
import scvi
import torch
from drvi.model import DRVI

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--adata", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
p.add_argument("--batch-key", default="batch"); p.add_argument("--n-hvg", type=int, default=2000)
p.add_argument("--n-latent", type=int, default=128); p.add_argument("--epochs", type=int, default=400); p.add_argument("--seed", type=int, default=0)
args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True); scvi.settings.seed = args.seed
a = ad.read_h5ad(args.adata)
if "counts" not in a.layers: a.layers["counts"] = a.X.copy()
a.obs[args.batch_key] = a.obs[args.batch_key].astype("category")
if args.n_hvg > 0:
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=args.n_hvg, layer="counts", batch_key=args.batch_key, subset=True)
DRVI.setup_anndata(a, layer="counts", batch_key=args.batch_key, is_count_data=True)
m = DRVI(a, n_latent=args.n_latent, encoder_dims=[128, 128], decoder_dims=[128, 128])
m.train(max_epochs=args.epochs, early_stopping=False, early_stopping_patience=20, plan_kwargs={"n_epochs_kl_warmup": args.epochs},
        accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, enable_progress_bar=False)
z = m.get_latent_representation().astype(np.float32)
np.savez_compressed(args.out / "emb.npz", z_c=z, z=z, cell_ids=a.obs_names.to_numpy(dtype=str), genes=a.var_names.to_numpy(dtype=str))
print("done", args.out, z.shape, "genes", a.n_vars)
