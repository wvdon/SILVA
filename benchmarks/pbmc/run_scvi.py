#!/usr/bin/env python
"""scVI baseline as in the scvi-tools tutorial: raw counts, Seurat-v3 HVG (default 2000, batch-aware), batch = donor.

  python benchmarks/pbmc/run_scvi.py --adata data/kang.h5ad --batch-key batch --out runs/scvi_s0 --n-hvg 2000
latent 32 (scVI default is 10); ZINB (scVI default); early stopping on the validation ELBO. Use --n-hvg 0 for all genes.
"""
import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import scanpy as sc
import scvi
import torch

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--adata", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
p.add_argument("--batch-key", default="batch"); p.add_argument("--n-hvg", type=int, default=2000)
p.add_argument("--n-latent", type=int, default=32); p.add_argument("--max-epochs", type=int, default=400); p.add_argument("--seed", type=int, default=0)
args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True); scvi.settings.seed = args.seed
a = ad.read_h5ad(args.adata)
if "counts" not in a.layers: a.layers["counts"] = a.X.copy()
if args.n_hvg > 0:
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=args.n_hvg, layer="counts", batch_key=args.batch_key, subset=True)
scvi.model.SCVI.setup_anndata(a, layer="counts", batch_key=args.batch_key)
m = scvi.model.SCVI(a, n_latent=args.n_latent)
m.train(max_epochs=args.max_epochs, early_stopping=True, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, enable_progress_bar=False)
z = m.get_latent_representation().astype(np.float32)
# HVG selection reorders nothing: cell order is unchanged
np.savez_compressed(args.out / "emb.npz", z_c=z, z=z, cell_ids=a.obs_names.to_numpy(dtype=str), genes=a.var_names.to_numpy(dtype=str))
print("done", args.out, z.shape, "genes", a.n_vars)
