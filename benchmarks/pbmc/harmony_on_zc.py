#!/usr/bin/env python
"""Optional post-hoc Harmony on a frozen shared latent z_c (harmonypy defaults, batch = donor). Reported separately as
"+ post-hoc Harmony"; it is NOT part of SILVA and changes neither the decoder nor the attribution.
  python benchmarks/pbmc/harmony_on_zc.py --adata data/kang.h5ad --emb runs/s0/emb.npz --batch-key batch --out runs/s0_harmony/emb.npz"""
import argparse
from pathlib import Path
import anndata as ad, harmonypy, numpy as np, pandas as pd
p = argparse.ArgumentParser(); p.add_argument("--adata", type=Path, required=True); p.add_argument("--emb", type=Path, required=True)
p.add_argument("--batch-key", default="batch"); p.add_argument("--out", type=Path, required=True); p.add_argument("--seed", type=int, default=0)
args = p.parse_args(); obs = ad.read_h5ad(args.adata, backed="r").obs
with np.load(args.emb, allow_pickle=True) as f: ids, genes, zc = f["cell_ids"].astype(str), f["genes"], f["z_c"]
ho = harmonypy.run_harmony(zc, obs.loc[ids, [args.batch_key]].reset_index(drop=True), args.batch_key, random_state=args.seed)
z = np.asarray(ho.Z_corr, dtype=np.float32); z = z if z.shape == zc.shape else z.T
args.out.parent.mkdir(parents=True, exist_ok=True); np.savez(args.out, z=z, z_c=z, cell_ids=ids, genes=genes); print("done", args.out, z.shape)
