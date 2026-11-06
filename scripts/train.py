#!/usr/bin/env python
"""Train SILVA (v3 + detach z_c in H) on an h5ad with raw counts.

  python scripts/train.py --adata data/kang.h5ad --donor-key batch --condition-key condition --out runs/kang_s0

Outputs in --out: model.pt (best validation weights), emb.npz (z_c, z_d, cell_ids, genes), config.json, history.json.
"""
import argparse, json, sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import silva


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adata", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--donor-key", default="donor_id", help="obs column of the known batch (donor/sample)")
    p.add_argument("--condition-key", nargs="+", default=["condition"], help="obs column(s) of the specific condition factor(s)")
    p.add_argument("--counts-layer", default="counts", help="layer with raw counts ('' = use .X)")
    p.add_argument("--split-csv", type=Path, default=None, help="optional csv: cell_id,split (train/validation)")
    p.add_argument("--zc-dim", type=int, default=32)
    p.add_argument("--zd-dim", type=int, default=2)
    p.add_argument("--dec-hidden", type=int, default=256)
    p.add_argument("--reg-kl-comm", type=float, default=1e-5)
    p.add_argument("--reg-mmd-comm", type=float, default=1e-4)
    p.add_argument("--no-detach-zc-h", action="store_true", help="earlier 'v3' behaviour (H gradient reaches Enc_c)")
    p.add_argument("--method", choices=["silva", "scdisinfact"], default="silva", help="scdisinfact = baseline trained with the same loop and split")
    p.add_argument("--max-epochs", type=int, default=400)
    p.add_argument("--tf32", action="store_true", help="allow TF32 matmuls (faster, but unstable with several condition factors)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high" if args.tf32 else "highest")   # full fp32 by default: TF32 made multi-factor training diverge (K=3)
    a = ad.read_h5ad(args.adata)
    split = None
    if args.split_csv:
        split = pd.read_csv(args.split_csv, dtype=str).set_index("cell_id")["split"].reindex(a.obs_names)
    t = time.time()
    model, res, zc, zd = silva.fit(
        a, donor_key=args.donor_key, condition_key=(args.condition_key[0] if len(args.condition_key) == 1 else args.condition_key), counts_layer=args.counts_layer or None,
        zc_dim=args.zc_dim, zd_dim=args.zd_dim, dec_hidden=args.dec_hidden, reg_kl_comm=args.reg_kl_comm,
        reg_mmd_comm=args.reg_mmd_comm, detach_zc_in_H=not args.no_detach_zc_h, method=args.method, max_epochs=args.max_epochs,
        split=split, seed=args.seed, out=args.out)
    torch.save(res.pop("best_state"), args.out / "model.pt")
    zd_list = zd if isinstance(zd, list) else [zd]   # one array per condition factor
    np.savez_compressed(args.out / "emb.npz", z=zc, z_c=zc, z_d=np.hstack(zd_list), cell_ids=a.obs_names.to_numpy(dtype=str),
                        genes=a.var_names.to_numpy(dtype=str), **{f"z_d{k}": z for k, z in enumerate(zd_list)})
    (args.out / "config.json").write_text(json.dumps(dict(vars(args), **{k: v for k, v in res.items() if k != "history"},
                                                       seconds=time.time() - t, silva_version=silva.__version__), indent=2, default=str))
    print("done", args.out, "best_epoch", res["best_epoch"], "val_total %.4f" % res["best_val_total"])


if __name__ == "__main__":
    main()
