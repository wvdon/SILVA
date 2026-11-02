"""Prepare the Kang 2018 PBMC IFN-beta benchmark (GSE96583) into the SILVA canonical
format: raw integer counts + obs keys condition / batch / cell_type.

Source: pertpy figshare. Download via the ndownloader.figshare.com host (follows the
302 to S3); the figshare.com/ndownloader form returns 202/0-bytes.

Mapping (raw -> canonical):  label -> condition (ctrl/stim);  replicate -> batch
(8 donors);  cell_type -> cell_type (eval only).  Genes filtered to those expressed in
>= min_cells cells (all genes kept otherwise; NO HVG — SILVA uses the full transcriptome).

Run: python benchmarks/pbmc/prepare_kang.py   (-> data/kang.h5ad)
Output: data/kang.h5ad  (raw counts in .layers["counts"] and .X; obs: condition, batch = donor, cell_type)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import urllib.request
import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent.parent
RAW_URL = "https://ndownloader.figshare.com/files/34464122"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "kang_2018_raw.h5ad"))
    ap.add_argument("--out", default=str(ROOT / "data" / "kang.h5ad"))
    ap.add_argument("--min-cells", type=int, default=3)
    args = ap.parse_args()
    raw = Path(args.raw); raw.parent.mkdir(parents=True, exist_ok=True)

    if not raw.exists():
        print(f"downloading Kang raw -> {raw}")
        req = urllib.request.Request(RAW_URL, headers={"User-Agent": "silva"})
        with urllib.request.urlopen(req, timeout=600) as r, open(raw, "wb") as f:
            f.write(r.read())
    a = ad.read_h5ad(raw)
    print(f"raw: {a.shape}; obs: {list(a.obs.columns)}")

    # canonical obs keys (with fallbacks across pertpy/GEO variants)
    def pick(cands):
        for c in cands:
            if c in a.obs:
                return c
        raise KeyError(f"none of {cands} in obs")
    a.obs["condition"] = a.obs[pick(["label", "condition", "stim"])].astype(str).values
    a.obs["batch"] = a.obs[pick(["replicate", "ind", "batch", "donor"])].astype(str).values
    a.obs["cell_type"] = a.obs[pick(["cell_type", "cell", "celltype"])].astype(str).values

    # raw integer counts -> .layers['counts'] (+ .X); NB needs integers
    X = a.layers["counts"] if "counts" in a.layers else a.X
    X = X if sp.issparse(X) else sp.csr_matrix(X)
    assert np.allclose(X.data, np.round(X.data)), "X is not raw integer counts"
    a.X = X; a.layers["counts"] = X.copy()

    sc.pp.filter_genes(a, min_cells=args.min_cells)
    a = a[:, np.asarray((a.layers["counts"] > 0).sum(0)).ravel() > 0].copy()
    a.write_h5ad(args.out)
    print(f"saved {args.out}: {a.shape} | cond={a.obs['condition'].nunique()} "
          f"batch={a.obs['batch'].nunique()} cell_type={a.obs['cell_type'].nunique()}")


if __name__ == "__main__":
    main()
