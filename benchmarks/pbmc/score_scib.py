#!/usr/bin/env python
"""Score embeddings with scib-metrics (Benchmarker defaults; Total = 0.4 Batch + 0.6 Bio; min_max_scale=False).

  python benchmarks/pbmc/score_scib.py --adata data/kang.h5ad --batch-key batch --label-key cell_type \
      --embeddings SILVA_s0=runs/s0/emb.npz:z_c SILVA_s1=runs/s1/emb.npz:z_c --out results/scib

Each embedding is an npz with `cell_ids` and an array (default key `z_c`, the shared latent). Labels are used for scoring only.
The Unintegrated reference is a PCA32 of log1p(CP1e5) standardized expression over all genes (the same recipe for every dataset).
Strict KMeans ARI = mean over 10 single-init restarts (the scIB Bio score uses one run and is noisy).
"""
import argparse, os
from pathlib import Path
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors


def raw_pca(adata, layer="counts", dims=32, seed=0):
    X = adata.layers[layer] if layer in adata.layers else adata.X
    X = sp.csr_matrix(X); lib = np.asarray(X.sum(1)).ravel()
    x = np.log1p((sp.diags(1e5 / lib) @ X).toarray())
    x = (x - x.mean(0)) / np.maximum(x.std(0), 1e-8)
    return PCA(dims, random_state=seed, svd_solver="randomized").fit_transform(x).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adata", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--batch-key", default="batch"); p.add_argument("--label-key", default="cell_type")
    p.add_argument("--embeddings", nargs="+", required=True, help="name=path.npz[:key]")
    p.add_argument("--raw-npz", type=Path, default=None, help="optional precomputed Unintegrated embedding (npz with z, cell_ids)")
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    from scib_metrics.benchmark import Benchmarker
    from scib_metrics.nearest_neighbors import NeighborsResults

    a = ad.read_h5ad(args.adata); obs = a.obs[[args.batch_key, args.label_key]].copy()
    data = ad.AnnData(np.zeros((len(obs), 1), np.float32), obs=obs)
    if args.raw_npz:
        with np.load(args.raw_npz, allow_pickle=True) as f:
            ix = pd.Index(f["cell_ids"].astype(str)).get_indexer(obs.index.astype(str)); data.obsm["Unintegrated"] = f["z"][ix].astype(np.float32)
    else:
        data.obsm["Unintegrated"] = raw_pca(a)
    keys = {}
    for spec in args.embeddings:
        name, rest = spec.split("=", 1); path, _, key = rest.partition(":")
        with np.load(path, allow_pickle=True) as f:
            ix = pd.Index(f["cell_ids"].astype(str)).get_indexer(obs.index.astype(str)); assert (ix >= 0).all(), name
            keys[name] = data.obsm[name] = f[key or "z_c"][ix].astype(np.float32)

    def neighbors(x, n):
        d, i = NearestNeighbors(n_neighbors=n, n_jobs=args.jobs).fit(x).kneighbors(x)
        return NeighborsResults(indices=i, distances=d)

    bm = Benchmarker(data, batch_key=args.batch_key, label_key=args.label_key, embedding_obsm_keys=list(keys),
                     pre_integrated_embedding_obsm_key="Unintegrated", n_jobs=args.jobs, progress_bar=False)
    bm.prepare(neighbor_computer=neighbors); bm.benchmark()
    res = bm.get_results(min_max_scale=False); res.to_csv(args.out / "metrics.csv")
    lab = obs[args.label_key].astype(str).to_numpy(); k = len(set(lab))
    km = pd.DataFrame([dict(embedding=n, restart=r, ARI=adjusted_rand_score(lab, KMeans(k, n_init=1, random_state=r).fit_predict(z)))
                       for n, z in keys.items() for r in range(10)])
    km.to_csv(args.out / "kmeans_restarts.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.T.to_string()); print(km.groupby("embedding").ARI.agg(["mean", "std"]))


if __name__ == "__main__":
    main()
