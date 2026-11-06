"""High-level entry point: fit SILVA (or the scDisInFact baseline) on an AnnData with raw counts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ._skeleton.model import create_scdisinfact_dataset, scdisinfact
from .model import SILVA
from .trainer import embed, make_split, train


def fit(adata, donor_key: str = "donor_id", condition_key: str | list[str] = "condition", counts_layer: str | None = "counts",
        zc_dim: int = 32, zd_dim: int = 2, dec_hidden: int = 256, reg_kl_comm: float = 1e-5, reg_mmd_comm: float = 1e-4,
        detach_zc_in_H: bool = True, method: str = "silva", max_epochs: int = 400, split: pd.Series | None = None, val_frac: float = 0.1,
        seed: int = 0, device: str | torch.device | None = None, out=None, verbose: bool = True):
    """Train SILVA and return (model, result, z_c, z_d).

    adata        : cells x genes with RAW integer counts in `counts_layer` (or .X when counts_layer is None).
    donor_key    : obs column of the known batch (donor/sample). Enters the decoder (B term) and the shared encoder.
    condition_key: obs column of the specific condition (one factor, e.g. ctrl/stim), or a list of columns for several
                   independent condition factors (one z_d and one H term per factor).
    split        : optional Series "train"/"validation" indexed by cell id; default: 10% validation inside every
                   donor x condition stratum.
    Returns z_c (cells x zc_dim) and z_d (cells x zd_dim; a list of such arrays when several condition keys are given)
    as posterior means in adata.obs_names order.
    """
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    counts = adata.layers[counts_layer] if counts_layer else adata.X
    counts = counts if sp.issparse(counts) else sp.csr_matrix(counts)
    if not np.allclose(counts.data, np.round(counts.data)):
        raise ValueError("SILVA needs raw integer counts")
    keys = [condition_key] if isinstance(condition_key, str) else list(condition_key)
    meta = pd.DataFrame({**{k: adata.obs[k].astype(str).to_numpy() for k in keys},
                         "batch": adata.obs[donor_key].astype(str).to_numpy()}, index=adata.obs_names)
    data_dict = create_scdisinfact_dataset(counts, meta, condition_key=keys, batch_key="batch")
    kw = dict(data_dict=data_dict, Ks=[zc_dim] + [zd_dim] * len(keys), batch_size=64, interval=5, lr=5e-4,
              reg_mmd_comm=reg_mmd_comm, reg_mmd_diff=1e-4, reg_gl=1, reg_class=1,
              reg_kl_comm=reg_kl_comm, reg_kl_diff=1e-2, seed=seed, device=device)
    if method == "silva":
        model = SILVA(**kw, dec_param="logmu", dec_hidden=dec_hidden, dec_interaction=True, reg_sparse=1e-3,
                          dec_detach_zc_in_H=detach_zc_in_H)
    elif method == "scdisinfact":   # baseline trained with the same loop and split
        model = scdisinfact(**kw)
    else:
        raise ValueError("method must be 'silva' or 'scdisinfact'")
    if split is None:
        split = make_split(meta.assign(donor=meta["batch"]), keys + ["batch"], val_frac, seed)
    result = train(model, data_dict, split, max_epochs=max_epochs, out=out, verbose=verbose,
                   std_floor=(method == "scdisinfact"))
    z_c, z_d = embed(model, data_dict, adata.obs_names, all_factors=len(keys) > 1)
    return model, result, z_c, z_d
