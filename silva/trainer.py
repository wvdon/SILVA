"""Training loop for SILVA: alternating two-step update with validation early stopping.

Per minibatch (one batch per donor x condition stratum, 64 cells each, shortest stratum sets the epoch length):
  step 1  update Enc_c + decoder   with  recon + MMD_c + KL_c            (Enc_d and the classifier are frozen)
  step 2  update Enc_d + classifier + decoder with  recon + MMD_d + class + KL_d + group lasso   (Enc_c frozen)
One Adam optimizer. Validation total loss is evaluated every `eval_every` epochs on a held-out 10% of every stratum;
the best weights are kept, the learning rate is cut by 0.6 after 3 evaluations without improvement, and training stops after
`patience` evaluations without improvement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .model import SILVA


def make_split(obs: pd.DataFrame, group_cols, val_frac: float = 0.1, seed: int = 0) -> pd.Series:
    """Random validation cells inside every stratum defined by `group_cols` (at least one per stratum).
    Returns a Series of "train" / "validation" indexed by cell id."""
    rng = np.random.default_rng(seed)
    split = np.array(["train"] * len(obs), dtype=object)
    for _, ix in obs.groupby(list(group_cols), observed=True).indices.items():
        split[rng.choice(ix, size=max(1, int(round(val_frac * len(ix)))), replace=False)] = "validation"
    return pd.Series(split, index=obs.index, name="split")


def loss_terms(model, batch, std_floor: bool = False):
    """(recon, KL_c, KL_d, MMD_c, MMD_d, class, group-lasso) for a list of per-stratum minibatches.
    std_floor: scDisInFact's training-time posterior standard-deviation floor (0.01); used only for the
    scDisInFact baseline, never for SILVA."""
    cn, bid, mbid, sf, ct, dl = SILVA._assemble(model, batch)
    kw = dict(clamp_comm=0.01, clamp_diff=0.01) if std_floor else {}
    di = model.inference(counts=cn, batch_ids=bid, print_stat=False, **kw)
    dg = model.generative(z_c=di["z_c"], z_d=di["z_d"], batch_ids=bid)
    return model.loss(di, dg, sf, ct, mbid, dl, "NB")


def total_loss(model, terms):
    recon, klc, kld, mmc, mmd, cls, gl = terms
    w = model.lambs
    return (recon + w["kl_comm"] * klc + w["kl_diff"] * kld + w["mmd_comm"] * mmc
            + w["mmd_diff"] * mmd + w["class"] * cls + w["gl"] * gl)


def _set_phase(model: SILVA, phase: int):
    shared = phase == 1
    for k in range(model.n_diff_factors):
        for p in model.Enc_ds[k].parameters():
            p.requires_grad = not shared
        for p in model.classifiers[k].parameters():
            p.requires_grad = not shared
    for p in model.Enc_c.parameters():
        p.requires_grad = shared
    for p in model.Dec.parameters():
        p.requires_grad = True


def train(model, data_dict: dict, split: pd.Series, max_epochs: int = 400, patience: int = 8,
          eval_every: int = 5, out: Path | None = None, verbose: bool = True, std_floor: bool = False) -> dict:
    """Train `model` in place and restore its best validation checkpoint.

    split: Series of "train" / "validation" indexed by cell id (see `make_split`).
    """
    tr, va = [], []
    for ds, block in zip(data_dict["datasets"], data_dict["meta_cells"]):
        labels = split.loc[block.index].to_numpy()
        ti, vi = np.where(labels == "train")[0], np.where(labels == "validation")[0]
        if not len(ti) or not len(vi):
            raise ValueError("every donor x condition stratum needs training and validation cells")
        tr.append(DataLoader(Subset(ds, ti.tolist()), batch_size=model.batch_size, shuffle=True))
        va.append(DataLoader(Subset(ds, vi.tolist()), batch_size=512, shuffle=False))

    best, best_state, best_epoch, bad, lr_bad, updates = float("inf"), None, None, 0, 0, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for batch in zip(*tr):
            _set_phase(model, 1)
            model.opt.zero_grad()
            recon, klc, kld, mmc, mmd, cls, gl = loss_terms(model, batch, std_floor)
            (recon + model.lambs["mmd_comm"] * mmc + model.lambs["kl_comm"] * klc).backward()
            model.opt.step(); model.opt.zero_grad(); updates += 1
            _set_phase(model, 2)
            recon, klc, kld, mmc, mmd, cls, gl = loss_terms(model, batch, std_floor)
            (recon + model.lambs["mmd_diff"] * mmd + model.lambs["class"] * cls
             + model.lambs["kl_diff"] * kld + model.lambs["gl"] * gl).backward()
            model.opt.step(); model.opt.zero_grad(); updates += 1
            losses.append(float(recon.detach()))
        if epoch % eval_every == 0 or epoch == max_epochs - 1:
            model.eval()
            with torch.no_grad():
                vt = [loss_terms(model, b) for b in zip(*va)]
                v = float(np.mean([float(total_loss(model, t)) for t in vt]))
                w = model.lambs   # weighted validation terms, to see which one drives the total
                vterms = dict(zip(["nb", "kl_c", "kl_d", "mmd_c", "mmd_d", "cls", "gl"], [round(float(x), 4) for x in np.mean(
                    [[float(t[0]), float(w["kl_comm"] * t[1]), float(w["kl_diff"] * t[2]), float(w["mmd_comm"] * t[3]), float(w["mmd_diff"] * t[4]), float(w["class"] * t[5]), float(w["gl"] * t[6])] for t in vt], 0)]))
            if not np.isfinite(v):
                raise ValueError("non-finite validation loss")
            history.append(dict(epoch=epoch, optimizer_updates=updates, train_nb=float(np.mean(losses)),
                                val_total=v, lr=model.opt.param_groups[0]["lr"], val_terms=vterms))
            if verbose:
                print(json.dumps(history[-1]), flush=True)
            if v < best - 1e-4:
                best, best_epoch, bad, lr_bad = v, epoch, 0, 0
                best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                bad += 1; lr_bad += 1
                if lr_bad >= 3:
                    for pg in model.opt.param_groups:
                        pg["lr"] = max(pg["lr"] * 0.6, 1e-6)
                    lr_bad = 0
            if out is not None:
                (out / "history.json").write_text(json.dumps(history, indent=2))
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return dict(best_epoch=best_epoch, best_val_total=best, stopped_epoch=epoch, optimizer_updates=updates,
                history=history, best_state=best_state)


@torch.no_grad()
def embed(model, data_dict: dict, obs_names: pd.Index, chunk: int = 128, all_factors: bool = False):
    """Posterior means (z_c, z_d) for every cell, returned in `obs_names` order.
    z_d is the array of the first (only) condition factor, or a list with one array per factor when `all_factors` is True."""
    model.eval()
    zc, ids = [], []; zd = [[] for _ in range(model.n_diff_factors)]
    for ds, block in zip(data_dict["datasets"], data_dict["meta_cells"]):
        for lo in range(0, len(ds), chunk):
            sl = slice(lo, lo + chunk)
            di = model.inference(ds.counts_norm[sl].to(model.device), ds.batch_id[sl, None].to(model.device))
            zc.append(di["mu_c"].cpu().numpy()); ids.extend(block.index[sl])
            for k in range(model.n_diff_factors): zd[k].append(di["mu_d"][k].cpu().numpy())
    order = pd.Index(ids).get_indexer(obs_names)
    zds = [np.concatenate(z)[order] for z in zd]
    return np.concatenate(zc)[order], (zds if all_factors else zds[0])
