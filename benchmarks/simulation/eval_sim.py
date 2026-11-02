#!/usr/bin/env python
"""Per-gene variance shares vs the known simulation truth (lower MSE / higher Pearson r is better).

  python benchmarks/simulation/eval_sim.py \
      --runs s1=runs/sim1:results/sim/sim1_s1_truth_gene.csv s2=runs/sim2:results/sim/sim1_s2_truth_gene.csv \
      --models SILVA=silva_seed0 scDisInFact=scdisinfact_seed0 --out results/sim_eval.csv

--runs  name=<run folder with input.h5ad>:<truth csv from sim_v1_splatter.R>;  models live in <run>/methods/<tag>/ (model.pt, emb.npz).
Model share per gene = covariance allocation of the centred terms F (shared), H (condition), B (batch) on the log-rate scale, library excluded:
  SILVA: control-anchored F/H (silva.anchored), B = donor term. scDisInFact (name must start with 'scDisInFact'): counterfactual decoding of its
  joint decoder (B = donor effect, H = change when z_d is set to the control mean, F = remainder).
Truth share per gene = simulator variance share (cell type, condition, batch). Metrics over expressed genes:
  mse_/r_{shared,cond,batch}_share, r_delta_beta_affected (H contrast vs true log fold change), var_ratio_shared (collapse check, ideal 1).
"""
import argparse, sys
from pathlib import Path
import anndata as ad, numpy as np, pandas as pd, torch
from scipy.stats import pearsonr
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from silva.anchored import AnchorReference, AnchoredAdditiveDecoder, decoder_from_checkpoint


def silva_terms(folder, a, cond, donor):
    dec, sha = decoder_from_checkpoint(folder / "model.pt", n_genes=a.n_vars)
    with np.load(folder / "emb.npz", allow_pickle=True) as f:
        ix = pd.Index(f["cell_ids"].astype(str)).get_indexer(a.obs_names.astype(str)); assert (ix >= 0).all()
        zc, zd = f["z_c"][ix].astype(np.float32), f["z_d"][ix].astype(np.float32)
    w = AnchoredAdditiveDecoder(dec, AnchorReference.control_posterior_mean(zd, cond, "ctrl", checkpoint_sha256=sha)).eval()
    out = {k: np.empty((len(zc), a.n_vars), np.float32) for k in ("F", "H", "B")}
    with torch.no_grad():
        for i in range(0, len(zc), 2048):
            t = w.anchored_terms(torch.as_tensor(zc[i:i + 2048]), [torch.as_tensor(zd[i:i + 2048])], torch.as_tensor(donor[i:i + 2048])[:, None])
            for k, name in (("F", "shared_anchored"), ("H", "condition0_anchored"), ("B", "batch")):
                out[k][i:i + 2048] = t[name].numpy()
    return out


def scdisinfact_terms(folder, a, cond, donor, n_batches=8):
    from silva._skeleton import base_model
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    dec = base_model.Decoder(n_input=state["Dec.fc.fc_layers.Layer 0.0.weight"].shape[1] - n_batches, n_output=a.n_vars, n_cat_list=[n_batches],
                             n_layers=2, n_hidden=128, dropout_rate=0.2, use_batch_norm=False)
    dec.load_state_dict({k[4:]: v for k, v in state.items() if k.startswith("Dec.")}, strict=True); dec.eval()
    with np.load(folder / "emb.npz", allow_pickle=True) as f:
        ix = pd.Index(f["cell_ids"].astype(str)).get_indexer(a.obs_names.astype(str)); assert (ix >= 0).all()
        zc, zd = f["z_c"][ix].astype(np.float32), f["z_d"][ix].astype(np.float32)
    zref = torch.as_tensor(zd[cond == "ctrl"].mean(0))
    lm = lambda c, d, b: torch.log(dec(torch.cat([c, d], 1), torch.as_tensor(b)[:, None].long())[0] + 1e-8)
    out = {k: np.empty((len(zc), a.n_vars), np.float32) for k in ("F", "H", "B")}
    with torch.no_grad():
        for i in range(0, len(zc), 2048):
            c, d, b = torch.as_tensor(zc[i:i + 2048]), torch.as_tensor(zd[i:i + 2048]), donor[i:i + 2048]
            full = lm(c, d, b); avg_b = torch.stack([lm(c, d, np.full(len(b), j)) for j in range(n_batches)]).mean(0)
            B, H = full - avg_b, full - lm(c, zref.expand_as(d), b)
            out["B"][i:i + 2048], out["H"][i:i + 2048], out["F"][i:i + 2048] = B.numpy(), H.numpy(), (full - B - H).numpy()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--models", nargs="+", required=True); p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(); models = dict(m.split("=", 1) for m in args.models); rows = []
    for spec in args.runs:
        name, rest = spec.split("=", 1); run, truth = rest.split(":", 1); run = Path(run)
        a = ad.read_h5ad(run / "input.h5ad"); tr = pd.read_csv(truth); keep = tr.expressed.to_numpy()
        cond = a.obs.condition.astype(str).to_numpy(); donor, _ = pd.factorize(a.obs.donor_id.astype(str), sort=True)
        tv = tr[["var_celltype", "var_condition", "var_batch"]].to_numpy()[keep]; tshare = tv / tv.sum(1, keepdims=True); beta = tr.beta.to_numpy()
        for mname, tag in models.items():
            folder = run / "methods" / tag
            if not (folder / "model.pt").exists(): continue
            T = (scdisinfact_terms if mname.startswith("scDisInFact") else silva_terms)(folder, a, cond, donor)
            X = [T[k].astype(np.float64) for k in ("F", "H", "B")]; c = [M - M.mean(0) for M in X]; tot = sum(c); vt = np.maximum((tot ** 2).mean(0), 1e-12)
            alloc = np.stack([(ck * tot).mean(0) / vt for ck in c], 1)[keep]; var = np.stack([(ck ** 2).mean(0) for ck in c], 1)[keep]
            stim = cond == "stim"; delta = X[1][stim].mean(0) - X[1][~stim].mean(0); aff = (beta != 0)[keep]
            r = dict(sim=name, model=mname, var_ratio_shared=var[:, 0].sum() / tv[:, 0].sum(),
                     r_delta_beta_affected=pearsonr(delta[keep][aff], beta[keep][aff]).statistic)
            for j, n in enumerate(("shared", "cond", "batch")):   # truth columns (celltype, condition, batch); model columns (F, H, B)
                r[f"mse_{n}_share"] = ((alloc[:, j] - tshare[:, j]) ** 2).mean(); r[f"r_{n}_share"] = pearsonr(alloc[:, j], tshare[:, j]).statistic
            rows.append(r); print(name, mname, "done", flush=True)
    d = pd.DataFrame(rows); args.out.parent.mkdir(parents=True, exist_ok=True); d.to_csv(args.out, index=False)
    pd.set_option("display.width", 250, "display.float_format", "{:.4f}".format); print(d.groupby("model").mean(numeric_only=True).T)


if __name__ == "__main__":
    main()
