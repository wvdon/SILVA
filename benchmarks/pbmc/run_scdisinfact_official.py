"""scDisInFact with the OFFICIAL demo settings (github ZhangLabGT/scDisInFact demo.ipynb).
Ks=[8,2] (demo: [8,2,2] for two conditions; Kang has one), batch_size=64, nepochs=100, interval=10, lr=5e-4,
reg_mmd_comm=1e-4 reg_mmd_diff=1e-4 reg_kl_comm=1e-5 reg_kl_diff=1e-2 reg_class=1 reg_gl=1, NB loss.
All cells for training (no validation split, no early stopping), all genes, official create_scdisinfact_dataset preprocessing.
Embedding = mu_c of dict_inf from model.inference (as in the demo). Usage: python benchmarks/pbmc/run_scdisinfact_official.py --adata data/kang.h5ad --donor-key batch --condition-key condition --out runs/scdis_official_s0 --seed 0
Output: <out>/emb.npz (z_c, z_d, cell_ids, genes), config.json"""
import argparse, json, time, numpy as np, anndata as ad, scipy.sparse as sp, torch
from pathlib import Path
from scDisInFact import scdisinfact, create_scdisinfact_dataset
p = argparse.ArgumentParser(); p.add_argument("--adata", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
p.add_argument("--donor-key", default="batch"); p.add_argument("--condition-key", default="condition"); p.add_argument("--seed", type=int, default=0)
args = p.parse_args(); seed, out = args.seed, args.out; out.mkdir(parents=True, exist_ok=True)
a = ad.read_h5ad(args.adata); meta = a.obs[[args.condition_key]].rename(columns={args.condition_key: "condition"}).copy(); meta["batch"] = a.obs[args.donor_key].astype(str).values
counts = a.layers["counts"]; counts = counts if sp.issparse(counts) else sp.csr_matrix(counts)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dict = create_scdisinfact_dataset(counts, meta, condition_key=["condition"], batch_key="batch")
Ks = [8, 2]; lambs = dict(reg_mmd_comm=1e-4, reg_mmd_diff=1e-4, reg_kl_comm=1e-5, reg_kl_diff=1e-2, reg_class=1, reg_gl=1)
model = scdisinfact(data_dict=data_dict, Ks=Ks, batch_size=64, interval=10, lr=5e-4, seed=seed, device=device, **lambs)
t = time.time(); model.train(); model.train_model(nepochs=100, recon_loss="NB"); model.eval()
zc, zd = [], []
for ds in data_dict["datasets"]:
    with torch.no_grad():
        inf = model.inference(counts=ds.counts_norm.to(model.device), batch_ids=ds.batch_id[:, None].to(model.device), print_stat=False)
    zc.append(inf["mu_c"].cpu().numpy()); zd.append(inf["mu_d"][0].cpu().numpy())
ids = np.concatenate([m.index.to_numpy().astype(str) for m in data_dict["meta_cells"]])
np.savez(out / "emb.npz", z=np.concatenate(zc), z_c=np.concatenate(zc), z_d=np.concatenate(zd), cell_ids=ids, genes=a.var_names.to_numpy().astype(str))
json.dump(dict(Ks=Ks, nepochs=100, interval=10, lr=5e-4, batch_size=64, seed=seed, lambs=lambs, cells=int(len(ids)), genes=int(a.n_vars),
               protocol="official demo.ipynb defaults; all cells, no validation, no early stopping", seconds=time.time() - t), open(out / "config.json", "w"), indent=1)
(out / "status.json").write_text('{"state": "complete"}'); print("done", seed, time.time() - t)
