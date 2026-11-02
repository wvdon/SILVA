"""Convert a sim_v1_splatter.R output (<sim-dir>/<tag>_*) into a run folder (input.h5ad + split.csv) for scripts/train.py.
Counts in X and layers['counts']; obs: donor_id (= technical batch), condition, cell_type. split.csv: 10% validation, random within each
donor x condition stratum (seeded), as for the first three simulations."""
import argparse, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.io import mmread
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument("--tag", required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--seed", type=int, default=0)
p.add_argument("--sim-dir", type=Path, default=Path("results/sim"), help="directory written by sim_v1_splatter.R")
args = p.parse_args(); R = args.sim_dir
X = sp.csr_matrix(mmread(R / f"{args.tag}_counts.mtx").T.astype(np.float32))
genes = (R / f"{args.tag}_genes.txt").read_text().split(); obs = pd.read_csv(R / f"{args.tag}_obs.csv").set_index("cell")
a = ad.AnnData(X, obs=pd.DataFrame({"donor_id": obs.batch.values, "condition": obs.condition.values, "cell_type": obs.cell_type.values}, index=obs.index.astype(str)),
               var=pd.DataFrame(index=genes)); a.layers["counts"] = X.copy()
rng = np.random.default_rng(args.seed); split = np.array(["train"] * a.n_obs, dtype=object)
for _, ix in a.obs.groupby(["donor_id", "condition"]).indices.items():
    split[rng.choice(ix, size=max(1, int(round(.1 * len(ix)))), replace=False)] = "validation"
args.out.mkdir(parents=True, exist_ok=True); a.write_h5ad(args.out / "input.h5ad")
pd.DataFrame({"cell_id": a.obs_names, "split": split}).to_csv(args.out / "split.csv", index=False)
print(args.tag, a.shape, pd.Series(split).value_counts().to_dict())
