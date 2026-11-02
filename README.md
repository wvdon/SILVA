# SILVA

SILVA is a disentangled representation model for single-cell RNA-seq. It splits expression into a **shared cell-state
factor** `z_c` and a **condition-specific factor** `z_d`, treats the donor as a known nuisance variable, and uses an **additive negative-binomial
decoder** so that every factor's contribution is a readable per-gene quantity:

```
eta = gene_bias + F(z_c) + H(z_c, z_d) + B(donor)          log mu = log(library) + eta          X ~ NB(mu, theta_gene)
```

`H` is the per-gene condition attribution; decoding with `z_d` swapped (and the donor term dropped) gives a counterfactual.

**What SILVA adds**
- an additive NB decoder with MLP factor maps, so the shared, condition and donor contributions are separate per-gene terms;
- `detach z_c in H`: `H` reads `z_c`, but its gradient does not reach the `z_c` encoder ([docs/METHOD.md](docs/METHOD.md));
- reference-anchored per-gene attribution and counterfactual decoding;
- the benchmarks and the simulation with known per-gene variance shares.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch                       # pick the build matching your CUDA
pip install git+https://github.com/wvdon/SILVA.git     # or, from a clone: pip install -e .
pip install -e ".[benchmarks]"          # only for the benchmark scripts (scanpy, scib-metrics, scvi-tools, harmonypy)
python -m unittest discover -s tests    # 16 tests, CPU, a few seconds
```

## Use

```python
import anndata as ad, silva
adata = ad.read_h5ad("data.h5ad")        # raw integer counts in .layers["counts"]; obs has a donor column and a condition column
model, result, z_c, z_d = silva.fit(adata, donor_key="donor_id", condition_key="condition")
adata.obsm["X_silva"] = z_c              # shared latent (cell state), cells x 32
```

Command line (writes `model.pt`, `emb.npz`, `config.json`, `history.json`):

```bash
python scripts/train.py --adata data/kang.h5ad --donor-key batch --condition-key condition --out runs/kang_s0 --seed 0
```

Defaults ([configs/default.yaml](configs/default.yaml)): `z_c` 32, `z_d` 2, factor-map width 256, NB, all genes (no HVG selection), 10% of every
donor x condition stratum held out for early stopping. Per-gene attribution and counterfactual decoding: [docs/METHOD.md](docs/METHOD.md).

## Layout

```
silva/            model.py (AdditiveDecoder, SILVA), trainer.py (alternating updates + early stopping), api.py (fit), anchored.py (attribution)
scripts/          train.py
benchmarks/pbmc/  prepare_kang.py, score_scib.py, run_scvi.py, run_scdisinfact_official.py, harmony_on_zc.py, split.csv (the split used)
benchmarks/simulation/  sim_v1_splatter.R, make_sim_run.py, eval_sim.py
configs/ docs/ tests/
```

## License

GNU General Public License v3.0 ([LICENSE](LICENSE)).
