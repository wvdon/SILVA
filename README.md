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
pip install -e .                        # installs SILVA and its dependencies
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

## Benchmarks

Reproduce with the scripts in [benchmarks/](benchmarks/). Protocols, caveats and the negative results are in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

**PBMC (Kang 2018 IFN-beta, GSE96583; 24,562 cells, 15,701 genes, 8 donors x ctrl/stim = 16 samples).** scIB-metrics, `batch = donor`,
`label = cell type`, Total = 0.4 Batch + 0.6 Bio, shared latent 32-d (scDisInFact official: 8-d). SILVA rows: mean +- sd over 3 model seeds; other rows: one run.

| method | Batch | Bio | Total |
|---|---|---|---|
| Unintegrated (PCA32) | .525 | .663 | .608 |
| Harmony (donor) | .701 | .704 | .703 |
| CellANOVA | .654 | .678 | .669 |
| scVI (all genes, defaults, latent 32) | .649 | .639 | .643 |
| scDisInFact, same split/early stopping as SILVA (32 + 2) | .672 | .701 | .690 |
| scDisInFact, official demo settings (8 + 2, 100 epochs, 3 seeds) | .569 | .590 | .582 +- .012 |
| SILVA v3 | .676 | .673 | .674 |
| **SILVA v3 + detach** | .717 | .674 | **.691 +- .009** |
| SILVA v3 + detach + post-hoc Harmony on `z_c` | .776 | .679 | .717 +- .010 |

The scVI row used all genes rather than the HVG selection of the scVI tutorial. Post-hoc Harmony is a separate, optional step, not part of SILVA.
Kang's control and stimulated cells were loaded as separate pools, so lane differences cannot be separated from the IFN response.

**Simulation (Splatter, known per-gene variance shares; 3 simulations).** Mean squared error and Pearson r between each model's per-gene share of
shared / condition / batch variance and the truth.

| | MSE shared | MSE cond | MSE batch | r shared | r cond | r batch |
|---|---|---|---|---|---|---|
| scDisInFact (same trainer) | .124 | .0159 | .068 | .706 | .923 | **.877** |
| **SILVA v3 + detach** | **.051** | **.0071** | **.0285** | **.796** | **.960** | .839 |

SILVA has lower error on all three shares; scDisInFact's batch share correlates better with the truth (r .877 vs .839).

## Layout

```
silva/            model.py (AdditiveDecoder, SILVA), trainer.py (alternating updates + early stopping), api.py (fit), anchored.py (attribution)
scripts/          train.py
benchmarks/pbmc/  prepare_kang.py, score_scib.py, run_scvi.py, run_scdisinfact_official.py, harmony_on_zc.py, split.csv (the split used)
benchmarks/simulation/  sim_v1_splatter.R, make_sim_run.py, eval_sim.py
configs/ docs/ tests/
```

## License and citation

License: to be added.

SILVA is built on the [scDisInFact](https://github.com/ZhangLabGT/scDisInFact) skeleton.
