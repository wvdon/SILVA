# Method

## Model

SILVA decomposes the expression of cell *i* and gene *j* into additive, per-gene terms:

```
eta_ij    = a_j + F(z_c)_ij + H(z_c, z_d)_ij + B(donor_i)_j
log mu_ij = log(library_i) + eta_ij            X_ij ~ NB(mu_ij, theta_j)
```

| symbol | meaning |
|---|---|
| `z_c` | shared latent: cell state / cell type. Encoder input includes the donor one-hot. |
| `z_d` | specific latent: one condition factor (e.g. IFN stimulation, disease). Has a linear classifier and a learned per-condition Gaussian prior. |
| `F` | MLP `z_c -> genes`: cell-state term. |
| `H` | MLP `[z_c, z_d] -> genes`: condition term. It reads `z_c`, so the condition effect can be cell-state specific. |
| `B` | per-donor, per-gene offset (known nuisance variable; it is *not* a latent). |
| `a_j`, `theta_j` | gene bias and gene-wise NB dispersion. |

Each factor map is an MLP (`d -> 256 -> genes`), not a linear map. The terms are summed at the gene level, so each factor's
contribution to every gene stays readable.

SILVA is built on the scDisInFact skeleton; the additive NB decoder above is the new component.

### detach z_c in H (the final version)

`H` still receives `z_c` as an input (the forward pass is identical), but `z_c.detach()` is used, so the gradient of the reconstruction loss
reaches the `z_c` encoder only through `F`. Without it, `F` and `H` push `z_c` in opposite directions along directions that leave `eta`
unchanged; the KL term at its default weight (1e-5) is too weak to hold `z_c` there and `z_c` can drift. `tests/test_model.py` checks that the
forward values are identical and that only the gradient path differs.

## Training

Alternating two-step update per minibatch (one Adam optimizer, one batch of 64 cells from every donor x condition stratum):

1. update `Enc_c` and the decoder with `recon + w_mmd_c * MMD_c + w_kl_c * KL_c`
2. update `Enc_d`, the classifier and the decoder with `recon + w_mmd_d * MMD_d + w_class * CE + w_kl_d * KL_d + w_gl * GL`

`recon` is the mean NB negative log-likelihood over cells and genes. Because it is averaged over genes, the effective KL weight is
`w * n_genes`; a KL weight that works on one data set can collapse `z_c` on another (see DEVELOPMENT.md).

A random 10% of every stratum is held out; the validation total loss is evaluated every 5 epochs, the best weights are restored, the learning
rate is multiplied by 0.6 after 3 evaluations without improvement, and training stops after 8 evaluations without improvement.

Input to the encoders: per cell `counts / (total/100)`, `log1p`, then per-gene standardization (scDisInFact's `create_scdisinfact_dataset`).
The decoder is trained on the raw integer counts with the observed library size. All genes are used; no HVG selection.

## Per-gene attribution and counterfactuals

* `AnchoredAdditiveDecoder` (silva/anchored.py) re-labels the trained terms around a reference condition `z_ref` (default: mean posterior `z_d` of the
  control cells): `F_anchored = F + h(z_c, z_ref)`, `H_anchored = H - h(z_c, z_ref)`. The sum is unchanged, so nothing is retrained; `H_anchored` is exactly 0 at the
  reference. It is a definition of the components, not a proof that they are the true biological sources.
* Counterfactual expression: decode with `z_d` replaced by the target condition and the donor term dropped (`spec_override`, `drop_batch`).

```python
dec, sha = silva.decoder_from_checkpoint("runs/s0/model.pt", n_genes=adata.n_vars)
ref = silva.AnchorReference.control_posterior_mean(z_d, adata.obs["condition"].to_numpy(), "ctrl", checkpoint_sha256=sha)
terms = silva.AnchoredAdditiveDecoder(dec, ref).eval().anchored_terms(z_c_tensor, [z_d_tensor], donor_idx_tensor[:, None])
# keys: shared_anchored, condition0_anchored, batch, shared_original, condition0_original
```

## Known limitations

* The condition term is not identifiable from the donor term when the two are confounded in the data. In Kang et al., control and stimulated
  cells were loaded as two separate pools, so any lane difference between them is indistinguishable from the IFN response.
* A condition-specific *cell population* that is absent from the reference condition is not modelled explicitly (not tested).
* A fixed 10% validation split is used for model selection; results are reported for three model seeds on one fixed split.
