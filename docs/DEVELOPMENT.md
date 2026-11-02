# Development notes: how the final configuration was chosen

All numbers are from this repository's experiments (Kang PBMC, three model seeds unless stated; scIB with `batch=donor`, `label=cell type`).
"Sim" means the Splatter simulation with known per-gene variance shares (three development simulations).

## Versions

| version | change | result |
|---|---|---|
| v3 | additive NB decoder, MLP factor maps, alternating training | PBMC Total .674 |
| **v3 + detach** | v3 + `z_c.detach()` inside `H` | PBMC Total .691 +- .009 (Batch .717, Bio .674); no instability in the 3 simulations and 3 PBMC seeds we ran |

## Single-factor changes on top of the final version (3 seeds)

| change | PBMC Total | Bio | Batch | Sim (dev) | verdict |
|---|---|---|---|---|---|
| none (final) | .691 | .674 | .717 | reference | |
| KL_c 1e-4 | .700 | .703 | .694 | batch MSE .029 -> .039, shared MSE .051 -> .069 | trade-off, not adopted |
| KL_c 1e-3 | .686 | .736 | .612 | `var_ratio_shared` .68 (< 0.7) | collapse on Sim |
| MMD_c 1e-3 | .682 | .631 | .757 | batch MSE .029 -> .038 | worse Bio |
| z_c = 64 | .364 | .199 | .611 | looked normal | collapsed on PBMC |

Earlier runs of KL_c 5e-3 and 1e-2 (on an earlier variant) also collapsed on PBMC (the shared factor carried no cell-type variance). The KL_c scale is data
dependent because the reconstruction term is averaged over genes: 5e-3 was fine on Sim and collapsed on PBMC.

## Optional post-hoc steps (not part of SILVA)

| step | PBMC Total |
|---|---|
| final `z_c` | .691 |
| + Harmony on `z_c` (donor) | .717 |
| PCA32 of decoded log-rate: F only | .634 / .637 (centered / standardized) |
| PCA32 of decoded log-rate: counterfactual (control reference, donor dropped) | .642 / .645 |
| PCA32 of decoded log-rate: donor dropped, condition kept | .620 / .618 |

Decoded-expression PCA raises Bio (up to .705) but lowers Batch (PCR comparison near 0), so Total is lower than the latent itself.

## Stability checks used

No crash; no validation-loss spike above best + 0.3 after epoch 20; `var_ratio_shared` (variance carried by `F` relative to the true cell-type variance)
within [0.7, 1.3] in simulations. A run that scores well because `z_c` carries nothing fails the last check.
