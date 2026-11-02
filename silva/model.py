"""SILVA: additive negative-binomial decoder on the scDisInFact skeleton.

    eta_ij  = gene_bias_j + F(z_c)_ij + H(z_c, z_d)_ij + B(donor_i)_j      # each term [cells x genes]
    log mu_ij = log(library_i) + eta_ij
    X_ij ~ NB(mu_ij, theta_j)

F is the shared (cell-state) term, H the condition term (cell-state specific through z_c), B the donor term, so every
factor's contribution to every gene is readable. Final configuration ("v3 + detach"): H still reads z_c, but its gradient does
not flow into z_c (`detach_zc_in_H=True`), so the z_c encoder is shaped by F, KL and MMD only.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt

from scDisInFact.model import scdisinfact
import scDisInFact.loss_function as loss_func


def nb_log_prob(counts: torch.Tensor, log_mu: torch.Tensor,
                log_theta: torch.Tensor) -> torch.Tensor:
    """Stable fp32 negative-binomial log-likelihood. counts/log_mu: [B,G];
    log_theta: [G] or [B,G]. Parameterization: mean=mu, var = mu + mu^2/theta."""
    counts = counts.float(); log_mu = log_mu.float(); log_theta = log_theta.float()
    log_theta_mu = torch.logaddexp(log_theta, log_mu)
    return (torch.lgamma(counts + log_theta.exp()) - torch.lgamma(log_theta.exp())
            - torch.lgamma(counts + 1.0) + log_theta.exp() * (log_theta - log_theta_mu)
            + counts * (log_mu - log_theta_mu))


class AdditiveDecoder(nn.Module):
    """Additive per-gene decoder (gene_as_output): eta = bias + f_shared(z_c)
    + Σ_k f_spec(z_d_k) + f_batch(batch). Each term gene-mean centered per cell.
    Returns log_softmax(eta) so the library scale is applied in the loss."""

    def __init__(self, d_common: int, d_diffs: list[int], n_genes: int,
                 n_batches: int, param: str = "logmu", gauge_fix: bool = False,
                 hidden: int = 128, interaction: bool = True, spec_ref: bool = False,
                 detach_zc_in_H: bool = True):
        super().__init__()
        # spec_ref: reference-anchored specific branch. term_k = h(z_c, z_d) - h(z_c, z_d_ref), so the
        #   branch is exactly 0 at the reference condition and any pure-z_c component of h cancels.
        self.spec_ref = spec_ref
        # interaction: if True, f_spec takes [z_c, z_d] -> the condition's per-gene effect
        #   is CELL-STATE-SPECIFIC (cell-type-specific disease program), still an additive
        #   term so attribution holds. If False, f_spec(z_d) only (global condition shift).
        self.interaction = interaction
        # detach_zc_in_H: H still reads z_c as an input (identical forward values) but its gradient does not flow back
        #   into z_c / Enc_c, so z_c is shaped only by F (+ KL/MMD). True = final SILVA; False = the earlier "v3".
        self.detach_zc_in_H = detach_zc_in_H
        self.d_common = d_common
        # param: "softmax" -> log_mu = log(lib) + log_softmax(eta) (compositional, E3);
        #        "logmu"   -> log_mu = log(lib) + (bias + Σ terms) (independent per-gene,
        #                     closer to scDisInFact; still additive -> attribution holds).
        # gauge_fix: per-cell gene-mean center each term (E1; only meaningful for softmax).
        # hidden: 0 -> each factor map is Linear(d->G); >0 -> MLP(d->hidden->G) to match
        #         scDisInFact decoder capacity (still additive at the GENE level).
        assert param in ("softmax", "logmu")
        self.param = param
        self.gauge_fix = gauge_fix
        init_bias = 0.0 if param == "softmax" else -float(np.log(n_genes))
        self.gene_bias = nn.Parameter(torch.full((n_genes,), init_bias))
        self.log_theta = nn.Parameter(torch.zeros(n_genes))

        def mk_map(d):
            if hidden and hidden > 0:
                return nn.Sequential(nn.Linear(d, hidden), nn.LayerNorm(hidden),
                                     nn.ReLU(), nn.Linear(hidden, n_genes))
            return nn.Linear(d, n_genes)
        self.shared_map = mk_map(d_common)
        self.spec_maps = nn.ModuleList(
            mk_map(d + d_common if interaction else d) for d in d_diffs)
        self.batch_emb = nn.Embedding(n_batches, n_genes)   # per-batch per-gene shift
        nn.init.zeros_(self.batch_emb.weight)

    def _maybe_center(self, t: torch.Tensor) -> torch.Tensor:
        return t - t.mean(dim=1, keepdim=True) if self.gauge_fix else t

    def eta_terms(self, z_c, z_d, batch_ids, spec_override=None, z_ref=None):
        """spec_override: optional list replacing z_d per factor (counterfactual)."""
        zd = spec_override if spec_override is not None else z_d
        terms = {"shared": self._maybe_center(self.shared_map(z_c))}
        zc_h = z_c.detach() if self.detach_zc_in_H else z_c
        for k, m in enumerate(self.spec_maps):
            inp = torch.cat([zc_h, zd[k]], dim=1) if self.interaction else zd[k]
            out = m(inp)
            if self.spec_ref and z_ref is not None:
                ref = torch.cat([zc_h, z_ref[k]], dim=1) if self.interaction else z_ref[k]
                out = out - m(ref)
            terms[f"spec{k}"] = self._maybe_center(out)
        b = batch_ids[:, 0].long() if batch_ids.dim() == 2 else batch_ids.long()
        terms["batch"] = self._maybe_center(self.batch_emb(b))
        return terms

    def forward(self, z_c, z_d, batch_ids, drop_batch=False, spec_override=None, z_ref=None):
        terms = self.eta_terms(z_c, z_d, batch_ids, spec_override, z_ref)
        eta = self.gene_bias + terms["shared"] + sum(terms[f"spec{k}"]
                                                     for k in range(len(z_d)))
        if not drop_batch:
            eta = eta + terms["batch"]
        log_term = F.log_softmax(eta, dim=1) if self.param == "softmax" else eta
        return {"log_term": log_term, "eta_terms": terms}


class JointDecoder(nn.Module):
    """Control decoder with the scDisInFact-style *joint* structure (no additive per-factor terms).

    log_mu = log(library) + MLP([z_c, z_d, donor one-hot]); same NB likelihood, per-gene theta and
    log-rate offset as AdditiveDecoder, so only the decoder structure differs. There is no per-gene
    factor decomposition: 'shared'/'spec' terms are zero and the 'batch' term is defined (eval mode
    only) as the change in log-rate when the donor input is zeroed.
    """

    def __init__(self, d_common: int, d_diffs: list[int], n_genes: int, n_batches: int,
                 hidden: int = 256, n_layers: int = 2, **_unused):
        super().__init__()
        self.n_batches, self.n_genes = n_batches, n_genes
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        d_in = d_common + sum(d_diffs) + n_batches
        layers, d = [], d_in
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        out = nn.Linear(d, n_genes)
        nn.init.constant_(out.bias, -float(np.log(n_genes)))
        self.net = nn.Sequential(*layers, out)

    def _decode(self, z_c, zd, b, drop_batch=False):
        onehot = F.one_hot(b, self.n_batches).float()
        if drop_batch:
            onehot = torch.zeros_like(onehot)
        return self.net(torch.cat([z_c] + list(zd) + [onehot], dim=1))

    def forward(self, z_c, z_d, batch_ids, drop_batch=False, spec_override=None, z_ref=None):
        zd = spec_override if spec_override is not None else z_d
        b = batch_ids[:, 0].long() if batch_ids.dim() == 2 else batch_ids.long()
        log_term = self._decode(z_c, zd, b, drop_batch)
        zeros = torch.zeros_like(log_term)
        batch_term = zeros
        if not self.training:   # exported 'corrected' rate = log_term - batch_term
            with torch.no_grad():
                batch_term = log_term - self._decode(z_c, zd, b, drop_batch=True)
        return {"log_term": log_term, "eta_terms": {"shared": zeros, "spec0": zeros, "batch": batch_term}}


class SILVA(scdisinfact):
    """scDisInFact + additive NB decoder. Only __init__/generative/loss differ; inference, the dataset and the
    encoders are inherited. Train with `silva.trainer.fit` (alternating two-step update, validation early stopping)."""

    def __init__(self, *args, dec_param: str = "logmu", dec_gauge_fix: bool = False,
                 dec_hidden: int = 128, dec_interaction: bool = True,
                 reg_sparse: float = 1e-3, dec_spec_ref: bool = False, ref_label: int = 0, dec_type: str = "additive",
                 dec_detach_zc_in_H: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dec_spec_ref, self.ref_label = dec_spec_ref, ref_label
        self.reg_sparse = reg_sparse  # L1 on f_spec output weights (sparse gene programs)
        # replace the joint MLP decoder with the additive NB decoder, then rebuild opt
        _Dec = JointDecoder if dec_type == "joint" else AdditiveDecoder
        self.Dec = _Dec(
            d_common=self.Ks["common_factor"],
            d_diffs=self.Ks["diff_factors"],
            n_genes=self.ngenes,
            n_batches=len(self.uniq_batch_ids),
            param=dec_param, gauge_fix=dec_gauge_fix, hidden=dec_hidden,
            interaction=dec_interaction, spec_ref=dec_spec_ref,
            detach_zc_in_H=dec_detach_zc_in_H,
        ).to(self.device)
        self.opt = opt.Adam(self.parameters(), lr=self.lr)

    def anchored_decoder(self, reference):
        """Read-only reference-anchored view of self.Dec (see silva/anchored.py); changes nothing in training."""
        from .anchored import AnchoredAdditiveDecoder
        return AnchoredAdditiveDecoder(self.Dec, reference)

    def generative(self, z_c, z_d, batch_ids):
        z_ref = None
        if self.dec_spec_ref:   # reference = (detached) learned prior mean of the reference condition
            z_ref = [self.diff_prior_means[k][self.ref_label].detach()[None].expand(z_c.shape[0], -1)
                     for k in range(self.n_diff_factors)]
        out = self.Dec(z_c, z_d, batch_ids, z_ref=z_ref)
        out["z_pred"] = [self.classifiers[k](z_d[k]) for k in range(self.n_diff_factors)]
        return out

    def loss(self, dict_inf, dict_gen, size_factor, count, batch_id, diff_labels,
             recon_loss):
        """Recon = additive NB (log_mu = log(lib)+log_softmax(eta)); KL/MMD/class/
        group-lasso are computed exactly as in scDisInFact (copied)."""
        ce_loss = nn.CrossEntropyLoss(reduction="mean")

        # 1. additive NB reconstruction (replaces scDisInFact's softplus-NB Dec)
        library = count.sum(dim=1, keepdim=True).clamp_min(1.0)
        log_mu = library.log() + dict_gen["log_term"]
        loss_recon = -nb_log_prob(count, log_mu, self.Dec.log_theta).mean()

        # 2. KL (common to N(0,I); diff to per-condition learned prior) -- unchanged
        loss_kl_comm = torch.mean(-0.5 * torch.sum(
            1 + dict_inf["logvar_c"] - dict_inf["mu_c"] ** 2
            - dict_inf["logvar_c"].exp(), dim=1), dim=0)
        loss_kl_diff = 0
        for k in range(self.n_diff_factors):
            pm = self.diff_prior_means[k][diff_labels[k], :]
            plv = self.diff_prior_logvars[k][diff_labels[k], :]
            loss_kl_diff += torch.mean(-0.5 * torch.sum(
                1 + dict_inf["logvar_d"][k] - plv
                - (dict_inf["logvar_d"][k].exp()
                   + (dict_inf["mu_d"][k] - pm) ** 2) / plv.exp(), dim=1), dim=0)

        # 3. classifier on z_d -- unchanged
        loss_class = 0
        for k in range(self.n_diff_factors):
            loss_class += ce_loss(input=dict_gen["z_pred"][k], target=diff_labels[k])

        # 4. MMD (comm across batch_cond to ref; diff within each condition) -- unchanged
        loss_mmd_comm = loss_func.maximum_mean_discrepancy(
            xs=dict_inf["mu_c"], batch_ids=batch_id, device=self.device,
            ref_batch=self.mmd_ref_comm)
        loss_mmd_diff = 0
        for k in range(self.n_diff_factors):
            for diff_label in self.uniq_diff_labels[k]:
                idx = diff_labels[k] == diff_label
                if any(idx):
                    loss_mmd_diff += loss_func.maximum_mean_discrepancy(
                        xs=dict_inf["mu_d"][k][idx, :], batch_ids=batch_id[idx],
                        device=self.device, ref_batch=self.mmd_ref_diff[k][diff_label])

        # 5. group lasso on the diff-encoder first layer (unchanged) + L1 sparsity on the
        #    f_spec output weights (sparse, interpretable per-gene condition programs)
        loss_gl_d = 0
        for k in range(self.n_diff_factors):
            loss_gl_d += loss_func.grouplasso(
                self.Enc_ds[k].fc.fc_layers[0][0].weight, alpha=0.1)
            if hasattr(self.Dec, "spec_maps"):   # joint control decoder has no f_spec output to penalise
                sm = self.Dec.spec_maps[k]
                w = sm[-1].weight if isinstance(sm, nn.Sequential) else sm.weight
                loss_gl_d = loss_gl_d + self.reg_sparse * w.abs().mean()

        return (loss_recon, loss_kl_comm, loss_kl_diff, loss_mmd_comm,
                loss_mmd_diff, loss_class, loss_gl_d)

    def _assemble(self, data_batch):
        cn, bid, mbid, sf, ct = [], [], [], [], []
        dl = [[] for _ in range(self.n_diff_factors)]
        for x in data_batch:
            cn.append(x["counts_norm"]); bid.append(x["batch_id"][:, None])
            mbid.append(x["mmd_batch_id"]); sf.append(x["size_factor"]); ct.append(x["counts"])
            for k in range(self.n_diff_factors):
                dl[k].append(x["diff_labels"][k])
        d = self.device
        cn = torch.cat(cn).to(d); bid = torch.cat(bid).to(d); mbid = torch.cat(mbid).to(d)
        sf = torch.cat(sf).to(d); ct = torch.cat(ct).to(d)
        dl = [torch.cat(v).to(d) for v in dl]
        return cn, bid, mbid, sf, ct, dl
