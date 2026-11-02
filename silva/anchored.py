"""Reference-anchored view of the additive decoder (function-preserving, no change to training).

v3:   eta = a + F + H + B,   F = f(z_c),   H = h(z_c, z_d),   B = b[donor]
Choose a fixed reference z_ref in z_d space and let U = h(z_c, z_ref). Then

    shared_anchored    = F + U            (profile of a cell in the reference condition)
    condition_anchored = H - U            (exactly 0 when z_d = z_ref)
    batch              = B                (unchanged)

so shared_anchored + condition_anchored == F + H exactly (up to float rounding) and the NB likelihood,
the encoders, the losses and every v3 checkpoint are untouched. The wrapper only re-labels
components. It adds no parameters and nothing is added to a state_dict, so original checkpoints load
and reproduce their outputs; the reference is stored in a separate, explicit AnchorReference file.

Anchoring is a definition of the components, not a proof that they are the true biological sources.
All values are natural-log log-rates; the library offset log(lib) is never included.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .model import AdditiveDecoder

REFERENCE_KINDS = ("control_posterior_mean", "prior_mean", "overall_posterior_mean", "custom")


@dataclass
class AnchorReference:
    """Explicit z_d reference, one vector per specific factor, in the model's own z_d coordinates.

    kind: how it was defined (see REFERENCE_KINDS). z_d coordinates are NOT comparable across seeds or
    retrained models, so a reference is only valid for the checkpoint it was derived from
    (checkpoint_sha256 records which one when known).
    """
    z_ref: list[np.ndarray]
    kind: str = "control_posterior_mean"
    reference_label: str | None = None
    n_cells: list[int] | None = None
    checkpoint_sha256: str | None = None
    notes: dict = field(default_factory=dict)

    def __post_init__(self):
        self.z_ref = [np.asarray(z, dtype=np.float32).reshape(-1) for z in self.z_ref]
        if not self.z_ref or any(z.size == 0 or not np.isfinite(z).all() for z in self.z_ref):
            raise ValueError("z_ref must be non-empty and finite")
        if self.kind not in REFERENCE_KINDS:
            raise ValueError(f"kind must be one of {REFERENCE_KINDS}")

    # -- construction ---------------------------------------------------------------------------
    @classmethod
    def control_posterior_mean(cls, z_d, condition_labels, reference_label, checkpoint_sha256=None):
        """Mean posterior z_d of the cells whose condition label equals reference_label (per factor).

        z_d: array [cells, d] (one factor) or list of such arrays; condition_labels: matching 1-d array(s).
        Primary reference. It is an empirical centroid, not a verified unstimulated biological state.
        """
        zs = [z_d] if isinstance(z_d, np.ndarray) else list(z_d)
        ls = [condition_labels] if isinstance(condition_labels, np.ndarray) else list(condition_labels)
        if len(zs) != len(ls):
            raise ValueError("one label vector per factor required")
        out, counts = [], []
        for z, lab in zip(zs, ls):
            m = np.asarray(lab).astype(str) == str(reference_label)
            if not m.any():
                raise ValueError(f"no cells with condition label {reference_label!r}")
            out.append(np.asarray(z, dtype=np.float64)[m].mean(0)); counts.append(int(m.sum()))
        return cls(out, "control_posterior_mean", str(reference_label), counts, checkpoint_sha256)

    @classmethod
    def overall_posterior_mean(cls, z_d, checkpoint_sha256=None):
        zs = [z_d] if isinstance(z_d, np.ndarray) else list(z_d)
        return cls([np.asarray(z, dtype=np.float64).mean(0) for z in zs], "overall_posterior_mean", None,
                   [len(z) for z in zs], checkpoint_sha256)

    @classmethod
    def prior_mean(cls, diff_prior_means, reference_index, checkpoint_sha256=None):
        """diff_prior_means: list (per factor) of [n_conditions, d] learned prior means."""
        pm = [diff_prior_means] if isinstance(diff_prior_means, (np.ndarray, torch.Tensor)) else list(diff_prior_means)
        return cls([np.asarray(p.detach().cpu() if torch.is_tensor(p) else p)[reference_index] for p in pm],
                   "prior_mean", str(reference_index), None, checkpoint_sha256)

    # -- persistence ----------------------------------------------------------------------------
    def to_dict(self):
        return dict(z_ref=[z.tolist() for z in self.z_ref], kind=self.kind, reference_label=self.reference_label,
                    n_cells=self.n_cells, checkpoint_sha256=self.checkpoint_sha256, notes=self.notes)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["z_ref"], d["kind"], d["reference_label"], d["n_cells"], d["checkpoint_sha256"], d.get("notes", {}))


def file_sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class AnchoredAdditiveDecoder(nn.Module):
    """Read-only anchored view of an AdditiveDecoder. Shares the base decoder's parameters (no copy, no
    new parameters); ``base.state_dict()`` is untouched, so old checkpoints load and reproduce unchanged.

    forward() returns exactly what the base decoder returns ('log_term', original 'eta_terms') and adds
    'anchored_terms'. The reference lives in a non-persistent buffer: it is not part of the checkpoint;
    keep it with AnchorReference.save().
    """

    def __init__(self, base: AdditiveDecoder, reference: AnchorReference):
        super().__init__()
        if not isinstance(base, AdditiveDecoder):
            raise TypeError("anchoring needs an AdditiveDecoder")
        if base.spec_ref:
            raise ValueError("base decoder was trained with spec_ref=True: its specific term is already "
                             "reference-subtracted, so a second anchor would be ambiguous")
        if len(reference.z_ref) != len(base.spec_maps):
            raise ValueError("reference must provide one z_ref per specific factor")
        self.base = base
        self.reference = reference
        device = base.gene_bias.device
        for k, (z, m) in enumerate(zip(reference.z_ref, base.spec_maps)):
            first = m[0] if isinstance(m, nn.Sequential) else m
            d = first.in_features - (base.d_common if base.interaction else 0)
            if z.size != d:
                raise ValueError(f"z_ref[{k}] has dim {z.size}, factor {k} expects {d}")
            self.register_buffer(f"z_ref_{k}", torch.as_tensor(z, device=device)[None], persistent=False)

    @property
    def log_theta(self):
        return self.base.log_theta

    def _ref(self, k, n):
        return getattr(self, f"z_ref_{k}").expand(n, -1)

    def anchored_terms(self, z_c, z_d, batch_ids, spec_override=None):
        """All components as [cells, genes] log-rate tensors, original and anchored, plus the intercept.

        With spec_override (counterfactual z_d) condition_anchored = h(z_c, z_override) - h(z_c, z_ref)
        and the shared/reference terms are unchanged.
        """
        base = self.base
        orig = base.eta_terms(z_c, z_d, batch_ids, spec_override=spec_override)
        zd = spec_override if spec_override is not None else z_d
        out = {"intercept": base.gene_bias[None].expand(z_c.shape[0], -1),
               "shared_original": orig["shared"], "batch": orig["batch"]}
        shared_anchored = orig["shared"]
        for k, m in enumerate(base.spec_maps):
            ref_in = torch.cat([z_c, self._ref(k, z_c.shape[0])], 1) if base.interaction else self._ref(k, z_c.shape[0])
            u = base._maybe_center(m(ref_in))
            out[f"condition{k}_original"] = orig[f"spec{k}"]
            out[f"reference{k}"] = u                                  # U = h(z_c, z_ref)
            out[f"condition{k}_anchored"] = orig[f"spec{k}"] - u
            shared_anchored = shared_anchored + u
        out["shared_anchored"] = shared_anchored
        return out

    def forward(self, z_c, z_d, batch_ids, drop_batch=False, spec_override=None):
        out = self.base(z_c, z_d, batch_ids, drop_batch=drop_batch, spec_override=spec_override)
        out["anchored_terms"] = self.anchored_terms(z_c, z_d, batch_ids, spec_override)
        return out

    @torch.no_grad()
    def log_rates(self, z_c, z_d, batch_ids, drop_batch=False):
        """Full and corrected (drop_batch) eta from the anchored components; equals the base decoder's
        'log_term' for param='logmu'. For param='softmax' use the base decoder (its log_softmax is not additive)."""
        if self.base.param != "logmu":
            raise ValueError("additive log-rate reconstruction is exact only for param='logmu'")
        t = self.anchored_terms(z_c, z_d, batch_ids)
        eta = t["intercept"] + t["shared_anchored"] + sum(
            t[f"condition{k}_anchored"] for k in range(len(self.base.spec_maps)))
        return eta if drop_batch else eta + t["batch"]


def decoder_from_checkpoint(model_pt, n_genes=None, interaction=True, param="logmu", gauge_fix=False):
    """Rebuild an AdditiveDecoder from a saved SILVA ``model.pt`` (keys 'Dec.*') without touching the file.

    Shapes (latent dims, hidden width, number of batches, factors) are read from the state dict.
    ``param``/``interaction``/``gauge_fix`` are not recoverable from tensor shapes; defaults match v3.
    Returns (decoder, sha256-of-file).
    """
    state = torch.load(model_pt, map_location="cpu", weights_only=True)
    dec = {k[4:]: v for k, v in state.items() if k.startswith("Dec.")}
    if not dec:
        raise ValueError("no 'Dec.*' keys: not an SILVA checkpoint")
    n_genes = n_genes or dec["gene_bias"].numel()
    d_common = dec["shared_map.0.weight"].shape[1] if "shared_map.0.weight" in dec else dec["shared_map.weight"].shape[1]
    hidden = dec["shared_map.0.weight"].shape[0] if "shared_map.0.weight" in dec else 0
    n_fac = len([k for k in dec if k.startswith("spec_maps.") and k.endswith(".0.weight" if hidden else ".weight")])
    d_diffs = []
    for k in range(n_fac):
        w = dec[f"spec_maps.{k}.0.weight" if hidden else f"spec_maps.{k}.weight"]
        d_diffs.append(w.shape[1] - (d_common if interaction else 0))
    decoder = AdditiveDecoder(d_common, d_diffs, n_genes, dec["batch_emb.weight"].shape[0], param=param,
                              gauge_fix=gauge_fix, hidden=hidden, interaction=interaction)
    decoder.load_state_dict(dec, strict=True)
    return decoder.eval(), file_sha256(model_pt)


@torch.no_grad()
def export_contributions(decoder: AnchoredAdditiveDecoder, z_c, z_d, batch_ids, out_dir, cell_ids=None, genes=None,
                         chunk=2048, extra_manifest=None):
    """Write original AND anchored per-cell per-gene contributions to NEW files (never overwrites).

    Files in out_dir (float32 .npy [cells, genes], log-rate scale, library offset excluded):
      original:  shared_original.npy, condition{k}_original.npy, batch.npy
      anchored:  shared_anchored.npy, condition{k}_anchored.npy, reference{k}.npy (U = h(z_c, z_ref))
      intercept.npy [genes], eta_full is intercept+shared_anchored+sum condition_anchored+batch
    plus manifest.json recording the representation names, reference (kind/value) and max reconstruction error.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; refusing to overwrite")
    out_dir.mkdir(parents=True, exist_ok=True)
    decoder.eval()
    K = len(decoder.base.spec_maps)
    names = ["shared_original", "batch", "shared_anchored"] + [f"{p}{k}_{s}" for k in range(K) for p, s in
            (("condition", "original"), ("condition", "anchored"))] + [f"reference{k}" for k in range(K)]
    z_c, z_d = torch.as_tensor(z_c, dtype=torch.float32), [torch.as_tensor(z, dtype=torch.float32) for z in (z_d if isinstance(z_d, (list, tuple)) else [z_d])]
    b = torch.as_tensor(batch_ids).long()
    n, g = z_c.shape[0], decoder.base.gene_bias.numel()
    mm = {nm: np.lib.format.open_memmap(out_dir / f"{nm}.npy", mode="w+", dtype="float32", shape=(n, g)) for nm in names}
    max_err = 0.0
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        t = decoder.anchored_terms(z_c[sl], [z[sl] for z in z_d], b[sl, None])
        for nm in names:
            mm[nm][sl] = t[nm].cpu().numpy()
        orig = t["intercept"] + t["shared_original"] + sum(t[f"condition{k}_original"] for k in range(K)) + t["batch"]
        anch = t["intercept"] + t["shared_anchored"] + sum(t[f"condition{k}_anchored"] for k in range(K)) + t["batch"]
        max_err = max(max_err, float((orig - anch).abs().max()))
    for m in mm.values():
        m.flush()
    np.save(out_dir / "intercept.npy", decoder.base.gene_bias.detach().cpu().numpy().astype(np.float32))
    if cell_ids is not None:
        np.save(out_dir / "cell_ids.npy", np.asarray(cell_ids).astype(str))
    if genes is not None:
        np.save(out_dir / "genes.npy", np.asarray(genes).astype(str))
    manifest = dict(
        scale="natural-log NB log-rate; library offset log(lib) NOT included",
        representations=dict(original="eta = intercept + shared_original + sum_k condition{k}_original + batch  (v3 definition)",
                             anchored="eta = intercept + shared_anchored + sum_k condition{k}_anchored + batch, "
                                      "shared_anchored = shared_original + sum_k reference{k}, "
                                      "condition{k}_anchored = condition{k}_original - reference{k}"),
        reference=decoder.reference.to_dict(), n_cells=int(n), n_genes=int(g), files=names + ["intercept"],
        max_abs_difference_original_vs_anchored_eta=max_err, decoder_param=decoder.base.param,
        note="anchoring re-labels components; it does not by itself make them the true biological sources")
    manifest.update(extra_manifest or {})
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest
