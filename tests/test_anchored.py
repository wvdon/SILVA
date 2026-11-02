"""Reference-anchored decoder: exact re-labelling of the v3 additive decoder (no retraining involved)."""
import sys, tempfile, unittest
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from silva.model import AdditiveDecoder
from silva.anchored import (AnchorReference, AnchoredAdditiveDecoder, decoder_from_checkpoint,
                            export_contributions, file_sha256)
torch.set_num_threads(2)
TOL = dict(atol=2e-6, rtol=2e-6)


def fixture(interaction=True, hidden=8, d_diffs=(2,), param="logmu", gauge_fix=False, n_genes=7, n_batches=3):
    torch.manual_seed(0)
    dec = AdditiveDecoder(4, list(d_diffs), n_genes, n_batches, param=param, gauge_fix=gauge_fix, hidden=hidden,
                          interaction=interaction)
    with torch.no_grad():
        dec.batch_emb.weight.normal_(); dec.gene_bias.normal_()
    zc = torch.randn(6, 4)
    zd = [torch.randn(6, d) for d in d_diffs]
    b = torch.tensor([0, 1, 2, 0, 1, 2])[:, None]
    ref = AnchorReference([np.random.default_rng(1).normal(size=d) for d in d_diffs])
    return dec.eval(), zc, zd, b, ref


class AnchoredTests(unittest.TestCase):
    def test_prediction_equivalence(self):
        for interaction in (True, False):
            for hidden in (0, 8):
                for dd in ((2,), (2, 3)):
                    dec, zc, zd, b, ref = fixture(interaction, hidden, dd)
                    w = AnchoredAdditiveDecoder(dec, ref)
                    out, base = w(zc, zd, b), dec(zc, zd, b)
                    torch.testing.assert_close(out["log_term"], base["log_term"], atol=0, rtol=0)   # untouched outputs
                    for k, v in base["eta_terms"].items():
                        torch.testing.assert_close(out["eta_terms"][k], v, atol=0, rtol=0)
                    t = out["anchored_terms"]
                    eta = t["intercept"] + t["shared_anchored"] + sum(t[f"condition{k}_anchored"] for k in range(len(dd))) + t["batch"]
                    torch.testing.assert_close(eta, base["log_term"], **TOL)
                    torch.testing.assert_close(w.log_rates(zc, zd, b), base["log_term"], **TOL)

    def test_corrected_expression_equivalence(self):
        dec, zc, zd, b, ref = fixture()
        w = AnchoredAdditiveDecoder(dec, ref)
        torch.testing.assert_close(w.log_rates(zc, zd, b, drop_batch=True), dec(zc, zd, b, drop_batch=True)["log_term"], **TOL)

    def test_reference_anchoring(self):
        dec, zc, zd, b, ref = fixture()
        w = AnchoredAdditiveDecoder(dec, ref)
        zref = torch.as_tensor(ref.z_ref[0])[None].expand(len(zc), -1)
        t = w.anchored_terms(zc, [zref.clone()], b)                       # cells sitting exactly at the reference
        torch.testing.assert_close(t["condition0_anchored"], torch.zeros_like(t["condition0_anchored"]), atol=1e-6, rtol=0)
        t = w.anchored_terms(zc, zd, b)
        torch.testing.assert_close(t["shared_anchored"], t["shared_original"] + dec.spec_maps[0](torch.cat([zc, zref], 1)), **TOL)
        torch.testing.assert_close(t["condition0_anchored"], t["condition0_original"] - t["reference0"], **TOL)
        torch.testing.assert_close(t["batch"], dec.batch_emb(b[:, 0]), atol=0, rtol=0)
        # a different reference changes the split but never the sum
        other = AnchoredAdditiveDecoder(dec, AnchorReference([ref.z_ref[0] + 1.0]))
        t2 = other.anchored_terms(zc, zd, b)
        self.assertGreater(float((t2["shared_anchored"] - t["shared_anchored"]).abs().max()), 1e-3)
        torch.testing.assert_close(t2["shared_anchored"] + t2["condition0_anchored"], t["shared_anchored"] + t["condition0_anchored"], **TOL)
        # counterfactual swap keeps the reference fixed
        cf = w.anchored_terms(zc, zd, b, spec_override=[zref.clone()])
        torch.testing.assert_close(cf["condition0_anchored"], torch.zeros_like(cf["condition0_anchored"]), atol=1e-6, rtol=0)

    def test_gauge_fix_softmax_components_sum(self):
        dec, zc, zd, b, ref = fixture(param="softmax", gauge_fix=True)
        t = AnchoredAdditiveDecoder(dec, ref).anchored_terms(zc, zd, b)
        torch.testing.assert_close(t["shared_anchored"] + t["condition0_anchored"], t["shared_original"] + t["condition0_original"], **TOL)
        with self.assertRaises(ValueError):
            AnchoredAdditiveDecoder(dec, ref).log_rates(zc, zd, b)

    def test_checkpoint_loading_and_no_state_change(self):
        dec, zc, zd, b, ref = fixture()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "model.pt"
            torch.save({**{f"Dec.{k}": v for k, v in dec.state_dict().items()}, "diff_prior_means.0": torch.zeros(2, 2)}, path)
            sha = file_sha256(path)
            loaded, sha2 = decoder_from_checkpoint(path)
            self.assertEqual(sha, sha2)
            self.assertEqual(sha, file_sha256(path))                     # file not modified by loading
            torch.testing.assert_close(loaded(zc, zd, b)["log_term"], dec(zc, zd, b)["log_term"], atol=0, rtol=0)
            before = {k: v.clone() for k, v in loaded.state_dict().items()}
            w = AnchoredAdditiveDecoder(loaded, ref)
            self.assertEqual(set(loaded.state_dict()), set(before))      # no keys added
            self.assertFalse(any("z_ref" in k for k in w.state_dict()))  # reference is not part of the checkpoint
            for k, v in loaded.state_dict().items():
                torch.testing.assert_close(v, before[k], atol=0, rtol=0)
            self.assertEqual(sum(p.numel() for p in w.parameters()), sum(p.numel() for p in dec.parameters()))
            # reference round trip
            ref.save(Path(d) / "ref.json")
            back = AnchorReference.load(Path(d) / "ref.json")
            np.testing.assert_array_equal(back.z_ref[0], ref.z_ref[0])
            self.assertEqual(back.kind, ref.kind)

    def test_trained_spec_ref_decoder_rejected(self):
        dec = AdditiveDecoder(4, [2], 5, 2, hidden=4, spec_ref=True)
        with self.assertRaises(ValueError):
            AnchoredAdditiveDecoder(dec, AnchorReference([np.zeros(2)]))
        dec = AdditiveDecoder(4, [2], 5, 2, hidden=4)
        with self.assertRaises(ValueError):
            AnchoredAdditiveDecoder(dec, AnchorReference([np.zeros(3)]))

    def test_control_posterior_mean_definition(self):
        z = np.array([[0., 0.], [2., 2.], [10., 10.], [4., 0.]])
        lab = np.array(["ctrl", "ctrl", "stim", "ctrl"])
        r = AnchorReference.control_posterior_mean(z, lab, "ctrl")
        np.testing.assert_allclose(r.z_ref[0], [2., 2. / 3.], rtol=1e-6)
        self.assertEqual((r.kind, r.n_cells), ("control_posterior_mean", [3]))
        with self.assertRaises(ValueError):
            AnchorReference.control_posterior_mean(z, lab, "missing")

    def test_deterministic_inference(self):
        dec, zc, zd, b, ref = fixture()
        w = AnchoredAdditiveDecoder(dec, ref).eval()
        a, c = w.anchored_terms(zc, zd, b), w.anchored_terms(zc.clone(), [z.clone() for z in zd], b)
        for k in a:
            torch.testing.assert_close(a[k], c[k], atol=0, rtol=0)
        with tempfile.TemporaryDirectory() as d:
            m1 = export_contributions(w, zc, zd, b[:, 0], Path(d) / "a", chunk=4)
            export_contributions(w, zc, zd, b[:, 0], Path(d) / "b", chunk=4)   # same chunking: bit-identical
            export_contributions(w, zc, zd, b[:, 0], Path(d) / "c", chunk=5)   # other chunking: float rounding only
            for nm in m1["files"]:
                x = np.load(Path(d) / "a" / f"{nm}.npy")
                np.testing.assert_array_equal(x, np.load(Path(d) / "b" / f"{nm}.npy"))
                np.testing.assert_allclose(x, np.load(Path(d) / "c" / f"{nm}.npy"), atol=1e-6)

    def test_export_labels_and_refuses_overwrite(self):
        dec, zc, zd, b, ref = fixture()
        w = AnchoredAdditiveDecoder(dec, ref)
        with tempfile.TemporaryDirectory() as d:
            m = export_contributions(w, zc, zd, b[:, 0], Path(d) / "x", genes=[f"g{i}" for i in range(7)])
            self.assertLess(m["max_abs_difference_original_vs_anchored_eta"], 1e-5)
            self.assertIn("anchored", m["representations"]); self.assertEqual(m["reference"]["kind"], ref.kind)
            self.assertIn("library offset", m["scale"])
            F = np.load(Path(d) / "x" / "shared_original.npy"); Fa = np.load(Path(d) / "x" / "shared_anchored.npy")
            U = np.load(Path(d) / "x" / "reference0.npy")
            np.testing.assert_allclose(Fa, F + U, atol=2e-6)
            with self.assertRaises(FileExistsError):
                export_contributions(w, zc, zd, b[:, 0], Path(d) / "x")


if __name__ == "__main__":
    unittest.main()
