"""End-to-end smoke test on a tiny synthetic data set (CPU, a few seconds)."""
import sys, unittest
from pathlib import Path
import anndata as ad, numpy as np, pandas as pd, scipy.sparse as sp, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import silva


def toy(n_per=60, genes=40, seed=0):
    rng = np.random.default_rng(seed); rows, obs = [], []
    for d in range(3):
        for c in ("ctrl", "stim"):
            for t in range(2):
                mu = np.exp(rng.normal(0, .3, genes) + (t * 1.2 * (np.arange(genes) < 10)) + (c == "stim") * .8 * (np.arange(genes) > 30))
                rows.append(rng.poisson(mu * 5, size=(n_per, genes))); obs += [(f"d{d}", c, f"t{t}")] * n_per
    X = sp.csr_matrix(np.vstack(rows).astype(np.float32))
    a = ad.AnnData(X, obs=pd.DataFrame(obs, columns=["donor_id", "condition", "cell_type"], index=[f"c{i}" for i in range(len(obs))]))
    a.var_names = [f"g{i}" for i in range(genes)]; a.layers["counts"] = X.copy(); return a


class SmokeTest(unittest.TestCase):
    def test_fit(self):
        torch.set_num_threads(2); a = toy()
        model, res, zc, zd = silva.fit(a, zc_dim=4, zd_dim=2, dec_hidden=16, max_epochs=3, device="cpu", verbose=False)
        self.assertEqual(zc.shape, (a.n_obs, 4)); self.assertEqual(zd.shape, (a.n_obs, 2))
        self.assertTrue(np.isfinite(zc).all() and np.isfinite(zd).all()); self.assertTrue(model.Dec.detach_zc_in_H)
        self.assertTrue(np.isfinite(res["best_val_total"]))

    def test_rejects_non_counts(self):
        a = toy(); a.layers["counts"] = a.layers["counts"].astype(np.float32) * 0.5
        with self.assertRaises(ValueError): silva.fit(a, device="cpu", max_epochs=1, verbose=False)


if __name__ == "__main__":
    unittest.main()
