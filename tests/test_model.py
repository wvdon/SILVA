"""Core model properties: NB likelihood, additive decomposition, detach z_c in H."""
import sys, unittest
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from silva.model import AdditiveDecoder, nb_log_prob
torch.set_num_threads(2)


class NBTest(unittest.TestCase):
    def test_matches_scipy(self):
        from scipy.stats import nbinom
        x = torch.tensor([[0., 1., 5., 30.]]); mu = torch.tensor([[0.5, 2.0, 4.0, 25.]]); th = torch.tensor([0.7, 3.0, 1.5, 10.])
        got = nb_log_prob(x, mu.log(), th.log()).numpy()
        p = th / (th + mu)   # scipy parameterization: n=theta, p=theta/(theta+mu)
        ref = nbinom.logpmf(x.numpy(), th.numpy(), p.numpy())
        np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


class DecoderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.zc = torch.randn(5, 4, requires_grad=True); self.zd = [torch.randn(5, 2)]; self.b = torch.tensor([[0], [1], [2], [0], [1]])

    def dec(self, detach):
        torch.manual_seed(1)
        d = AdditiveDecoder(4, [2], 9, 3, hidden=8, detach_zc_in_H=detach)
        with torch.no_grad():
            d.batch_emb.weight.normal_(); d.gene_bias.normal_()
        return d

    def test_additive(self):
        d = self.dec(True); out = d(self.zc, self.zd, self.b); t = out["eta_terms"]
        np.testing.assert_allclose(out["log_term"].detach(), (d.gene_bias + t["shared"] + t["spec0"] + t["batch"]).detach(), atol=1e-6)

    def test_drop_batch_removes_only_B(self):
        d = self.dec(True)
        full, nob = d(self.zc, self.zd, self.b), d(self.zc, self.zd, self.b, drop_batch=True)
        np.testing.assert_allclose((full["log_term"] - nob["log_term"]).detach(), full["eta_terms"]["batch"].detach(), atol=1e-6)

    def test_detach_blocks_H_gradient_but_not_forward(self):
        a, b = self.dec(True), self.dec(False)
        np.testing.assert_allclose(a(self.zc, self.zd, self.b)["log_term"].detach(), b(self.zc, self.zd, self.b)["log_term"].detach(), atol=1e-7)
        for dec, expect_flow in ((a, False), (b, True)):
            zc = self.zc.detach().clone().requires_grad_(True)
            dec(zc, self.zd, self.b)["eta_terms"]["spec0"].sum().backward()   # gradient of the H term only
            flows = zc.grad is not None and float(zc.grad.abs().sum()) > 0
            self.assertEqual(flows, expect_flow)

    def test_F_gradient_reaches_zc_when_detached(self):
        d = self.dec(True); zc = self.zc.detach().clone().requires_grad_(True)
        d(zc, self.zd, self.b)["eta_terms"]["shared"].sum().backward()
        self.assertGreater(float(zc.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
