"""CPU checks for the spiking network, its probe readout and training (small networks)."""
from pathlib import Path
import sys
import unittest

import numpy as np

try:
    import torch
    import snntorch  # noqa: F401
except ImportError:   # the review environment needs no torch
    torch = None
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))



def small():
    import network as nw
    return dict(nw.CONFIG, n_visual=32, n_motor=16)


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Network(unittest.TestCase):
    def setUp(self):
        import network as nw
        self.nw = nw
        self.net = nw.Network(8, small(), seed=0)

    def test_shapes_and_spikes(self):
        b, t = 3, 50
        drive = self.net.drive(torch.randn(b, 8))[:, None].expand(b, t, -1)
        g = torch.Generator().manual_seed(0)
        out_v, out_m, state = self.net.run(drive, torch.ones(b, t), torch.zeros(b, t), self.net.initial(b),
                                           self.net.noise(b, t, g))
        self.assertEqual(out_v.shape, (b, t, 32))
        self.assertEqual(out_m.shape, (b, t, 32))
        self.assertTrue(set(out_v.unique().tolist()) <= {0., 1.})                # spikes are binary
        self.assertEqual(len(state), 4)

    def test_decay_constants(self):
        self.assertAlmostEqual(self.net.alpha, np.exp(-1 / 5))
        self.assertAlmostEqual(self.net.beta, np.exp(-1 / 20))

    def test_units_and_resting_noise(self):
        """A constant input x settles at membrane x, and noise alone gives membrane SD 0.1."""
        net, n = self.net, 4000
        syn = torch.zeros(1); mem = torch.zeros(1)
        for _ in range(2000):
            syn = net.alpha * syn + net.scale * .7
            mem = net.beta * mem + syn
        self.assertAlmostEqual(mem.item(), .7, places=4)
        g = torch.Generator().manual_seed(0)
        eps = net.sigma * torch.randn(n, 2000, generator=g)
        syn = torch.zeros(n); mem = torch.zeros(n)
        for k in range(2000):
            syn = net.alpha * syn + net.scale * eps[:, k]
            mem = net.beta * mem + syn
        self.assertAlmostEqual(mem.std().item(), .1, delta=.005)

    def test_twin_branch_isolates_the_probe(self):
        """With zero amplitude the probe-minus-twin response is exactly 0: same state, same noise."""
        x = torch.randn(4, 8)
        drive, gate = self.net.drive(x), torch.ones(4)
        g = torch.Generator().manual_seed(1)
        state = self.nw.settle(self.net, drive, gate, 100, g)
        zero = self.nw.probe_response(self.net, state, drive, gate, 0., g)
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        strong = self.nw.probe_response(self.net, state, drive, gate, 50., g)
        self.assertTrue((strong[:, self.nw.FDI] > 0).all())

    def test_gate_scales_the_visuomotor_input(self):
        """The gate multiplies the visuomotor input: with gate 0 the motor membrane sits at its
        bias; with gate 1 it moves by the visual drive (subthreshold, so measured on the membrane)."""
        x = torch.randn(1, 8) * 3
        drive = self.net.drive(x)[:, None].expand(1, 300, -1)
        noise = torch.zeros(1, 300, 32 + 32)
        membranes = []
        for gate in (0., 1.):
            _, _, state = self.net.run(drive, torch.full((1, 300), gate), torch.zeros(1, 300), self.net.initial(1), noise)
            membranes.append(state[3])
        self.assertTrue(torch.allclose(membranes[0], self.net.b_m.detach()[None], atol=1e-3))
        self.assertGreater((membranes[1] - membranes[0]).abs().max().item(), 1e-3)

    def test_threshold_and_rate_twin(self):
        a = self.nw.threshold(self.net, trials=12, halvings=8)
        self.assertTrue(0 < a < 20)
        twin = self.nw.Network(8, small(), spiking=False, seed=0)
        b = 2
        drive = twin.drive(torch.randn(b, 8))[:, None].expand(b, 30, -1)
        _, out_m, _ = twin.run(drive, torch.ones(b, 30), torch.zeros(b, 30), twin.initial(b),
                               twin.noise(b, 30, torch.Generator().manual_seed(0)))
        self.assertTrue((out_m >= 0).all() and not set(out_m.unique().tolist()) <= {0., 1.})   # graded rates


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Training(unittest.TestCase):
    def test_learns_a_separable_movement(self):
        """Movement frames differ from static frames along one input direction: the FDI
        population learns to detect it, and ADM is held at equal rates."""
        import network as nw
        import train as tr
        rng = np.random.default_rng(0)
        static = rng.normal(0, 1, (40, 8)).astype(np.float32)
        movement = static.copy()
        movement[:, 0] += 3
        cfg = dict(tr.TRAIN, settle_ms=60, second_ms=120, window=(20, 120), batch=16, lr=3e-3)
        net = nw.Network(8, small(), seed=0)
        readout, history = tr.train(net, static, movement, cfg=cfg, updates=150)
        accuracy, _ = tr.evaluate_task(net, readout, static, movement, cfg=cfg, repeats=2)
        self.assertGreater(accuracy, .85)
        self.assertLess(history[-1]['task'], history[0]['task'])


if __name__ == '__main__':
    unittest.main()
