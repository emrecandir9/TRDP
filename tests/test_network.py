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
        self.assertTrue(torch.allclose(membranes[0], self.net.motor_bias().detach()[None], atol=1e-3))
        self.assertGreater((membranes[1] - membranes[0]).abs().max().item(), 1e-3)

    def test_motor_populations_are_homogeneous_pools(self):
        """Without noise every neuron of a motor population has the same membrane potential: the
        pool shares its visuomotor weights and bias, so its excitability moves as one."""
        self.assertTrue(self.net.pooled)
        self.assertEqual(tuple(self.net.w_vm.weight.shape), (2, 32))
        x = torch.randn(2, 8) * 3
        drive = self.net.drive(x)[:, None].expand(2, 200, -1)
        _, _, state = self.net.run(drive, torch.ones(2, 200), torch.zeros(2, 200), self.net.initial(2),
                                   torch.zeros(2, 200, 32 + 32))
        for population in (slice(0, 16), slice(16, 32)):
            mem = state[3][:, population]
            self.assertTrue(torch.allclose(mem, mem[:, :1].expand_as(mem)))

    def test_spinal_weight_leaves_motoneurons_silent_at_the_movement_limit(self):
        """The declared spinal weight: cortical spikes at the 3 Hz movement limit leave the motoneurons
        essentially silent (<= 0.1 Hz); a weight 20% larger does not."""
        nw, net = self.nw, self.net
        def rate(w, trials=150, steps=3000, n=64):
            g = torch.Generator().manual_seed(1)
            syn = torch.zeros(trials, n); mem = torch.zeros(trials, n); spikes = 0.
            for t in range(steps):
                cortical = (torch.rand(trials, n, generator=g) < nw.SPINAL['at_hz'] / 1000).float().sum(1, keepdim=True)
                syn = net.alpha * syn + w * cortical + net.scale * net.sigma * torch.randn(trials, n, generator=g)
                mem = net.beta * mem + syn
                s = (mem >= 1).float(); mem = mem - s
                spikes += s.mean().item() if t >= 300 else 0.
            return 1000 * spikes / (steps - 300)
        self.assertLessEqual(rate(nw.SPINAL['weight']), 1.5 * nw.SPINAL['silent_hz'])
        self.assertGreater(rate(1.2 * nw.SPINAL['weight']), nw.SPINAL['silent_hz'])

    def test_spinal_readout_counts_recruited_motoneurons(self):
        """With the spinal stage the probe response counts recruited motoneurons, at most one per
        neuron, and the network's saved parameters are unchanged."""
        keys = sorted(self.net.state_dict())
        self.net.readout_stage = 'spinal'
        try:
            n = 6
            drive, gate = self.nw.rest_drive(self.net, n), torch.ones(n)
            g = torch.Generator().manual_seed(2)
            state = self.nw.settle(self.net, drive, gate, 100, g)
            self.assertEqual(len(state), 6)
            response = self.nw.probe_response(self.net, state, drive, gate, 200., g)
            self.assertTrue((response <= self.net.n_motor).all() and (response[:, self.nw.FDI] > 0).all())
        finally:
            self.net.readout_stage = 'cortical'
        self.assertEqual(sorted(self.net.state_dict()), keys)

    def test_probe_response_grows_with_amplitude(self):
        """The evoked response (counted from the probe) does not fall as the probe grows."""
        n = 64
        drive, gate = self.nw.rest_drive(self.net, n), torch.ones(n)
        g = torch.Generator().manual_seed(5)
        state = self.nw.settle(self.net, drive, gate, 200, g)
        means = []
        for amplitude in (.5, 1., 2., 4., 8.):
            g.manual_seed(6)                       # same noise for every amplitude
            means.append(self.nw.probe_response(self.net, state, drive, gate, amplitude, g)[:, self.nw.FDI].mean().item())
        self.assertTrue(all(b >= a for a, b in zip(means, means[1:])), means)

    def test_rate_twin_matches_the_spiking_neuron(self):
        """The twin's unit, given a constant input, emits the rate at which the noisy spiking neuron
        fires under that input (independent simulation, other seed)."""
        net, twin = self.net, self.nw.Network(8, small(), spiking=False, seed=0)
        g = torch.Generator().manual_seed(9)
        for x in (.8, 1., 1.3):
            syn = torch.zeros(2000); mem = torch.zeros(2000); spikes = 0.
            for t in range(1500):
                syn = net.alpha * syn + net.scale * (x + net.sigma * torch.randn(2000, generator=g))
                mem = net.beta * mem + syn
                spike = (mem >= 1).float()
                mem = mem - spike
                spikes += spike.mean().item() if t >= 300 else 0.
            spiking_rate = spikes / 1200
            settled = torch.full((1,), twin.scale * x / (1 - twin.alpha))      # synaptic state at steady input x
            out, _, _ = twin._rate(torch.full((1,), twin.scale * x), settled, torch.full((1,), x))   # membrane at x
            self.assertAlmostEqual(out.item(), spiking_rate, delta=.1 * spiking_rate + 2e-4)

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
        # 16 motor neurons over 100 ms cannot separate the declared 2 vs 5 Hz; the cap has its own test
        cfg = dict(tr.TRAIN, settle_ms=60, second_ms=120, window=(20, 120), batch=16, lr=1e-2, motor_rest_hz=5.,
                   motor_max_hz=40.)
        net = nw.Network(8, small(), seed=0)
        readout, history = tr.train(net, static, movement, cfg=cfg, updates=300)
        task = tr.evaluate_task(net, readout, static, movement, cfg=cfg, repeats=2)
        self.assertGreater(task['auc'], .9)                                # the declared task criterion
        self.assertGreater(task['accuracy'], .85)
        self.assertLess(history[-1]['task'], history[0]['task'])

    def test_interrupted_training_resumes_exactly(self):
        """A run stopped after a checkpoint and resumed ends with the same weights, readout and
        history as an uninterrupted run: the trials and noise drawn are the same."""
        import tempfile
        import network as nw
        import train as tr
        rng = np.random.default_rng(1)
        static = rng.normal(0, 1, (12, 8)).astype(np.float32)
        movement = static + 1
        cfg = dict(tr.TRAIN, settle_ms=20, second_ms=40, window=(5, 40), batch=4, rest_trials=2)
        whole = nw.Network(8, small(), seed=3)
        readout_whole, history_whole = tr.train(whole, static, movement, cfg=cfg, seed=3, updates=6)
        with tempfile.TemporaryDirectory() as d:
            partial = Path(d) / 'run.partial.pt'
            first = nw.Network(8, small(), seed=3)
            tr.train(first, static, movement, cfg=cfg, seed=3, updates=6, resume=partial, every=3)
            self.assertTrue(partial.is_file())                       # checkpoint written at update 3
            state = torch.load(partial, weights_only=False)
            self.assertEqual(state['update'], 3)
            resumed = nw.Network(8, small(), seed=99)                 # fresh object, weights come from the file
            readout_resumed, history_resumed = tr.train(resumed, static, movement, cfg=cfg, seed=3, updates=6,
                                                        resume=partial, every=3)
        for a, b in zip(whole.state_dict().values(), resumed.state_dict().values()):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))
        self.assertAlmostEqual(readout_whole['n0'].item(), readout_resumed['n0'].item(), places=5)
        self.assertEqual([h['update'] for h in history_whole], [h['update'] for h in history_resumed])

    def test_movement_cap_penalises_only_movement_rates_above_it(self):
        """Observation without execution: motor firing on movement frames above motor_max_hz is
        penalised, at or below it is not, and static frames are not touched by the cap."""
        import network as nw
        import train as tr
        net = nw.Network(8, small(), seed=0)
        cfg = dict(tr.TRAIN, settle_ms=10, second_ms=400, window=(0, 400))
        labels = torch.tensor([0., 1.])
        out_v = torch.zeros(2, 410, 32)
        readout = dict(kappa=torch.tensor(1.), n0=torch.tensor(0.))
        def loss(movement_hz, static_hz=cfg['motor_rest_hz']):
            out_m = torch.zeros(2, 410, 32)
            for row, hz in ((0, static_hz), (1, movement_hz)):
                out_m[row, 10::int(1000 / hz)] = 1.                  # regular firing at the given rate
            return tr.objective(net, out_v, out_m, labels, readout, cfg)
        _, at_cap = loss(5.)
        self.assertAlmostEqual(at_cap['fdi_movement_hz'], 5., delta=.5)
        base = loss(5.)[0] - loss(5.)[1]['task']
        above = loss(20.)[0] - loss(20.)[1]['task']
        self.assertGreater(above.item(), base.item() + 10)          # (20 - 5)^2 x 2 populations / 5

    def test_health_diagnostics_run_on_a_small_network(self):
        """diagnose() returns every declared criterion; task probes have one value per trial."""
        import network as nw
        import train as tr
        rng = np.random.default_rng(0)
        static = rng.normal(0, 1, (6, 8)).astype(np.float32)
        movement = static + 2
        net = nw.Network(8, small(), seed=0)
        readout = dict(kappa=torch.tensor(1.), n0=torch.tensor(0.))
        cfg = dict(tr.TRAIN, settle_ms=50, second_ms=120, window=(20, 120))
        report = tr.diagnose(net, readout, static, movement, cfg)
        self.assertEqual(set(report['criteria']), {'task', 'silent_visual', 'fast_visual', 'threshold', 'dose', 'fixation_motor', 'facilitation',
                                                   'background', 'excitability', 'not_saturated', 'readout_not_silent', 'readout_peak_found'})
        task = tr.task_probe(net, report['threshold'] * 1.2, static, movement, cfg, repeats=2)
        self.assertEqual(task['movement']['responses'].shape, (2, 6, 2))
        rates = [r for r, _ in report['excitability_curve']]
        self.assertTrue(all(b > a for a, b in zip(rates, rates[1:])))   # the sweep depolarises FDI step by step




@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Stacked(unittest.TestCase):
    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'needs a CUDA GPU')
    def test_stack_resumes_exactly_on_the_gpu(self):
        """On the GPU, the generators' saved states are restored from the CPU copy (a CUDA RNG state
        must be a CPU ByteTensor): an interrupted stack ends with the uninterrupted stack's weights."""
        import tempfile
        import network as nw
        import stack as st
        import train as tr
        rng = np.random.default_rng(2)
        data = [(s, s + 1, None) for s in (rng.normal(0, 1, (10, 8)).astype(np.float32) for _ in range(3))]
        cfg = dict(tr.TRAIN, settle_ms=20, second_ms=40, window=(5, 40), batch=4, rest_trials=2)
        seeds = [3, 11, 4]
        whole, _, _ = st.train_stack([nw.Network(8, small(), seed=s) for s in seeds], data, seeds, cfg=cfg, updates=6,
                                     device='cuda')
        with tempfile.TemporaryDirectory() as d:
            partial = Path(d) / 'stack.partial.pt'
            st.train_stack([nw.Network(8, small(), seed=s) for s in seeds], data, seeds, cfg=cfg, updates=4, device='cuda',
                           resume=partial, every=3)
            resumed, _, _ = st.train_stack([nw.Network(8, small(), seed=s) for s in seeds], data, seeds, cfg=cfg, updates=6,
                                           device='cuda', resume=partial, every=3)
        for a, b in zip(whole, resumed):
            for x, y in zip(a.state_dict().values(), b.state_dict().values()):
                self.assertTrue(torch.equal(x, y))

    def test_stack_trains_the_same_networks_as_one_at_a_time(self):
        """Three networks with their own seeds, data and (for one) an H1 gain, trained together,
        end with the weights, readouts and histories that train.train gives each alone."""
        import network as nw
        import stack as st
        import train as tr
        rng = np.random.default_rng(2)
        data = []
        for i in range(3):
            static = rng.normal(0, 1, (10 + i, 8)).astype(np.float32)
            gain = (1 + rng.random(10 + i)).astype(np.float32) if i == 2 else None
            data.append((static, static + 1, gain))
        cfg = dict(tr.TRAIN, settle_ms=20, second_ms=40, window=(5, 40), batch=4, rest_trials=2)
        seeds = [3, 11, 4]
        alone = []
        for (static, movement, gain), seed in zip(data, seeds):
            net = nw.Network(8, small(), seed=seed)
            readout, history = tr.train(net, static, movement, cfg=cfg, seed=seed, updates=5, gain=gain)
            alone.append((net, readout, history))
        nets, readouts, histories = st.train_stack([nw.Network(8, small(), seed=s) for s in seeds], data, seeds,
                                                   cfg=cfg, updates=5)
        for (net, readout, history), other, other_readout, other_history in zip(alone, nets, readouts, histories):
            for a, b in zip(net.state_dict().values(), other.state_dict().values()):
                self.assertTrue(torch.allclose(a, b, atol=1e-5), (a - b).abs().max())
            self.assertAlmostEqual(readout['n0'].item(), other_readout['n0'].item(), places=5)
            self.assertAlmostEqual(history[-1]['loss'], other_history[-1]['loss'], places=4)



@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Recurrent(unittest.TestCase):
    """The recurrent variant (docs/model-method.md §6.20)."""

    def recurrent(self):
        return dict(small(), recurrent=True)

    def run_both(self, a, b, spiking=True):
        steps, n = 80, 4
        g = torch.Generator().manual_seed(1)
        x = torch.randn(n, 8, generator=g)
        noise = a.noise(n, steps, g)
        out = []
        for net in (a, b):
            drive = net.drive(x)[:, None].expand(n, steps, -1)
            ones = torch.ones(n, steps)
            out.append(net.run(drive, ones, ones * 0, net.initial(n), noise))
        return out

    def test_starts_as_the_feedforward_network_of_the_same_seed(self):
        import network as nw
        for spiking in (True, False):
            ff, rec = nw.Network(8, small(), spiking, seed=5), nw.Network(8, self.recurrent(), spiking, seed=5)
            for name, value in ff.state_dict().items():
                self.assertTrue(torch.equal(value, rec.state_dict()[name]), name)
            self.assertEqual(rec.w_rec.abs().sum().item(), 0.)
            (v1, m1, s1), (v2, m2, s2) = self.run_both(ff, rec)
            self.assertTrue(torch.equal(v1, v2) and torch.equal(m1, m2))
            self.assertEqual((len(s1), len(s2)), (4, 5))
            self.assertTrue(torch.equal(s2[4], v2[:, -1]))                  # the state carries the last visual output

    def test_recurrent_weights_act_with_one_step_delay_and_no_self_connection(self):
        import network as nw
        ff, rec = nw.Network(8, small(), seed=5), nw.Network(8, self.recurrent(), seed=5)
        with torch.no_grad():
            rec.w_rec.copy_(torch.eye(32) * 5)                              # self-connections only: ignored
        (v1, m1, _), (v2, m2, _) = self.run_both(ff, rec)
        self.assertTrue(torch.equal(v1, v2) and torch.equal(m1, m2))
        with torch.no_grad():
            rec.w_rec.copy_(torch.full((32, 32), .5))
        (v1, _, _), (v2, _, _) = self.run_both(ff, rec)
        self.assertTrue(torch.equal(v1[:, 0], v2[:, 0]))                    # nothing to feed back at the first step
        self.assertGreater(v2.mean().item(), v1.mean().item())              # excitatory recurrence raises the rate

    def test_training_uses_the_recurrent_weights(self):
        import network as nw
        import train as tr
        rng = np.random.default_rng(0)
        static = rng.normal(0, 1, (12, 8)).astype(np.float32)
        cfg = dict(tr.TRAIN, settle_ms=20, second_ms=40, window=(5, 40), batch=4, rest_trials=2)
        net = nw.Network(8, self.recurrent(), seed=0)
        tr.train(net, static, static + 1, cfg=cfg, updates=3)
        self.assertGreater(net.w_rec.abs().sum().item(), 0)
        self.assertEqual(torch.diagonal(net.w_rec).abs().sum().item(), 0.)  # no gradient reaches self-connections

    def test_stack_trains_the_same_recurrent_networks_as_one_at_a_time(self):
        import network as nw
        import stack as st
        import train as tr
        rng = np.random.default_rng(3)
        data = [(s, s + 1, None) for s in (rng.normal(0, 1, (10, 8)).astype(np.float32) for _ in range(2))]
        cfg = dict(tr.TRAIN, settle_ms=20, second_ms=40, window=(5, 40), batch=4, rest_trials=2)
        seeds = [2, 9]
        alone = []
        for (static, movement, gain), seed in zip(data, seeds):
            net = nw.Network(8, self.recurrent(), seed=seed)
            alone.append((net, tr.train(net, static, movement, cfg=cfg, seed=seed, updates=4)))
        nets, readouts, _ = st.train_stack([nw.Network(8, self.recurrent(), seed=s) for s in seeds], data, seeds, cfg=cfg,
                                           updates=4)
        for (net, (readout, _)), other, other_readout in zip(alone, nets, readouts):
            self.assertTrue(other.recurrent)
            for name, a in net.state_dict().items():
                self.assertTrue(torch.allclose(a, other.state_dict()[name], atol=1e-5), name)
            self.assertGreater(other.w_rec.abs().sum().item(), 0)
            self.assertAlmostEqual(readout['n0'].item(), other_readout['n0'].item(), places=5)

    def test_recurrent_share_is_zero_without_recurrent_weights(self):
        import network as nw
        import recurrence as rc
        import train as tr
        rng = np.random.default_rng(1)
        static = rng.normal(0, 1, (6, 8)).astype(np.float32)
        cfg = dict(tr.TRAIN, settle_ms=20, second_ms=40)
        net = nw.Network(8, self.recurrent(), seed=0)
        self.assertEqual(rc.recurrent_share(net, static, static + 1, cfg), 0.)
        with torch.no_grad():
            net.w_rec.fill_(.3)
        self.assertGreater(rc.recurrent_share(net, static, static + 1, cfg), 0.)

    def test_names_and_checkpoint_round_trip(self):
        import tempfile
        import network as nw
        import train as tr
        self.assertEqual(tr.run_name('h2', 'openclip', True, 3, 1, recurrent=True), 'health_openclip_snn_rec_fold3_seed1')
        self.assertEqual(tr.run_name('h0', 'openclip', True, 3, 1), 'health_h0_openclip_snn_fold3_seed1')
        net = nw.Network(8, self.recurrent(), seed=0)
        with torch.no_grad():
            net.w_rec.normal_()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'health_x.pt'
            torch.save(dict(state=net.state_dict(), readout={}, config=net.config, scaler=[[0.] * 8, [1.] * 8]), path)
            path.with_suffix('.json').write_text('{"spiking": true, "seed": 0}')
            loaded = tr.checkpoint(path)[0]
        self.assertTrue(loaded.recurrent and torch.equal(loaded.w_rec, net.w_rec))


@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Campaign(unittest.TestCase):
    def test_folds_inner_sources_and_resume(self):
        """Every source is an inner source once, never its own fold's held-out source, and finished
        runs are skipped."""
        import tempfile
        import campaign as cp
        inner = [cp.inner_source(f) for f in cp.SOURCES]
        self.assertEqual(sorted(inner), list(cp.SOURCES))
        self.assertTrue(all(i != f for i, f in zip(inner, cp.SOURCES)))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for suffix in ('json', 'pt'):
                (d / f'health_h1_openclip_snn_fold3_seed0.{suffix}').write_text('{}')
            done = {(f, s): flag for f, s, _, flag in cp.runs('h1', 'openclip', True, [3, 4], [0], d)}
            self.assertEqual(done, {(3, 0): True, (4, 0): False})
            command = cp.command('h1', 'openclip', False, 4, 0, d)
            self.assertEqual(command[command.index('--inner') + 1], '5')
            self.assertIn('--rate', command)

@unittest.skipIf(torch is None, 'torch and snntorch not installed')
class Variations(unittest.TestCase):
    """Robustness variations (docs/model-method.md §6.21)."""

    def test_each_variation_changes_one_declared_setting(self):
        import network as nw
        import train as tr
        for name, (net, cfg) in tr.VARIATIONS.items():
            self.assertEqual(len(net) + len(cfg), 1, name)
            self.assertTrue(set(net) <= set(nw.CONFIG) and set(cfg) <= set(tr.TRAIN), name)
            changed = {**net, **cfg}
            key = next(iter(changed))
            self.assertNotEqual(changed[key], {**nw.CONFIG, **tr.TRAIN}[key], name)
        self.assertEqual(tr.run_name('h2', 'openclip', True, 2, 0, variation='tm10'), 'health_openclip_snn_tm10_fold2_seed0')

    def test_campaign_passes_the_variation(self):
        import campaign as cp
        command = cp.command('h2', 'openclip', True, 4, 0, Path('.'), variation='lr3e-3')
        self.assertEqual(command[command.index('--variation') + 1], 'lr3e-3')
        names = [n for _, _, n, _ in cp.runs('h2', 'openclip', True, [4], [0], Path('.'), variation='nv128')]
        self.assertEqual(names, ['health_openclip_snn_nv128_fold4_seed0'])

    def test_stack_follows_changed_time_constants(self):
        """A stack of networks with a changed membrane constant trains as each network alone."""
        import network as nw
        import stack as st
        import train as tr
        config = dict(small(), tau_mem=10.)
        rng = np.random.default_rng(4)
        data = [(s, s + 1, None) for s in (rng.normal(0, 1, (10, 8)).astype(np.float32) for _ in range(2))]
        cfg = dict(tr.TRAIN, settle_ms=20, second_ms=40, window=(5, 40), batch=4, rest_trials=2, lr=3e-3)
        seeds = [1, 6]
        alone = []
        for (static, movement, _), seed in zip(data, seeds):
            net = nw.Network(8, config, seed=seed)
            tr.train(net, static, movement, cfg=cfg, seed=seed, updates=3)
            alone.append(net)
        nets, _, _ = st.train_stack([nw.Network(8, config, seed=s) for s in seeds], data, seeds, cfg=cfg, updates=3)
        for a, b in zip(alone, nets):
            self.assertEqual(b.config['tau_mem'], 10.)
            for name, x in a.state_dict().items():
                self.assertTrue(torch.allclose(x, b.state_dict()[name], atol=1e-5), name)


class Claims(unittest.TestCase):
    def test_a_network_is_claimed_once_and_unfinished_files_are_skipped(self):
        import tempfile
        import train as tr
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            old, tr.CLAIMS = tr.CLAIMS, d / 'claims'
            try:
                self.assertTrue(tr.claim('fit_gate', 'health_x'))
                self.assertFalse(tr.claim('fit_gate', 'health_x'))
                self.assertTrue(tr.claim('gate2_sweep', 'health_x'))       # claims are per step
                tr.release('fit_gate', 'health_x')
                self.assertTrue(tr.claim('fit_gate', 'health_x'))
            finally:
                tr.CLAIMS = old
            for name in ('health_a.pt', 'health_a.json', 'health_b.pt', 'health_c.trained.pt', 'health_c.trained.json'):
                (d / name).write_text('{}')
            found = tr.finished_networks(sorted(str(p) for p in d.glob('health_*.pt')))
            self.assertEqual([Path(p).name for p in found], ['health_a.pt'])


if __name__ == '__main__':
    unittest.main()
