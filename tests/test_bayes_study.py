import json
from pathlib import Path
import unittest

import numpy as np

import prepare_bayes_study as study


class DynamicCausalityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(76)
        self.z = rng.normal(size=(120, 3))
        self.y = rng.normal(size=120)
        self.initial = study.initial_state(self.z[:40], self.y[:40])

    def run_filter(self, z=None, y=None):
        return study.dynamic_forecast(self.z if z is None else z, self.y if y is None else y,
                                      46, 39, *self.initial, .995)

    def test_unmatured_labels_cannot_change_predictions(self):
        original = self.run_filter()
        altered = self.y.copy()
        altered[65:] += 1000
        changed = self.run_filter(y=altered)
        # Origin 65 is not observable before close 71.
        np.testing.assert_array_equal(original['mean'][:71 - 46], changed['mean'][:71 - 46])
        self.assertTrue(np.any(original['mean'][71 - 46:] != changed['mean'][71 - 46:]))

    def test_appended_future_observations_cannot_change_history(self):
        full = self.run_filter()
        prefix = study.dynamic_forecast(self.z[:90], self.y[:90], 46, 39, *self.initial, .995)
        np.testing.assert_array_equal(full['mean'][:44], prefix['mean'])
        np.testing.assert_array_equal(full['std'][:44], prefix['std'])

    def test_updates_are_delayed_and_nonoverlapping(self):
        out = self.run_filter()
        origins = [u['origin_index'] for u in out['updates']]
        self.assertGreaterEqual(origins[0] - 39, 6)
        self.assertTrue(all(b - a >= 6 for a, b in zip(origins, origins[1:])))
        self.assertTrue(all(u['origin_index'] + 6 == u['asof_index'] for u in out['updates']))
        self.assertGreaterEqual(np.linalg.eigvalsh(out['final_covariance']).min(), -1e-10)
        self.assertTrue(np.all(out['std'] > 0))


class StudyArtifactTests(unittest.TestCase):
    def test_training_input_contains_no_future_labels(self):
        if not (study.CACHE / 'bart_inputs.npz').exists():
            self.skipTest('Local optional training cache is not distributed with the website')
        data = np.load(study.CACHE / 'bart_inputs.npz')
        meta = json.loads((study.CACHE / 'study_inputs.json').read_text(encoding='utf-8'))
        self.assertEqual(set(data.files), {'x_train', 'y_train', 'x_score', 'train_indices', 'n_seed', 'score_dates'})
        self.assertEqual(data['x_train'].shape, (765, 28))
        self.assertEqual(meta['last_train_label_exit'], '2026-06-30')
        self.assertLessEqual(meta['last_train_signal'], '2026-06-22')
        self.assertEqual(meta['discount'], min((r for r in meta['validation'] if r['discount'] < 1),
                                               key=lambda r: r['rmse'])['discount'])
        self.assertFalse(any(f['clip_enabled'] for f in meta['scaler_model']['features'][:8]))

    def test_report_reconciles_and_preserves_production(self):
        path = study.ROOT / study.OUTPUT
        if not path.exists():
            self.skipTest('Report not built yet')
        data = json.loads(path.read_text(encoding='utf-8'))
        self.assertTrue(data['meta']['factor_selection_lookahead'])
        self.assertEqual(set(data['meta']['protected_hashes']), set(study.PROTECTED))
        self.assertEqual(data['meta']['last_train_label_exit'], '2026-06-30')
        self.assertFalse(any(f['clip_enabled'] for f in data['meta']['preprocessing'][:8]))
        n = len(data['dates'])
        self.assertEqual(sum(y is not None for y in data['target']), n - 6)
        for model in data['variants'].values():
            for mode, item in model['modes'].items():
                sim = item['sim']
                self.assertEqual(len(sim['equity']), n)
                self.assertAlmostEqual(sum(sim['daily_pnl']), sim['metrics']['total'], places=4)
                self.assertAlmostEqual(sum(m['pnl'] for m in item['months'].values()), sim['metrics']['total'], places=4)
                self.assertAlmostEqual(sum(t['pnl'] for t in sim['trades']), sim['metrics']['total'], places=4)
                self.assertEqual(sim['position'][0], 0)
                if mode == 'cold':
                    self.assertTrue(all(p is None for p in item['gate']['percentile'][:60]))
            if 'predictive_std' in model:
                self.assertTrue(all(s > 0 for s in model['predictive_std']))
                self.assertTrue(all(0 <= p <= 1 for p in model['probability_positive']))
                self.assertTrue(all(a < b for a, b in zip(model['interval80']['low'], model['interval80']['high'])))


if __name__ == '__main__':
    unittest.main()
