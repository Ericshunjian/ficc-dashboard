import unittest
from unittest.mock import patch

import numpy as np

import prepare_ridge_retrain_202606 as study
from prepare_direction_screen import baseline_features, load_candidates
from prepare_strategy_model import make_offset_labels
from prepare_curve_timing_walkforward import completed_indices


class ExpandedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = study.read(study.OUTPUT)
        cls.models = study.read(study.MODELS)

    def test_only_completed_pre_july_labels_are_in_training(self):
        _, dates, curve, _ = load_candidates()
        level = 3 * curve['T主力'] - curve['TL主力']
        cutoff = dates.index('2026-06-30')
        train = completed_indices(level, make_offset_labels(level, 6), cutoff, 6)
        self.assertTrue(np.all(train + 6 <= cutoff))
        self.assertEqual(len(train), self.models['meta']['train_n'])
        self.assertEqual(dates[train[-1]], self.models['meta']['last_train_signal'])
        self.assertEqual(dates[train[-1] + 6], '2026-06-30')
        self.assertEqual(self.data['dates'][0], '2026-07-01')

    def test_fixed_factors_and_unclipped_spreads(self):
        sets = study.read('curve_timing_capacity_factor_sets.json')['sets']
        for arm, model in self.models['models'].items():
            self.assertEqual([f['id'] for f in model['features'][8:]], sets[arm])
            self.assertTrue(all(not f['clip_enabled'] for f in model['features'][:8]))
        self.assertIn('全历史筛选', self.data['meta']['caveat'])

    def test_cold_and_seeded_gates_never_read_future_forecasts(self):
        days = self.data['dates']
        level = self.data['level']
        pred = np.array(self.data['variants']['expanded20']['prediction'])
        for seed in (None, np.arange(250) / 100):
            g, _ = study.gate_and_simulate(days, level, pred, seed)
            changed = pred.copy(); changed[-2:] = 1e9
            other, _ = study.gate_and_simulate(days, level, changed, seed)
            self.assertEqual(g['percentile'][:-2], other['percentile'][:-2])
            self.assertEqual(g['signal'][:-2], other['signal'][:-2])
            if seed is None:
                self.assertEqual(g['percentile'][:60], [None] * 60)

    def test_accounting_and_months_reconcile_without_initial_positions(self):
        for variant in self.data['variants'].values():
            for mode, item in variant['modes'].items():
                sim = item['sim']
                self.assertEqual(sim['position'][0], 0)
                self.assertEqual(len(sim['equity']), len(self.data['dates']))
                np.testing.assert_allclose(np.cumsum(sim['daily_pnl']), sim['equity'], atol=1e-9)
                self.assertAlmostEqual(sum(t['pnl'] for t in sim['trades']), sim['equity'][-1], places=9)
                self.assertAlmostEqual(sum(m['pnl'] for m in item['months'].values()), sim['equity'][-1], places=9)
                for month in item['months'].values():
                    self.assertAlmostEqual(month['pnl'], month['long_pnl'] + month['short_pnl'], places=9)
                for trade in sim['trades']:
                    self.assertGreaterEqual(trade['signal_date'], '2026-07-01')
                    self.assertGreaterEqual(trade['entry_session'], '2026-07-02')

    def test_prediction_metrics_exclude_unfinished_six_day_labels(self):
        self.assertEqual(self.data['target'][-6:], [None] * 6)
        for variant in self.data['variants'].values():
            self.assertEqual(variant['forecast']['n'], self.data['meta']['test_completed_n'])

    def test_rerun_uses_saved_model_and_never_touches_production(self):
        names = study.PROTECTED + (study.MODELS, study.OUTPUT)
        before = {n: study.sha(n) for n in names}
        with patch.object(study, 'fit_model', side_effect=AssertionError('Unexpected refit')):
            study.build()
        self.assertEqual(before, {n: study.sha(n) for n in names})


if __name__ == '__main__':
    unittest.main()
