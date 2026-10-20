import contextlib
import copy
import io
import json
import unittest

import numpy as np

import prepare_threshold_study as study


class ThresholdMechanicsTests(unittest.TestCase):
    def test_boundaries_missing_and_original_forecasts_are_preserved(self):
        original = {'percentile': [None, 39.9, 40., 60., 60.1], 'history_n': [0, 60, 61, 62, 63]}
        snapshot = json.dumps(original)
        gate = study.mapped_gate(original, 40, 60)
        self.assertEqual(gate['signal'].tolist(), [0, -1, 0, 0, 1])
        self.assertEqual(snapshot, json.dumps(original))

    def test_trade_win_rate_excludes_open_and_preserves_zero_trades(self):
        trades = [{'closed': True, 'pnl': .1}, {'closed': True, 'pnl': -.6},
                  {'closed': True, 'pnl': 0}, {'closed': False, 'pnl': 3.}]
        stats = study.trade_stats(trades)
        self.assertAlmostEqual(stats['win_rate'], 1 / 3)
        self.assertEqual(stats['closed_n'], 3)
        self.assertEqual(stats['breakeven'], 1)
        self.assertAlmostEqual(stats['closed_pnl'], -.5)
        self.assertAlmostEqual(stats['floating_pnl'], 3.)

    def test_selector_does_not_choose_highest_win_rate_or_too_few_trades(self):
        rows = [{'band': '45_55', 'metrics': {'closed_n': 10, 'mean_pnl': .1,
                 'closed_pnl': 1., 'max_drawdown': .2, 'win_rate': .55}},
                {'band': '30_70', 'metrics': {'closed_n': 10, 'mean_pnl': -.1,
                 'closed_pnl': -1., 'max_drawdown': 1.2, 'win_rate': .9}},
                {'band': '35_65', 'metrics': {'closed_n': 2, 'mean_pnl': 1.,
                 'closed_pnl': 2., 'max_drawdown': .1, 'win_rate': 1.}}]
        self.assertEqual(study.choose_band(rows)['band'], '45_55')
        self.assertIsNone(study.choose_band(rows[1:])['band'])


class ThresholdArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = study.read(study.ROOT / study.OUTPUT)

    def test_validation_and_test_have_no_cross_boundary_labels(self):
        json.dumps(self.data, allow_nan=False)
        val = self.data['periods']['validation2025']
        test = self.data['periods']['test2026']
        self.assertTrue(all(d.startswith('2025-') for d in val['dates']))
        self.assertTrue(all(d.startswith('2026-') for d in test['dates']))
        self.assertTrue(all(y is None for y in val['target'][-6:]))
        self.assertTrue(all(y is None for y in test['target'][-6:]))
        self.assertEqual(self.data['validation_bart_diagnostics']['train_through'], '2025-06-30')
        local = study.CACHE / 'linear_inputs.npz'
        if local.exists():
            self.assertTrue(np.isnan(np.load(local)['y'][-6:]).all())

    def test_all_accounts_and_sides_reconcile(self):
        for model in self.data['models'].values():
            for period in model['periods'].values():
                for mode, rules in period.items():
                    for rule, rows in rules.items():
                        self.assertEqual(len(rows), 4)
                        for row in rows:
                            sim, m = row['sim'], row['metrics']
                            self.assertAlmostEqual(sum(sim['daily_pnl']), m['total'], places=4)
                            self.assertAlmostEqual(m['closed_pnl'] + m['floating_pnl'], m['total'], places=4)
                            self.assertAlmostEqual(sum(row['months'].values()), m['total'], places=4)
                            self.assertAlmostEqual(sum(s['closed_pnl'] + s['floating_pnl']
                                                       for s in row['by_side'].values()), m['total'], places=4)
                            self.assertEqual(sum(s['closed_n'] for s in row['by_side'].values()), m['closed_n'])
                            if rule == 'fixed':
                                self.assertTrue(all(t['holding_sessions'] == 6 for t in sim['trades'] if t['closed']))
                            self.assertEqual(sim['position'][0], 0)

    def test_original_dynamic_40_60_is_exactly_reproduced(self):
        source = study.read(study.ROOT / 'curve_timing_bayes.json')
        for key, model in self.data['models'].items():
            for mode in ('cold', 'seeded'):
                row = next(r for r in model['periods']['test2026'][mode]['dynamic'] if r['band'] == '40_60')
                for field in ('position', 'daily_pnl', 'equity', 'drawdown'):
                    np.testing.assert_allclose(row['sim'][field], source['variants'][key]['modes'][mode]['sim'][field], atol=1e-12)

    def test_selection_only_uses_validation_fixed_cold(self):
        for key, original_model in self.data['models'].items():
            model = copy.deepcopy(original_model)
            selected = study.choose_band(model['periods']['validation2025']['cold']['fixed'])
            self.assertEqual(selected, self.data['selection'][key])
            # Perturb all 2026 metrics: selection must remain exactly unchanged.
            for row in model['periods']['test2026']['cold']['fixed']:
                row['metrics']['total'] += 1e6
            self.assertEqual(selected, study.choose_band(model['periods']['validation2025']['cold']['fixed']))

    def test_rebuild_is_idempotent_and_does_not_change_models(self):
        before = study.digest(study.ROOT / study.OUTPUT)
        protected = {n: study.digest(study.ROOT / n) for n in study.PROTECTED}
        with contextlib.redirect_stdout(io.StringIO()):
            study.build_report()
        self.assertEqual(before, study.digest(study.ROOT / study.OUTPUT))
        self.assertEqual(protected, {n: study.digest(study.ROOT / n) for n in study.PROTECTED})


if __name__ == '__main__':
    unittest.main()
