"""Frozen-model inference, archive continuity and current-date chart accounting."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import prepare_curve_comparison_daily as comparison
import prepare_ridge_daily as daily
from prepare_direction_screen import baseline_features, load_candidates
from research_inputs import research_input


class ComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = daily.read(daily.OUTPUT)
        cls.current = cls.report['comparison']
        cls.source = daily.read('curve_timing_nonlinear.json')
        cls.frozen = daily.read(comparison.MODELS)

    def test_all_twelve_paths_and_all_chart_dates_are_current(self):
        dates = self.current['daily']['dates']
        self.assertEqual(dates, [r['date'] for r in self.report['records']])
        self.assertEqual(dates[-1], self.report['meta']['signal_date'])
        self.assertTrue(self.current['meta']['healthy'])
        self.assertEqual(len(comparison.KEYS), 12)
        for rule in ('dynamic', 'fixed'):
            self.assertEqual(set(self.current['strategies'][rule]), set(comparison.KEYS))
            for sim in self.current['strategies'][rule].values():
                for k in ('position', 'equity', 'daily_pnl', 'drawdown'):
                    self.assertEqual(len(sim[k]), len(dates))
                np.testing.assert_allclose(np.cumsum(sim['daily_pnl']), sim['equity'], atol=1e-9)
                self.assertAlmostEqual(sum(t['pnl'] for t in sim['trades']), sim['equity'][-1], places=8)
                self.assertAlmostEqual(-min(sim['drawdown']), sim['metrics']['max_drawdown'], places=5)

    def test_existing_daily_authorities_and_accounting_are_reused_exactly(self):
        for n in (20, 30):
            self.assertEqual(self.current['strategies']['dynamic'][f'ridge{n}'], self.report['strategies'][f'f{n}'])
        for n in (20, 30, 40):
            self.assertEqual(self.current['strategies']['dynamic'][f'logistic{n}'], self.report['logistic']['strategies'][f'f{n}'])
        for i, r in enumerate(self.report['records']):
            for n in (20, 30):
                rec = self.current['records'][i]['models'][f'ridge{n}']
                for k in ('prediction', 'percentile', 'signal', 'status'):
                    self.assertEqual(rec[k], r['models'][f'f{n}'][k])

    def test_original_research_forecasts_and_path_prefixes_are_unchanged(self):
        size = len(self.source['daily']['dates'])
        log = daily.read('curve_timing_logistic.json')
        for key in comparison.KEYS:
            source = log if key.startswith('logistic') else self.source
            self.assertEqual(self.current['daily']['prediction'][key][:size], source['daily']['prediction'][key])
            for rule in ('dynamic', 'fixed'):
                for field in ('position', 'equity', 'daily_pnl', 'drawdown'):
                    np.testing.assert_allclose(self.current['strategies'][rule][key][field][:size],
                                               source['strategies'][rule][key][field], atol=1e-9, rtol=0)

    def test_supplemental_backfill_is_labelled_and_final_plan_is_not_a_trade(self):
        for row in self.current['records']:
            if self.current['meta']['research_asof'] < row['date'] <= self.current['meta']['supplemental_backfill_through']:
                for key in ('ridge40', 'tree20', 'tree30', 'tree40', 'xgb20', 'xgb30', 'xgb40'):
                    self.assertEqual(row['models'][key]['origin'], 'supplemental-backfill')
                    self.assertEqual(row['models'][key]['status'], 'deployment-backfill')
        dates = self.current['daily']['dates']
        for sim in self.current['strategies']['dynamic'].values():
            for t in sim['trades']:
                self.assertIn(t['entry_session'], dates)
                if not t['closed']:
                    self.assertIsNone(t['exit_session'])

    def test_only_mature_labels_enter_forecast_metrics(self):
        level = np.array(self.current['daily']['level'])
        y = np.array(self.current['daily']['target'], float)
        self.assertTrue(np.isnan(y[-6:]).all())
        np.testing.assert_allclose(y[:-6], level[6:] - level[:-6], atol=1e-12)
        for item in self.current['forecast'].values():
            self.assertLessEqual(item['test']['n'], len(y) - 6)

    def test_frozen_tree_booster_and_ridge_reproduce_original_predictions(self):
        snapshot = research_input('factor_library.json', self.source['meta']['input_hashes']['factor_library.json'])
        with patch('prepare_factor_checkup.LIB_PATH', str(snapshot)), patch('prepare_strategy_model.BASE', str(snapshot.parent)):
            _, dates, curve, candidates = load_candidates()
            baselines, _ = baseline_features(3 * curve['T主力'] - curve['TL主力'], curve, dates)
        features = {f['id']: f['values'] for f in candidates + baselines}
        axis, boosters = {d: i for i, d in enumerate(dates)}, {}
        for key, model in self.frozen['models'].items():
            for i, day in enumerate(self.source['daily']['dates']):
                expected = self.source['daily']['prediction'][key][i]
                if expected is None:
                    continue
                values = [features[f['id']][axis[day]] for f in model['preprocessing']]
                self.assertAlmostEqual(comparison.predict(model, values, boosters, key), expected, places=10,
                                       msg=f'{key} {day}')

    def test_repeated_update_never_refits_or_changes_issued_forecasts(self):
        before = copy.deepcopy(self.current['records'])
        names = comparison.AUTHORITIES + comparison.SOURCES + (daily.OUTPUT,)
        hashes = {n: daily.sha(n) for n in names}
        with patch('prepare_curve_timing_capacity.fit_model', side_effect=AssertionError('Unexpected Ridge fit')), \
             patch('prepare_direction_screen.screen_snapshot', side_effect=AssertionError('Unexpected selection')), \
             patch('sklearn.tree.DecisionTreeRegressor.fit', side_effect=AssertionError('Unexpected tree fit')):
            result, changes = daily.build()
        self.assertEqual(changes['new_comparison_dates'], [])
        self.assertFalse(changes['signal_changed'])
        self.assertEqual(result['comparison']['records'], before)
        self.assertEqual(hashes, {n: daily.sha(n) for n in names})

    def test_model_version_change_is_rejected(self):
        old = copy.deepcopy(self.current)
        old['meta']['frozen_hashes'][comparison.MODELS] = 'changed'
        from datetime import datetime
        with self.assertRaisesRegex(ValueError, 'Frozen comparison'):
            comparison.build_archive(self.report, old, {}, [], {}, set(), set(), datetime.now())


if __name__ == '__main__':
    unittest.main()
