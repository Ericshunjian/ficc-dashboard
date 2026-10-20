import copy
import json
import unittest

import numpy as np

import prepare_temporal_study as study
from prepare_curve_timing_capacity import TypeAwareScaler
from prepare_curve_timing_walkforward import completed_indices
from prepare_strategy_model import make_offset_labels


class TemporalConstructionTests(unittest.TestCase):
    def setUp(self):
        self.level = np.arange(80, dtype=float) ** 2
        self.original = [{'id': f'baseline|dummy{i}' if i < 8 else f'tech|test|{i}',
                          'label': str(i), 'values': np.arange(80, dtype=float) * (i + 1)} for i in range(28)]

    def test_lags_changes_and_volatility_are_exact_and_do_not_wrap(self):
        fs = study.temporal_features(self.level, self.original)
        self.assertEqual(len(fs), 92)
        self.assertEqual(len({f['id'] for f in fs}), 92)
        by = {f['id']: f['values'] for f in fs}
        self.assertTrue(np.isnan(by['baseline|spread_mom3'][:3]).all())
        self.assertEqual(by['baseline|spread_mom3'][30], self.level[30] - self.level[27])
        self.assertTrue(np.isnan(by['baseline|spread_vol20'][:20]).all())
        self.assertAlmostEqual(by['baseline|spread_vol20'][30], np.std(np.diff(self.level[10:31]), ddof=0))
        self.assertEqual(by['tech|test|8|history_lag1'][30], self.original[8]['values'][29])
        self.assertEqual(by['tech|test|8|history_lag5'][30], self.original[8]['values'][25])
        self.assertEqual(by['tech|test|8|history_change5'][30], self.original[8]['values'][30] - self.original[8]['values'][25])

    def test_future_rows_cannot_change_any_past_feature(self):
        reference = study.matrix(study.temporal_features(self.level, self.original))
        altered = copy.deepcopy(self.original)
        for f in altered:
            f['values'][50:] += 10000
        future_level = self.level.copy()
        future_level[50:] *= -100
        changed = study.matrix(study.temporal_features(future_level, altered))
        np.testing.assert_array_equal(reference[:50], changed[:50])
        prefix = [dict(f, values=f['values'][:50]) for f in self.original]
        np.testing.assert_array_equal(reference[:50], study.matrix(study.temporal_features(self.level[:50], prefix)))

    def test_missing_history_is_not_filled_during_feature_construction(self):
        self.original[8]['values'][25] = np.nan
        self.level[25] = np.nan
        by = {f['id']: f['values'] for f in study.temporal_features(self.level, self.original)}
        self.assertTrue(np.isnan(by['tech|test|8|history_lag5'][30]))
        self.assertTrue(np.isnan(by['tech|test|8|history_change5'][30]))
        self.assertTrue(np.isnan(by['baseline|spread_vol20'][30]))

    def test_parameter_selector_uses_error_not_trading_pnl(self):
        rows = [{'alpha': 15, 'rmse_exact': .5, 'pnl': 100},
                {'alpha': 50, 'rmse_exact': .4, 'pnl': -100}]
        self.assertEqual(study.choose(rows, 'alpha'), 50)
        rows[0]['rmse_exact'] = .4
        self.assertEqual(study.choose(rows, 'alpha'), 50)
        with self.assertRaises(ValueError):
            study.fit_training(['2026-01-01'], np.ones(1), np.ones((1, 28)), np.ones((1, 92)), [], [])


class TemporalArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = study.read(study.OUTPUT)
        cls.bundle = study.read(study.MODELS)

    def test_training_validation_boundaries_and_no_test_fitted_preprocessing(self):
        d, level, xb, _, base, cut, train = study.load_data()
        fs = study.temporal_features(level, base)
        xt = study.matrix(fs)
        meta, val = self.bundle['meta'], self.bundle['validation']
        self.assertEqual(meta['train_n'], 649)
        self.assertEqual(meta['last_train_signal'], '2025-12-23')
        self.assertEqual(meta['last_train_label_exit'], '2025-12-31')
        self.assertEqual(val['last_inner_label_exit'], '2025-06-30')
        self.assertEqual(val['start'], '2025-07-01')
        self.assertEqual(val['last_label_exit'], '2025-12-31')
        self.assertEqual(val['n'], 120)
        self.assertEqual(meta['alpha_temporal'], study.choose(val['alphas'], 'alpha'))
        self.assertEqual(meta['discount'], study.choose(val['discounts'], 'discount'))
        self.assertEqual(meta['rounds'], int(np.argmin(val['xgb_rmse_path'])) + 1)
        scaler = TypeAwareScaler(fs).fit(xt[train])
        record = self.bundle['models']['ridge_temporal']
        np.testing.assert_array_equal(scaler.transform(xt), study.transform_record(record, xt))
        self.assertEqual(meta['input_sha256'], study.training_digest(xb[:cut+1], xt[:cut+1],
                         make_offset_labels(level[:cut+1], 6), d[:cut+1]))
        clipped = {f['id'] for f in record['features'] if f['clip_enabled']}
        self.assertFalse(any(i.startswith(('baseline|', 'tech|', 'curve|')) for i in clipped))
        self.assertEqual(self.report['quality']['test_missing_cells'], 0)
        self.assertEqual(self.report['meta']['models_sha256'], study.sha(study.MODELS))

    def test_ridge_reference_exact_and_all_accounts_reconcile(self):
        r = self.report
        original = study.read('curve_timing_bayes.json')
        self.assertEqual(set(r['variants']), set(study.KEYS))
        self.assertEqual(r['dates'], original['dates'])
        self.assertEqual(r['target'][-6:], [None] * 6)
        self.assertEqual(len(r['features']), 92)
        self.assertTrue(all(day.startswith('2026') for day in r['dates']))
        np.testing.assert_allclose(r['variants']['ridge_original']['prediction'],
                                   original['variants']['ridge20']['prediction'], atol=1e-12, rtol=0)
        for key, variant in (r['variants'] | r['controls']).items():
            for mode, row in variant['modes'].items():
                sim, t = row['sim'], row['trade_stats']
                self.assertEqual(sim['position'][0], 0)
                self.assertAlmostEqual(sum(sim['daily_pnl']), sim['metrics']['total'], places=4)
                self.assertAlmostEqual(sum(sim['daily_pnl']), sum(t['pnl'] for t in sim['trades']), places=4)
                self.assertAlmostEqual(sum(sim['daily_pnl']), sum(m['pnl'] for m in row['months'].values()), places=4)
                self.assertAlmostEqual(t['closed_pnl'] + t['floating_pnl'], sim['metrics']['total'], places=4)
                if t['closed_n']:
                    self.assertAlmostEqual(t['win_rate'], t['wins'] / t['closed_n'])
                self.assertTrue(all(1 <= t['holding_sessions'] <= 6 for t in sim['trades']))
                np.testing.assert_allclose(sim['daily_pnl'][1:], np.array(sim['position'][1:]) * np.diff(r['level']), atol=1e-10)
                if mode == 'cold':
                    self.assertTrue(all(p is None for p in row['gate']['percentile'][:60]))
                if key == 'ridge_original':
                    for field in ('position', 'daily_pnl', 'equity', 'drawdown'):
                        np.testing.assert_allclose(sim[field], original['variants']['ridge20']['modes'][mode]['sim'][field])

    def test_dynamic_updates_delayed_and_future_targets_do_not_rewrite_predictions(self):
        updates = self.report['diagnostics']['dlm']['updates']
        self.assertTrue(all(u['asof'] == u['label_exit'] for u in updates))
        self.assertTrue(all(b['origin_index'] - a['origin_index'] >= 6 for a, b in zip(updates, updates[1:])))
        d, level, _, y, base, cut, _ = study.load_data()
        z = study.transform_record(self.bundle['models']['ridge_temporal'],
                                    study.matrix(study.temporal_features(level, base)))
        state = self.bundle['models']['dlm_temporal']
        args = (cut + 1, self.bundle['meta']['last_train_index'], np.array(state['mean']),
                np.array(state['covariance']), state['noise_variance'], self.bundle['meta']['discount'])
        original = study.dynamic_forecast(z, y, *args)
        changed_y = y.copy()
        change_origin = cut + 50
        changed_y[change_origin:] += 1000
        changed = study.dynamic_forecast(z, changed_y, *args)
        count_unchanged = change_origin + 6 - (cut + 1)
        np.testing.assert_array_equal(original['mean'][:count_unchanged], changed['mean'][:count_unchanged])
        prefix_end = cut + 100
        prefix = study.dynamic_forecast(z[:prefix_end], y[:prefix_end], *args)
        np.testing.assert_array_equal(original['mean'][:prefix_end-cut-1], prefix['mean'])

    def test_xgb_restores_and_study_is_idempotent_and_protects_production(self):
        before = {n: study.sha(n) for n in study.PROTECTED + (study.MODELS, study.OUTPUT)}
        study.build()
        self.assertEqual(before, {n: study.sha(n) for n in before})
        self.assertEqual(self.report['meta']['protected_hashes'], {n: study.sha(n) for n in study.PROTECTED})
        json.dumps(self.report, allow_nan=False)
        json.dumps(self.bundle, allow_nan=False)
        d, level, _, _, base, cut, _ = study.load_data()
        z = study.transform_record(self.bundle['models']['ridge_temporal'],
                                    study.matrix(study.temporal_features(level, base)))
        restored = study.restore_xgb(self.bundle['models']['xgb_temporal'])
        pred = restored.predict(study.xgboost.DMatrix(z[cut+1:]))
        np.testing.assert_array_equal(pred, self.report['variants']['xgb_temporal']['prediction'])


if __name__ == '__main__':
    unittest.main()
