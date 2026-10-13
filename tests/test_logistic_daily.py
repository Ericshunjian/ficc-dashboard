"""Frozen Logistic reproduction, probability gates and immutable daily archives."""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
import prepare_logistic_daily as logistic
import prepare_ridge_daily as daily


class LogisticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = logistic.read(logistic.MODELS)
        cls.research = logistic.read(logistic.SOURCE)
        cls.report = daily.read(daily.OUTPUT)['logistic']

    def test_fixed_model_and_complete_training_configuration(self):
        logistic.validate_models(self.frozen)
        self.assertEqual(self.frozen['meta']['train_n'], 406)
        self.assertEqual(self.frozen['meta']['train_through'], '2024-12-31')
        self.assertEqual(self.report['meta']['model_sha256'], logistic.sha(logistic.MODELS))
        for arm in logistic.ARMS:
            self.assertEqual(self.frozen['models'][arm]['C'], .001)
            self.assertTrue(all(not f['clip_enabled'] for f in self.frozen['models'][arm]['features'][:8]))

    def test_live_probability_and_logit_audit(self):
        for arm in logistic.ARMS:
            r = self.report['latest']['models'][arm]
            self.assertNotEqual(r['status'], 'blocked')
            score = self.frozen['models'][arm]['intercept'] + sum(x['contribution'] for x in r['inputs'])
            self.assertAlmostEqual(r['logit'], score, places=13)
            self.assertAlmostEqual(r['prediction'], expit(score), places=13)
            self.assertTrue(0 <= r['prediction'] <= 1)
        self.assertEqual(logistic.probability(1000), 1)
        self.assertEqual(logistic.probability(-1000), 0)

    def test_probability_not_percentile_drives_direction(self):
        # High historical rank cannot create a long below 60% probability.
        p = np.r_[np.repeat(.45, 60), .59, .6, .60001, .4, .39999, np.nan]
        g = logistic.gate(p)
        self.assertEqual(g['history_percentile'][60], 100)
        np.testing.assert_array_equal(g['signal'][60:], [0, 0, 1, 0, -1, 0])
        self.assertTrue((g['signal'][:60] == 0).all())
        p[-1] = 1
        altered = logistic.gate(p)
        np.testing.assert_allclose(g['history_percentile'][:-1], altered['history_percentile'][:-1], equal_nan=True)

    def test_local_approved_experiment_reproduced(self):
        expected = {'f20': (9.5518, 1.4085), 'f30': (8.2228, 1.4085), 'f40': (5.6525, 1.5993)}
        for arm, (pnl, dd) in expected.items():
            key = 'logistic' + arm[1:]
            m = self.research['strategies']['dynamic'][key]['metrics']
            self.assertEqual((m['total'], m['max_drawdown']), (pnl, dd))
            n = len(self.research['daily']['dates'])
            archived = logistic.archive_simulation(self.report['records'][:n], arm)
            for field in ('equity', 'daily_pnl', 'drawdown', 'position'):
                np.testing.assert_allclose(archived[field], self.research['strategies']['dynamic'][key][field], atol=1e-12)
            a = self.research['daily']['prediction'][key]
            b = [r['models'][arm]['prediction'] for r in self.report['records'][:n]]
            np.testing.assert_array_equal(a, b)

    def test_deployment_backfill_not_mislabelled_live(self):
        self.assertEqual(self.report['meta']['deployment_backfill_through'], '2026-10-12')
        day = next(r for r in self.report['records'] if r['date'] == '2026-10-12')
        self.assertEqual(day['origin'], 'deployment-backfill')
        self.assertTrue(all(r['status'] == 'deployment-backfill' for r in day['models'].values()))

    def test_probability_archive_and_accounting_immutable(self):
        rows = copy.deepcopy(self.report['records'])
        logistic.enrich(rows)
        self.assertEqual(rows, self.report['records'])
        level = np.array([r['level'] for r in rows])
        for arm in logistic.ARMS:
            s = self.report['strategies'][arm]
            np.testing.assert_allclose(s['daily_pnl'], np.array(s['position']) * np.r_[0, np.diff(level)], atol=1e-12)
            np.testing.assert_allclose(s['equity'], np.cumsum(s['daily_pnl']), atol=1e-12)
            self.assertAlmostEqual(sum(t['pnl'] for t in s['trades']), s['equity'][-1], places=10)
            self.assertTrue(all(t['holding_sessions'] <= 6 for t in s['trades']))
            self.assertTrue(all(t['signal_date'] < rows[-1]['date'] for t in s['trades']))
            self.assertEqual(s['metrics'], self.report['positions'][arm]['metrics'])

    def test_daily_build_never_refits_and_is_idempotent(self):
        names = (daily.MODELS, daily.OUTPUT, logistic.MODELS, logistic.SOURCE, 'factor_library.json', 'data_version.json')
        before = {n: daily.sha(n) for n in names}
        with patch('prepare_curve_timing_capacity.fit_model', side_effect=AssertionError('Unexpected Ridge fit')), \
             patch('prepare_direction_screen.screen_snapshot', side_effect=AssertionError('Unexpected screening')), \
             patch('sklearn.linear_model.LogisticRegression.fit', side_effect=AssertionError('Unexpected Logistic fit')):
            _, changes = daily.build()
        self.assertEqual(changes['new_logistic_dates'], [])
        self.assertFalse(changes['signal_changed'])
        self.assertEqual(before, {n: daily.sha(n) for n in names})


if __name__ == '__main__':
    unittest.main()
