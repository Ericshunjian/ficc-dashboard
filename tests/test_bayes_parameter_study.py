import copy
import unittest

import numpy as np

import prepare_bayes_parameter_study as study


class CausalFilterTests(unittest.TestCase):
    def test_tuning_refuses_post_2025_input(self):
        with self.assertRaisesRegex(ValueError, '2025-12-31'):
            study.tune(['2026-01-05'], np.zeros(1), np.zeros((1, 1)), {})

    def test_selection_does_not_use_test_metrics_or_pnl(self):
        rows = {'a': {'validation': {'rmse': 1}, 'discount': .98, 'covariance_multiplier': 6},
                'b': {'validation': {'rmse': .9}, 'discount': 1, 'covariance_multiplier': 1}}
        self.assertEqual(study.select(rows), 'b')
        changed = copy.deepcopy(rows)
        changed['a'].update(metrics={'rmse': 0}, total_pnl=999)
        changed['b'].update(metrics={'rmse': 100}, total_pnl=-999)
        self.assertEqual(study.select(changed), 'b')

    def test_covariance_controls_gain_and_forecasts_are_causal(self):
        z = np.ones((26, 1))
        y = np.arange(26, dtype=float)
        state = {'mean': [0., 0.], 'covariance': [[6., 0.], [0., 6.]], 'noise_variance': 1.}
        dates = [str(i) for i in range(26)]
        low = study.run(z, y, 7, 0, state, .995, 1., dates)
        high = study.run(z, y, 7, 0, state, .995, 18., dates)
        self.assertLess(abs(low['mean'][5]), abs(high['mean'][5]))
        prefix = study.run(z[:20], y[:20], 7, 0, state, .995, 1., dates[:20])
        np.testing.assert_array_equal(low['mean'][:13], prefix['mean'])
        future_changed = y.copy()
        future_changed[14:] = -9999
        altered = study.run(z, future_changed, 7, 0, state, .995, 1., dates)
        np.testing.assert_array_equal(low['mean'][:13], altered['mean'][:13])
        for row in low['updates']:
            self.assertEqual(row['asof_index']-row['origin_index'], 6)
        self.assertGreaterEqual(np.linalg.eigvalsh(low['final_covariance']).min(), -1e-12)


class ParameterArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = study.read(study.OUTPUT)
        cls.frozen = study.read(study.MODELS)
        cls.old = study.read('forecast_model_comparison.json')

    def test_prior_artifacts_are_unchanged(self):
        for name, digest in self.r['meta']['protected_hashes'].items():
            self.assertEqual(study.sha(name), digest, name)

    def test_old_parameter_replication_and_shared_dates(self):
        self.assertEqual(self.r['dates'], self.old['dates'])
        self.assertEqual(self.r['target'], self.old['target'])
        np.testing.assert_allclose(self.r['grid']['d1_c6']['prediction'],
                                   self.old['models']['dlm_temporal']['prediction'], atol=1e-11, rtol=0)
        self.assertEqual(len(self.r['grid']), 9)

    def test_validation_is_purged_and_selection_frozen(self):
        v = self.r['validation']
        self.assertLessEqual(v['last_train_label_exit'], v['initial_train_through'])
        self.assertGreater(v['dates'][0], v['initial_train_through'])
        self.assertLessEqual(v['last_complete_exit'], '2025-12-31')
        self.assertEqual(v['n'], sum(t is not None for t in v['target']))
        self.assertEqual(self.r['meta']['selected'], study.select(self.r['grid']))
        self.assertEqual(self.frozen['selected'], self.r['meta']['selected'])
        selected = self.r['grid'][self.frozen['selected']]
        oldstate = study.read('curve_timing_temporal_models.json')['models']['dlm_temporal']
        np.testing.assert_allclose(self.frozen['state']['mean'], oldstate['mean'], atol=0, rtol=0)
        np.testing.assert_allclose(self.frozen['state']['covariance'],
                                   np.array(oldstate['covariance'])*selected['covariance_multiplier']/6.)

    def test_all_updates_wait_for_labels_and_do_not_overlap(self):
        for row in self.r['grid'].values():
            for field in ['validation_updates', 'updates']:
                updates = row[field]
                for u in updates:
                    self.assertEqual(u['asof'], u['label_exit'])
                    self.assertEqual(u['asof_index']-u['origin_index'], 6)
                self.assertTrue(all(b['origin_index']-a['origin_index']>=6 for a,b in zip(updates,updates[1:])))
            self.assertEqual(row['validation']['n'], self.r['validation']['n'])
            self.assertEqual(row['metrics']['n'], self.r['meta']['completed_n'])
            self.assertEqual(len(row['prediction']), len(self.r['dates']))

    def test_prediction_and_trading_metrics_reconcile(self):
        for key, model in self.r['models'].items():
            self.assertEqual(study.metrics(model['prediction'], self.r['target']), model['metrics'])
            for trade in model['trading'].values():
                s = trade['sim']
                self.assertAlmostEqual(sum(s['daily_pnl']), s['metrics']['total'], places=5)
                self.assertAlmostEqual(sum(t['pnl'] for t in s['trades']), sum(s['daily_pnl']), places=10)
                self.assertLessEqual(s['metrics']['max_holding'], 6)
            self.assertEqual(sum(q['metrics']['n'] for q in model['quarters'].values()), model['metrics']['n'])
        expected = study.read('exit_rules_study.json')['periods']['full']['models']['ridge_temporal']['sign']['sim']
        self.assertEqual(self.r['models']['ridge']['trading']['sign']['sim'], expected)


if __name__ == '__main__':
    unittest.main()
