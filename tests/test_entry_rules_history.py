import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

import prepare_entry_rules_history as study
from prepare_curve_timing_percentile import signed_percentile_gate


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=study.read(study.OUTPUT)
        cls.full=cls.r['periods']['history_full']
        cls.train=cls.r['periods']['history_train']
        cls.source=study.read('forecast_model_comparison.json')

    def test_price_start_and_training_label_boundary_are_explicit(self):
        self.assertEqual(self.full['dates'][0],'2023-04-21')
        self.assertEqual(self.train['dates'][-1],'2025-12-31')
        self.assertEqual(self.train['target'][-6:],[None]*6)
        self.assertEqual(sum(y is not None for y in self.train['target']),649)
        self.assertEqual(self.r['meta']['train_last_signal'],'2025-12-23')
        self.assertEqual(self.r['meta']['train_price_n'],655)
        self.assertEqual(len(self.full['dates']),844)
        self.assertTrue(np.isfinite(self.full['level']).all())
        self.assertGreater(self.r['meta']['missing_temporal_cells'],0)

    def test_archived_test_forecasts_preserved_bayes_prehistory_is_only_initial_mean(self):
        n=len(self.train['dates'])
        self.assertEqual(self.full['dates'][n:],self.source['dates'])
        self.assertEqual(self.full['target'][n:],self.source['target'])
        for key in study.KEYS:
            self.assertEqual(self.full['prediction'][key][n:],self.source['models'][key]['prediction'])
            self.assertEqual(self.train['prediction'][key],self.full['prediction'][key][:n])
        self.assertEqual(self.train['prediction']['ridge_temporal'],self.train['prediction']['dlm_temporal'])
        self.assertNotEqual(self.full['prediction']['ridge_temporal'][n:],self.full['prediction']['dlm_temporal'][n:])

    def test_all_ledgers_daily_stages_years_and_months_reconcile(self):
        for period in (self.full,self.train):
            level=np.array(period['level']); n=len(level)
            for models in period['models'].values():
                for row in models.values():
                    sim=row['sim']; pnl=np.array(sim['daily_pnl']); pos=np.array(sim['position'])
                    np.testing.assert_allclose(pnl,pos*np.r_[0.,np.diff(level)],atol=1e-10)
                    np.testing.assert_allclose(np.cumsum(pnl),sim['equity'],atol=1e-10)
                    for total in [sum(t['pnl'] for t in sim['trades']),sum(v['total'] for v in row['stages'].values()),
                                  sum(v['total'] for v in row['years'].values()),sum(v['total'] for v in row['months'].values())]:
                        self.assertAlmostEqual(total,pnl.sum(),places=9)
                    self.assertEqual(pos[0],0)
                    self.assertLessEqual(np.abs(pos).max(),1)
                    self.assertAlmostEqual(sim['metrics']['exposure'],np.mean(pos[1:]!=0))
                    for t in sim['trades']:
                        i=period['dates'].index(t['signal_date'])
                        self.assertEqual(t['entry_session'],period['dates'][i+1])
                        if t['closed']:
                            self.assertEqual(t['holding_sessions'],6)
                            self.assertEqual(t['exit_session'],period['dates'][i+6])
                    self.assertEqual(len(sim['decisions']),n)

    def test_continuous_path_carries_history_and_training_account_is_same_prefix(self):
        n=len(self.train['dates'])
        for key in study.KEYS:
            expected=signed_percentile_gate(np.array(self.full['prediction'][key]))
            rank=self.full['models'][key]['rank']
            np.testing.assert_allclose(np.array(rank['percentile'],float),expected['percentile'],equal_nan=True)
            self.assertEqual(rank['history_n'][n],250)
            self.assertEqual(rank['percentile'][:60],[None]*60)
            for rule in study.RULES:
                full=self.full['models'][key][rule]; train=self.train['models'][key][rule]
                for field in ['daily_pnl','equity','position']:
                    np.testing.assert_array_equal(full['sim'][field][:n],train['sim'][field])
                self.assertEqual(full['signal'][:n],train['signal'])

    def test_2026_continuous_segment_is_not_mislabeled_as_restart(self):
        for models in self.full['models'].values():
            for row in models.values():
                s=row['stages']['test']
                self.assertEqual(s['start'],self.source['dates'][0])
                self.assertIn('不重置持仓',s['note'])
        old=study.read('entry_rules_study.json')
        self.assertNotEqual(self.full['models']['ridge_temporal']['asymmetric']['stages']['test']['total'],
                            old['periods']['full']['models']['ridge_temporal']['asymmetric']['sim']['metrics']['total'])

    def test_idempotence_and_old_models_forecasts_and_tests_are_unchanged(self):
        files=study.PROTECTED+(study.OUTPUT,)
        before={n:study.sha(n) for n in files}
        with patch('prepare_temporal_study.fit_training',side_effect=AssertionError('No retraining')):
            with patch('prepare_bayes_study.dynamic_forecast',side_effect=AssertionError('No invented historical updates')):
                with contextlib.redirect_stdout(io.StringIO()):study.build()
        self.assertEqual(before,{n:study.sha(n) for n in files})
        self.assertEqual(self.r['meta']['protected_hashes'],{n:study.sha(n) for n in study.PROTECTED})
        json.dumps(self.r,allow_nan=False)


if __name__=='__main__':unittest.main()
