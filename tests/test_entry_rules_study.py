import contextlib
import io
import json
import unittest

import numpy as np

import prepare_entry_rules_study as study


class GateTests(unittest.TestCase):
    def test_strict_sign_and_asymmetric_boundaries(self):
        p = np.array([-.301, -.3, -.1, 0, .001, .3])
        np.testing.assert_array_equal(study.entry_gate(p, 'sign', .3)['signal'], [-1,-1,-1,0,1,1])
        np.testing.assert_array_equal(study.entry_gate(p, 'asymmetric', .3)['signal'], [-1,0,0,0,1,1])
        with self.assertRaises(ValueError): study.entry_gate(np.array([np.nan]), 'sign', .3)

    def test_rank_has_strict_boundaries_excludes_today_and_future(self):
        # 24/60 = 40%, 36/60 = 60%; equality remains neutral.
        for today, expected in [(23.5,0),(35.5,0),(23,-1),(36,1)]:
            p = np.r_[np.arange(60.), today]
            g = study.entry_gate(p, 'rank', .3)
            self.assertFalse(g['eligible'][:60].any())
            self.assertEqual(g['history_n'][-1], 60)
            self.assertEqual(g['signal'][-1], expected)
        p = np.linspace(-1,1,90)
        original = study.entry_gate(p,'rank',.3)
        revised = study.entry_gate(np.r_[p[:75],np.ones(20)*100],'rank',.3)
        np.testing.assert_array_equal(original['signal'][:75],revised['signal'][:75])
        np.testing.assert_allclose(original['percentile'][:75],revised['percentile'][:75],equal_nan=True)

    def test_fixed_six_ignores_middle_reversal_and_execution_is_next_session(self):
        dates = [f'2026-01-{i+1:02}' for i in range(9)]
        level = np.arange(9.)
        p = np.array([1,-1,-1,0,0,-1,-1,1,1.])
        row = study.evaluate(dates,level,p,np.full(9,np.nan),study.entry_gate(p,'sign',.3),'sign',.3)
        s = row['sim']
        self.assertEqual(s['position'],[0,1,1,1,1,1,1,-1,-1])
        first = s['trades'][0]
        self.assertEqual(first['entry_session'],dates[1])
        self.assertEqual(first['exit_session'],dates[6])
        self.assertEqual(first['holding_sessions'],6)
        self.assertEqual(first['pnl'],6)
        self.assertEqual(s['decisions'][1]['next_position'],1)
        self.assertIsNone(first['entry_percentile'])
        self.assertIsNone(first['entry_history_n'])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = study.read(study.OUTPUT)

    def test_forecasts_prices_and_boundaries_are_frozen_inputs(self):
        source = study.read('forecast_model_comparison.json')
        calibration = study.read('forecast_signal_calibration_2025.json')
        self.assertEqual(self.r['periods']['full']['dates'], source['dates'])
        self.assertEqual(self.r['periods']['full']['target'][-6:],[None]*6)
        self.assertEqual(self.r['periods']['full']['level'],study.read('curve_timing_bayes.json')['level'])
        for key in study.KEYS:
            self.assertEqual(self.r['models'][key]['prediction'],source['models'][key]['prediction'])
            self.assertEqual(self.r['meta']['negative_cuts'][key],calibration['boundaries'][key]['-1']['cuts'][1])

    def test_common_restart_is_flat_with_preserved_prior_rank_history(self):
        full, common = self.r['periods']['full'],self.r['periods']['common']
        start = common['start_index']
        self.assertEqual(start,60)
        self.assertEqual(common['dates'],full['dates'][start:])
        for key in study.KEYS:
            for rule in study.RULES:
                f,c = full['models'][key][rule],common['models'][key][rule]
                self.assertEqual(c['sim']['position'][0],0)
                self.assertEqual(c['sim']['equity'][0],0)
                self.assertEqual(c['signal'],f['signal'][start:])
                if rule == 'rank':
                    self.assertEqual(c['history_n'][0],60)
                    self.assertEqual(c['percentile'],f['percentile'][start:])
        # A reset is materially different from carrying the original positions.
        self.assertNotEqual(common['models']['ridge_temporal']['sign']['sim']['position'],
                            full['models']['ridge_temporal']['sign']['sim']['position'][start:])

    def test_all_eighteen_accounts_reconcile_and_holds_are_exactly_six(self):
        for period in self.r['periods'].values():
            level = np.array(period['level'])
            for key in study.KEYS:
                for rule in study.RULES:
                    r=period['models'][key][rule]; s=r['sim']; t=r['trade_stats']
                    position=np.array(s['position']); daily=np.array(s['daily_pnl'])
                    np.testing.assert_allclose(daily,position*np.r_[0,np.diff(level)],atol=1e-12)
                    np.testing.assert_allclose(np.cumsum(daily),s['equity'],atol=1e-12)
                    self.assertLessEqual(max(abs(position)),1)
                    self.assertEqual(len(s['decisions']),len(level))
                    self.assertAlmostEqual(sum(x['pnl'] for x in s['trades']),sum(daily))
                    self.assertAlmostEqual(t['closed_pnl']+t['floating_pnl'],sum(daily))
                    self.assertAlmostEqual(sum(x['total'] for x in r['months'].values()),sum(daily))
                    self.assertAlmostEqual(sum(x['long']+x['short'] for x in r['months'].values()),sum(daily))
                    self.assertAlmostEqual(s['metrics']['exposure'],np.mean(position[1:]!=0))
                    closed=[x for x in s['trades'] if x['closed']]
                    self.assertEqual(t['closed_n'],len(closed))
                    self.assertAlmostEqual(t['win_rate'],np.mean([x['pnl']>0 for x in closed]))
                    for trade in s['trades']:
                        i=period['dates'].index(trade['signal_date'])
                        self.assertEqual(trade['entry_session'],period['dates'][i+1])
                        self.assertEqual(trade['entry_price'],level[i])
                        if trade['closed']:
                            self.assertEqual(trade['holding_sessions'],6)
                            self.assertEqual(trade['exit_session'],period['dates'][i+6])
                        if rule!='rank': self.assertIsNone(trade['entry_percentile'])
                    for a,b in zip(s['trades'],s['trades'][1:]):
                        self.assertLessEqual(a['exit_signal_date'],b['signal_date'])

    def test_original_rank_reproduces_previous_independent_report(self):
        old=study.read('curve_timing_thresholds.json')['models']['ridge20']['periods']['test2026']['cold']['fixed']
        ref=next(r['sim'] for r in old if r['band']=='40_60')
        now=self.r['periods']['full']['models']['ridge_original']['rank']['sim']
        for field in ('position','equity','daily_pnl','drawdown'):
            np.testing.assert_allclose(now[field],ref[field],atol=1e-10,rtol=0)

    def test_idempotent_all_old_artifacts_protected_and_no_refit(self):
        files=study.PROTECTED+(study.OUTPUT,)
        before={n:study.sha(n) for n in files}
        with contextlib.redirect_stdout(io.StringIO()): study.build()
        self.assertEqual(before,{n:study.sha(n) for n in files})
        self.assertEqual(self.r['meta']['protected_hashes'],{n:study.sha(n) for n in study.PROTECTED})
        json.dumps(self.r,allow_nan=False)


if __name__=='__main__': unittest.main()
