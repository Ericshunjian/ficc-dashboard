import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

import prepare_strategy_stability as study


class StabilityTests(unittest.TestCase):
    def test_slices_keep_carried_pnl_and_rebase_drawdown(self):
        dates=['2026-01-05','2026-01-06','2026-01-07','2026-01-08']
        sim={'daily_pnl':[0,2,-1,-2],'position':[0,1,1,-1]}
        r=study.slice_stats(dates,sim,[2,3])
        self.assertEqual(r['total'],-3)
        self.assertEqual(r['max_drawdown'],3)
        self.assertEqual(r['long_pnl'],-1)
        self.assertEqual(r['short_pnl'],-2)
        self.assertTrue(r['first_interval_is_carry'])

    def test_concentration_removes_closed_wins_only_and_keeps_float(self):
        row={'sim':{'daily_pnl':[0,12],'trades':[{'pnl':v,'closed':True} for v in [1,2,3,4,-5]]+[{'pnl':7,'closed':False}]}}
        r=study.concentration(row)
        self.assertEqual(r['top3_pnl'],9)
        self.assertEqual(r['total_after_top3'],3)
        self.assertEqual(r['share_of_net'],.75)
        self.assertEqual(r['share_of_gross_wins'],.9)


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=study.read(study.OUTPUT)
        cls.reverse=study.read('exit_rules_study.json')
        cls.neutral=study.read('neutral_exit_study.json')

    def test_all_months_quarters_recent_windows_and_sides_reconcile(self):
        for period,p in self.r['periods'].items():
            for key,c in study.CANDIDATES.items():
                row=p['models'][key]
                source=(self.reverse if c['exit']=='reverse' else self.neutral)['periods'][period]['models']['ridge_temporal'][c['rule']]
                self.assertEqual(row['metrics'],source['sim']['metrics'])
                for partitions in ['months','quarters']:
                    self.assertAlmostEqual(sum(v['total'] for v in row[partitions].values()),row['metrics']['total'],places=8)
                self.assertAlmostEqual(sum(s['closed_pnl']+s['floating_pnl'] for s in row['sides'].values()),row['metrics']['total'],places=8)
                for window in (30,60):
                    r=row['recent'][str(window)]
                    self.assertEqual(r['n'],window)
                    self.assertEqual(r['end'],p['dates'][-1])
                    self.assertEqual(r['start'],p['dates'][-window])
                    self.assertAlmostEqual(r['total'],sum(source['sim']['daily_pnl'][-window:]))
                    self.assertIsNone(row['rolling'][str(window)][window-2])
                    self.assertAlmostEqual(row['rolling'][str(window)][-1],r['total'])
                con=row['concentration']
                self.assertAlmostEqual(con['total_after_top3']+con['top3_pnl'],row['metrics']['total'],places=8)
                self.assertTrue(all(t['closed'] and t['pnl']>0 for t in con['trades']))

    def test_six_starts_include_original_and_are_not_optimized(self):
        for p in self.r['periods'].values():
            for row in p['models'].values():
                s=row['starts'];self.assertEqual(len(s['rows']),6)
                self.assertEqual([v['offset'] for v in s['rows']],list(range(6)))
                self.assertEqual(s['rows'][0]['start'],p['dates'][0])
                self.assertEqual(s['rows'][0]['metrics'],row['metrics'])
                values=[v['metrics']['total'] for v in s['rows']]
                self.assertEqual(s['min'],min(values));self.assertEqual(s['max'],max(values))
                self.assertEqual(s['positive_n'],sum(v>0 for v in values))

    def test_shared_long_book_is_disclosed(self):
        for p in self.r['periods'].values():
            rel=p['relationship']
            self.assertTrue(rel['long_trades_equal'])
            self.assertGreater(rel['long_trade_n'],0)
            self.assertLessEqual(rel['same_position_fraction'],1)
            self.assertGreaterEqual(rel['daily_pnl_correlation'],-1)

    def test_bayes_diagnostics_use_completed_labels_and_report_net_sign_changes(self):
        b=self.r['bayes'];f=study.read('forecast_model_comparison.json');y=np.array(f['target'],float)
        self.assertEqual(b['periods']['all']['n'],int(np.isfinite(y).sum()))
        self.assertFalse(b['last_prediction']['actual_complete'])
        self.assertEqual(self.r['meta']['last_complete_signal'],f['dates'][np.flatnonzero(np.isfinite(y))[-1]])
        s=b['sign_changes'];self.assertEqual(s['n'],len(s['rows']))
        self.assertEqual(s['helped'],sum(v['effect']=='helped' for v in s['rows']))
        self.assertEqual(s['hurt'],sum(v['effect']=='hurt' for v in s['rows']))
        allm=b['periods']['all']['models']
        self.assertEqual(s['helped']-s['hurt'],allm['dlm_temporal']['direction_correct']-allm['ridge_temporal']['direction_correct'])
        for change in s['rows']:
            self.assertTrue(np.isfinite(y[f['dates'].index(change['date'])]))
        for v in b['recent'].values():self.assertLessEqual(v['end'],self.r['meta']['last_complete_signal'])

    def test_updates_are_delayed_and_attribution_matches_prediction_difference(self):
        b=self.r['bayes']
        self.assertEqual(b['config']['updates'],len(b['updates']))
        for u in b['updates']:
            self.assertEqual(u['asof'],u['label_exit'])
            self.assertEqual(u['asof_index']-u['origin_index'],6)
        self.assertTrue(all(v['origin_index']-u['origin_index']>=6 for u,v in zip(b['updates'],b['updates'][1:])))
        self.assertAlmostEqual(sum(v['mean'] for v in b['attribution'].values()),np.mean(b['prediction_delta']),places=12)
        self.assertLessEqual(b['interval80_contains_zero_n'],b['interval_n'])
        for p in b['pure_strategy_control'].values():
            self.assertEqual(set(p),{'ridge_temporal','dlm_temporal'})

    def test_idempotent_without_retraining_or_changing_any_prior_artifact(self):
        before={n:study.sha(n) for n in study.PROTECTED+(study.OUTPUT,)}
        with patch('prepare_temporal_study.fit_training',side_effect=AssertionError('Do not fit')):
            with contextlib.redirect_stdout(io.StringIO()):study.build()
        self.assertEqual(before,{n:study.sha(n) for n in before})
        self.assertEqual(self.r['meta']['protected_hashes'],{n:study.sha(n) for n in study.PROTECTED})


if __name__=='__main__':unittest.main()
