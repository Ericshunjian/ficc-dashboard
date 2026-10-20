import contextlib
import io
import json
import unittest

import numpy as np

import prepare_neutral_exit_study as study
from prepare_exit_rules_study import simulate_reverse,build as rebuild_reverse


def simulate(p,rule='asymmetric',signals=None):
    p=np.array(p,float);n=len(p);dates=[f'2026-01-{i+1:02}' for i in range(n)]
    gate=study.entry_gate(p,rule,.3)
    if signals is not None:
        gate['signal']=np.array(signals,int);gate['eligible']=np.ones(n,bool)
        gate['percentile']=np.ones(n)*50;gate['history_n']=np.ones(n,int)*60
    return simulate_reverse(dates,np.arange(n,dtype=float),p,gate,rule,.3,exit_policy='neutral')


class NeutralExitTests(unittest.TestCase):
    def test_weak_negative_exits_short_next_session_without_opening_long(self):
        s=simulate([-.4,-.1,-.2,-.4,-.4,-.4])
        self.assertEqual(s['position'],[0,-1,0,0,-1,-1])
        self.assertEqual(s['decisions'][1]['next_position'],0)
        self.assertEqual(s['trades'][0]['exit_signal_date'],'2026-01-02')
        self.assertEqual(s['trades'][0]['exit_session'],'2026-01-03')
        self.assertEqual(s['trades'][0]['reason'],'neutral')
        self.assertEqual(s['trades'][0]['pnl'],-1)

    def test_neutral_boundaries_exit_both_sides_and_do_not_reenter(self):
        for p in [[-.4,-.3,-.3],[1,0,0],[1,-.1,-.1],[-.4,0,0]]:
            s=simulate(p)
            self.assertEqual(s['decisions'][1]['next_position'],0)
            self.assertEqual(s['trades'][0]['reason'],'neutral')
            self.assertEqual(s['metrics']['closed_trades'],1)
        # Reaching the opposite entry standard still permits a reversal.
        s=simulate([1,-.4,-.4])
        self.assertEqual(s['decisions'][1]['next_position'],-1)
        self.assertEqual(s['trades'][0]['reason'],'reverse')

    def test_rank_neutral_and_boundaries_exit_despite_positive_raw_prediction(self):
        for signals in ([1,0,0],[-1,0,0]):
            s=simulate([1,1,1],'rank',signals)
            self.assertEqual(s['decisions'][1]['next_position'],0)
            self.assertEqual(s['trades'][0]['reason'],'neutral')
        for value in (23.5,35.5):
            gate=study.entry_gate(np.r_[np.arange(60.),value],'rank',.3)
            self.assertEqual(gate['signal'][-1],0)

    def test_zero_is_only_sign_neutral_and_last_day_is_plan_only(self):
        s=simulate([1,0,1,1],'sign')
        self.assertEqual(s['position'],[0,1,0,1])
        self.assertEqual(s['trades'][0]['reason'],'neutral')
        short=simulate([-.4,-.4,-.1])
        self.assertEqual(short['metrics']['closed_trades'],0)
        self.assertFalse(short['trades'][0]['closed'])
        self.assertEqual(short['trades'][0]['pnl'],-2)
        self.assertEqual(short['decisions'][-1]['next_position'],0)
        self.assertEqual(short['decisions'][-1]['exit_reason'],'neutral')
        longer=simulate([-.4,-.4,-.1,-.4,-.4])
        self.assertEqual(short['daily_pnl'],longer['daily_pnl'][:3])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=study.read(study.OUTPUT)
        cls.fixed={**study.read('entry_rules_study.json')['periods'],**study.read('entry_rules_history.json')['periods']}
        cls.reverse=study.read('exit_rules_study.json')

    def test_all_accounts_reconcile_and_entry_signals_are_unchanged(self):
        for name,p in self.r['periods'].items():
            old=self.fixed[name];price=np.array(old['level'])
            self.assertEqual(p['dates'],old['dates'])
            for key,model in p['models'].items():
                for rule,row in model.items():
                    s=row['sim'];daily=np.array(s['daily_pnl']);pos=np.array(s['position'])
                    for field in ('signal','eligible','percentile','history_n','counts','signal_stats'):
                        self.assertEqual(row[field],old['models'][key][rule][field])
                    np.testing.assert_allclose(daily,pos*np.r_[0,np.diff(price)],atol=1e-10)
                    np.testing.assert_allclose(np.cumsum(daily),s['equity'],atol=1e-10)
                    self.assertAlmostEqual(sum(t['pnl'] for t in s['trades']),daily.sum())
                    self.assertAlmostEqual(sum(v['total'] for v in row['months'].values()),daily.sum())
                    self.assertEqual(sum(row['exit_counts'].values()),s['metrics']['closed_trades'])
                    if 'years' in row:
                        self.assertAlmostEqual(sum(v['total'] for v in row['years'].values()),daily.sum())
                        self.assertAlmostEqual(sum(v['total'] for v in row['stages'].values()),daily.sum())
                    self.assertLessEqual(np.abs(pos).max(),1)
                    self.assertEqual(pos[0],0)
                    for trade in s['trades']:
                        if not trade['closed']:continue
                        i=p['dates'].index(trade['exit_signal_date'])
                        self.assertEqual(trade['exit_price'],price[i])
                        if trade['reason']=='max_hold':
                            self.assertEqual(trade['holding_sessions'],6)
                            self.assertEqual(trade['exit_session'],p['dates'][i])
                        else:
                            self.assertLess(trade['holding_sessions'],6)
                            self.assertEqual(trade['exit_session'],p['dates'][i+1])
                            self.assertEqual(row['signal'][i],0 if trade['reason']=='neutral' else -trade['direction'])

    def test_pure_direction_is_explicitly_duplicate_without_new_threshold(self):
        for name,p in self.r['periods'].items():
            for key in study.KEYS:
                row=p['models'][key]['sign'];old=self.reverse['periods'][name]['models'][key]['sign']
                self.assertTrue(row['same_as_reverse'])
                self.assertIn(key+'|sign',p['same_as_reverse'])
                self.assertEqual(row['sim']['daily_pnl'],old['sim']['daily_pnl'])
                self.assertEqual(row['sim']['position'],old['sim']['position'])

    def test_neutral_rank_matches_independently_archived_dynamic_rank(self):
        reference=study.read('curve_timing_temporal.json')
        for key in study.KEYS:
            ref=reference['variants'][key]['modes']['cold']['sim']
            now=self.r['periods']['full']['models'][key]['rank']['sim']
            for field in ('daily_pnl','position','equity','drawdown'):
                np.testing.assert_allclose(now[field],ref[field],atol=1e-10)

    def test_both_generators_are_idempotent_and_old_18_strategies_unchanged(self):
        before={n:study.sha(n) for n in study.PROTECTED+(study.OUTPUT,)}
        with contextlib.redirect_stdout(io.StringIO()):
            rebuild_reverse();study.build()
        self.assertEqual(before,{n:study.sha(n) for n in before})
        self.assertEqual(self.r['meta']['protected_hashes'],{n:study.sha(n) for n in study.PROTECTED})
        json.dumps(self.r,allow_nan=False)


if __name__=='__main__':unittest.main()
