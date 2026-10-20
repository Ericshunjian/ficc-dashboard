import contextlib
import io
import json
import unittest

import numpy as np

import prepare_exit_rules_study as study


def simulate(p,rule='sign',cut=.3,signals=None):
    p=np.array(p,float);n=len(p);dates=[f'2026-01-{i+1:02}' for i in range(n)]
    gate=study.entry_gate(p,rule,cut)
    if signals is not None:
        gate['signal']=np.array(signals,int);gate['eligible']=np.ones(n,bool)
        gate['percentile']=np.ones(n)*80;gate['history_n']=np.ones(n,int)*60
    return study.simulate_reverse(dates,np.arange(n,dtype=float),p,gate,rule,cut)


class ExitTests(unittest.TestCase):
    def test_raw_reversal_exits_next_session_after_current_pnl_and_can_reverse(self):
        s=simulate([1,1,-1,-1,-1])
        self.assertEqual(s['position'],[0,1,1,-1,-1])
        self.assertEqual(s['daily_pnl'],[0,1,1,-1,-1])
        t=s['trades'][0]
        self.assertEqual(t['exit_signal_date'],'2026-01-03')
        self.assertEqual(t['exit_session'],'2026-01-04')
        self.assertEqual(t['exit_price'],2)
        self.assertEqual(t['pnl'],2)
        self.assertEqual(t['reason'],'reverse')

    def test_weak_negative_closes_long_without_automatically_opening_short(self):
        s=simulate([1,-.1,-.4,-.4,-.4],'asymmetric')
        self.assertEqual(s['position'],[0,1,0,-1,-1])
        self.assertEqual(s['decisions'][1]['next_position'],0)
        self.assertEqual(s['trades'][1]['signal_date'],'2026-01-03')
        self.assertEqual(s['trades'][1]['entry_cutoff'],-.3)

    def test_zero_and_weak_same_direction_do_not_exit_or_reset_six_day_clock(self):
        for p,rule in [([1,1,0,1,1,1,1,1,1],'sign'),
                       ([-.4,-.1,0,-.1,-.1,-.4,-.1,-.4,-.4],'asymmetric')]:
            s=simulate(p,rule)
            self.assertEqual(s['trades'][0]['reason'],'max_hold')
            self.assertEqual(s['trades'][0]['holding_sessions'],6)
            self.assertEqual(s['trades'][0]['exit_session'],'2026-01-07')

    def test_rank_follows_opposite_rank_instead_of_raw_sign_or_neutral(self):
        s=simulate([-1]*5,'rank',signals=[1,0,1,-1,-1])
        self.assertEqual(s['position'],[0,1,1,1,-1])
        self.assertEqual(s['trades'][0]['holding_sessions'],3)
        self.assertEqual(s['trades'][0]['reason'],'reverse')
        for d in (-1,1):
            self.assertFalse(study.reverses(d,-d,0,'rank'))
            self.assertFalse(study.reverses(d,0,0,'sign'))

    def test_last_reversal_is_plan_only_and_future_cannot_change_past_pnl(self):
        s=simulate([1,1,-1])
        self.assertEqual(s['metrics']['closed_trades'],0)
        self.assertFalse(s['trades'][0]['closed'])
        self.assertEqual(s['trades'][0]['pnl'],2)
        self.assertEqual(s['decisions'][-1]['next_position'],-1)
        self.assertEqual(s['decisions'][-1]['exit_reason'],'reverse')
        long=simulate([1,1,-1,-1,1,1])
        self.assertEqual(s['daily_pnl'],long['daily_pnl'][:3])
        self.assertEqual(s['position'],long['position'][:3])


class ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=study.read(study.OUTPUT)
        cls.fixed={**study.read('entry_rules_study.json')['periods'],**study.read('entry_rules_history.json')['periods']}

    def test_36_reverse_accounts_reconcile_with_unchanged_entry_signals(self):
        self.assertEqual(self.r['meta']['strategies'],18)
        for name,p in self.r['periods'].items():
            old=self.fixed[name];level=np.array(old['level'])
            self.assertEqual(p['dates'],old['dates'])
            for key,model in p['models'].items():
                prediction=old.get('prediction',{}).get(key)
                if prediction is None:
                    prediction=study.read('entry_rules_study.json')['models'][key]['prediction'][old['start_index']:]
                for rule,row in model.items():
                    fixed=old['models'][key][rule];s=row['sim'];daily=np.array(s['daily_pnl']);pos=np.array(s['position'])
                    for field in ('signal','eligible','percentile','history_n','counts','signal_stats'):
                        self.assertEqual(row[field],fixed[field])
                    np.testing.assert_allclose(daily,pos*np.r_[0,np.diff(level)],atol=1e-10)
                    np.testing.assert_allclose(np.cumsum(daily),s['equity'],atol=1e-10)
                    self.assertAlmostEqual(sum(t['pnl'] for t in s['trades']),daily.sum())
                    self.assertAlmostEqual(sum(m['total'] for m in row['months'].values()),daily.sum())
                    self.assertAlmostEqual(row['trade_stats']['closed_pnl']+row['trade_stats']['floating_pnl'],daily.sum())
                    self.assertLessEqual(np.abs(pos).max(),1)
                    self.assertEqual(pos[0],0)
                    self.assertEqual(len(s['decisions']),len(level))
                    self.assertEqual(sum(row['exit_counts'].values()),s['metrics']['closed_trades'])
                    if 'years' in row:
                        self.assertAlmostEqual(sum(y['total'] for y in row['years'].values()),daily.sum())
                        self.assertAlmostEqual(sum(y['total'] for y in row['stages'].values()),daily.sum())
                    for t in s['trades']:
                        i=p['dates'].index(t['signal_date'])
                        self.assertEqual(t['entry_session'],p['dates'][i+1])
                        self.assertLessEqual(t['holding_sessions'],6)
                        if not t['closed']:continue
                        j=p['dates'].index(t['exit_signal_date'])
                        self.assertEqual(t['exit_price'],level[j])
                        if t['reason']=='max_hold':
                            self.assertEqual(t['holding_sessions'],6)
                            self.assertEqual(t['exit_session'],p['dates'][j])
                        else:
                            self.assertLess(t['holding_sessions'],6)
                            self.assertEqual(t['exit_session'],p['dates'][j+1])
                            self.assertTrue(study.reverses(t['direction'],prediction[j],row['signal'][j],rule))

    def test_october_bayesian_long_exits_when_prediction_turns_negative(self):
        p=self.r['periods']['history_full'];row=p['models']['dlm_temporal']['asymmetric']
        # Earlier reversals alter entry dates too: this account entered long on
        # Sep 30 rather than inheriting the fixed-exit account's Oct 12 trade.
        trades=[t for t in row['sim']['trades'] if t['direction']==1 and t['exit_signal_date']=='2026-10-13']
        self.assertEqual(len(trades),1)
        self.assertEqual(trades[0]['exit_signal_date'],'2026-10-13')
        self.assertEqual(trades[0]['exit_session'],'2026-10-14')
        i=p['dates'].index('2026-10-13')
        self.assertEqual(row['sim']['position'][i],1)
        self.assertEqual(row['sim']['decisions'][i]['next_position'],0)
        self.assertEqual(row['sim']['position'][i+1],0)

    def test_idempotent_with_every_fixed_archive_unchanged(self):
        before={n:study.sha(n) for n in study.PROTECTED+(study.OUTPUT,)}
        with contextlib.redirect_stdout(io.StringIO()):study.build()
        self.assertEqual(before,{n:study.sha(n) for n in before})
        self.assertEqual(self.r['meta']['protected_hashes'],{n:study.sha(n) for n in study.PROTECTED})
        json.dumps(self.r,allow_nan=False)


if __name__=='__main__':unittest.main()
