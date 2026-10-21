import contextlib
import copy
import io
import unittest
from datetime import datetime,timedelta
from unittest.mock import patch

import numpy as np

import prepare_curve_signal_daily as study


class ProbabilityTests(unittest.TestCase):
    def test_predictive_probability_symmetry_and_uncertainty(self):
        self.assertEqual(study.probability(0,1)['up_probability'],.5)
        a,b=study.probability(1,1),study.probability(-1,1)
        self.assertAlmostEqual(a['up_probability'],b['down_probability'])
        self.assertLess(study.probability(1,2)['up_probability'],a['up_probability'])
        self.assertLess(a['interval80'][0],1);self.assertGreater(a['interval80'][1],1)
        for sd in [0,-1,np.nan]:
            with self.assertRaises(ValueError):study.probability(0,sd)

    def test_calibration_excludes_pending_labels_and_has_fixed_bins(self):
        c=study.probability_check([.5,1,.1],[1,-1,np.nan],[0,0,0],[1,1,1])
        self.assertEqual(c['n'],2)
        self.assertEqual(sum(b['n'] for b in c['bins']),2)
        self.assertEqual(c['bins'][5]['n'],1);self.assertEqual(c['bins'][9]['n'],1)
        self.assertAlmostEqual(c['brier'],(.25+1)/2)


class ResumeTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(15);n=32
        dates=[(datetime(2026,1,1)+timedelta(days=i)).date().isoformat() for i in range(n)]
        z=rng.normal(size=(n,92));level=100+np.cumsum(rng.normal(0,.1,n));x=z.copy()
        state={'mean':np.r_[.1,np.zeros(92)],'covariance':np.eye(93)*.01,'noise_variance':.2}
        y=study.make_offset_labels(level,6)
        out=study.dynamic_forecast(z[:20],study.make_offset_labels(level[:20],6),6,0,state['mean'],state['covariance'],.2,.995,dates[:20])
        records=[study.row(dates[t],level[t],x[t],z[t],{k:float(out['mean'][t-6]) for k in study.KEYS},{},'research-backfill','test',out['std'][t-6]) for t in range(6,20)]
        rec=lambda n:{'intercept':0.,'features':[{'median':0,'clip_enabled':False,'mean':0,'std':1,'coef':.01} for _ in range(n)]}
        bundle={'models':{'ridge_original':rec(28),'ridge_temporal':rec(92)}}
        available={'bond':set(dates),'repo':set(dates),'policy_verified':dates[-1]}
        return dates,z,level,x,state,out,records,bundle,available

    def test_resume_equals_uninterrupted_filter_and_ignores_old_revisions(self):
        dates,z,level,x,initial,out,records,bundle,available=self.fixture()
        state=study.posterior(out,dates,.2)
        expected=study.dynamic_forecast(z,study.make_offset_labels(level,6),6,0,initial['mean'],initial['covariance'],.2,.995,dates)
        rawlevel=level.copy();rawlevel[18]+=9
        rawz=z.copy();rawz[18]+=99
        before=copy.deepcopy(records)
        result,posterior,updates,new,issue,revisions=study.advance(records,state,copy.deepcopy(out['updates']),dates,rawlevel,x,rawz,bundle,available,'test',{})
        self.assertEqual(result[:len(before)],before)
        self.assertEqual(len(new),12);self.assertIsNone(issue);self.assertEqual(len(revisions),1)
        np.testing.assert_allclose([r['models']['bayes_temporal_new']['prediction'] for r in result],expected['mean'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(posterior['mean'],expected['final_mean'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(posterior['covariance'],expected['final_covariance'],atol=1e-12,rtol=0)
        for u in updates:self.assertEqual(u['asof_index']-u['origin_index'],6)

    def test_missing_new_data_blocks_without_manufacturing_signal(self):
        dates,z,level,x,_,out,records,bundle,available=self.fixture();before=copy.deepcopy(records)
        available['repo'].remove(dates[20])
        result,_,_,new,issue,_=study.advance(records,study.posterior(out,dates,.2),out['updates'],dates,level,x,z,bundle,available,'test',{})
        self.assertEqual(result,before);self.assertEqual(new,[]);self.assertEqual(issue['date'],dates[20])


class DailyArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.r=study.read(study.OUTPUT)

    def test_new_model_identity_and_historical_replication(self):
        r=self.r;source=study.read('three_model_strategy_review.json');params=study.read('bayes_parameter_study.json')
        n=len(source['predictions']['bayes_temporal_new'])
        for k in study.KEYS:
            self.assertEqual([v['models'][k]['prediction'] for v in r['records'][:n]],source['predictions'][k])
            self.assertEqual(r['simulation']['models'][k]['ytd'],source['periods']['full']['models'][k]['sign']['reverse']['sim'])
        std=params['grid']['d0.995_c1']['std']
        np.testing.assert_allclose([v['models']['bayes_temporal_new']['predictive_std'] for v in r['records'][:n]],std,atol=0,rtol=0)
        self.assertEqual(r['meta']['model_version'],'new-temporal-bayes-d0995-c1-alpha500')
        for f,h in r['meta']['frozen_hashes'].items():self.assertEqual(study.sha(f),h)

    def test_probability_coverage_and_price_ledgers_reconcile(self):
        r=self.r;y=study.make_offset_labels(np.array([v['level'] for v in r['records']]),6)
        b=[v['models']['bayes_temporal_new'] for v in r['records']]
        for v in b:
            self.assertAlmostEqual(v['up_probability']+v['down_probability'],1)
            self.assertEqual(v['signal'],int(np.sign(v['prediction'])))
            self.assertEqual(study.probability(v['prediction'],v['predictive_std']),{k:v[k] for k in ('up_probability','down_probability','predictive_std','interval80')})
        self.assertEqual(r['probability_check']['n'],int(np.isfinite(y).sum()))
        for m in r['simulation']['models'].values():
            for kind in ('history','ytd'):
                s=m[kind]
                self.assertAlmostEqual(sum(t['pnl'] for t in s['trades']),sum(s['daily_pnl']),places=8)
                self.assertLessEqual(s['metrics']['max_holding'],6)
        self.assertTrue(all(v['origin']=='research-backfill' for v in r['records'] if v['date']<study.DEPLOYMENT))

    def test_repeated_run_is_idempotent_and_does_not_fit(self):
        names=study.PROTECTED+(study.OUTPUT,'factor_library.json','data_version.json')
        before={n:study.sha(n) for n in names}
        with patch('prepare_curve_timing_capacity.fit_model',side_effect=AssertionError('No fit')),patch('prepare_direction_screen.screen_snapshot',side_effect=AssertionError('No screen')),contextlib.redirect_stdout(io.StringIO()):
            result,changes=study.build()
        self.assertEqual(changes['new_dates'],[]);self.assertFalse(changes['changed'])
        self.assertFalse(changes['library_changed'])
        # Legacy tests may update data_version's old file stamps before this
        # test; only new archive/model immutability is essential here.
        for name in names:
            if name!='data_version.json':self.assertEqual(study.sha(name),before[name],name)
        self.assertEqual(result['records'],self.r['records'])


if __name__=='__main__':unittest.main()
