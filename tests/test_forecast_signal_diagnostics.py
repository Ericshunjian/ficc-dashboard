import contextlib
import io
import json
import unittest
from unittest.mock import patch

import numpy as np

import prepare_forecast_signal_diagnostics as study
from prepare_temporal_study import matrix, temporal_features, linear_prediction


class FixedStrataTests(unittest.TestCase):
    def test_sign_strength_and_inclusive_boundaries(self):
        p=np.array([.1,.2,.3,-.1,-.2,-.3])
        b=study.fit_boundaries(p)
        q1,q2=b['1']['cuts']
        n1,n2=b['-1']['cuts']
        codes=study.classify(np.array([0,q1,q2,q2+.1,-n1,-n2,-n2-.1]),b)
        np.testing.assert_array_equal(codes,[0,1,2,3,-1,-2,-3])
        self.assertEqual(b['1']['n'],3)
        self.assertEqual(b['-1']['n'],3)

    def test_ties_allow_empty_groups_and_missing_direction_is_not_invented(self):
        b=study.fit_boundaries(np.array([.2,.2,.2]))
        np.testing.assert_array_equal(study.classify(np.array([.2,.3,-.1,0]),b),[1,3,-99,0])
        self.assertIsNone(b['-1']['cuts'])
        with self.assertRaises(ValueError):study.classify(np.array([np.nan]),b)
        with self.assertRaises(ValueError):study.fit_boundaries(np.array([np.nan]))

    def test_rows_reconcile_zero_pending_and_signed_actual_results(self):
        p=np.array([.1,.2,.3,-.1,-.2,-.3,0,.1])
        b=study.fit_boundaries(p[:6])
        y=np.array([.2,-.1,.4,-.3,.1,-.5,0,np.nan])
        r=study.summarize_period([str(i) for i in range(len(p))],p,y,b)
        self.assertEqual(sum(x['forecast_n'] for x in r['rows'])+r['quality']['zero_prediction_n'],8)
        self.assertEqual(sum(x['n'] for x in r['rows'])+r['quality']['zero_completed_n'],7)
        self.assertEqual(r['quality']['pending_n'],1)
        negative=r['rows'][-1]
        self.assertEqual(negative['actual_mean'],-.5)
        self.assertEqual(negative['signed_mean'],.5)
        self.assertEqual(negative['direction_accuracy'],1)
        self.assertEqual(negative['n'],1)
        self.assertTrue(negative['small_sample'])
        self.assertEqual(negative['interval95'],{})

    def test_bootstrap_keeps_date_blocks_and_refuses_interior_target_compression(self):
        p=np.r_[np.ones(20),-np.ones(20)]
        y=p.copy()
        b=study.fit_boundaries(np.r_[p,.2,.5,-.2,-.5])
        codes=study.classify(p,b)
        ci,_=study.bootstrap_groups(p,y,codes)
        for c in np.unique(codes):
            self.assertEqual(ci[str(c)]['direction_accuracy']['low'],1)
            self.assertEqual(ci[str(c)]['direction_accuracy']['high'],1)
        y[15]=np.nan
        with self.assertRaises(ValueError):study.bootstrap_groups(p,y,codes)


class DiagnosticArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=study.read(study.OUTPUT)
        cls.c=study.read(study.CALIBRATION)

    def test_calibration_uses_only_2025_completed_predictions(self):
        c=self.c
        self.assertEqual(c['meta']['train_n'],523)
        self.assertEqual(c['meta']['last_train_signal'],'2025-06-20')
        self.assertEqual(c['meta']['last_train_label_exit'],'2025-06-30')
        self.assertEqual(c['meta']['completed_n'],120)
        self.assertEqual(c['meta']['last_complete_signal'],'2025-12-23')
        self.assertEqual(c['target'][-6:],[None]*6)
        self.assertTrue(all(day.startswith('2025') for day in c['dates']))
        for k in study.KEYS:
            p=np.array(c['predictions'][k])
            complete=np.isfinite(np.array(c['target'],float))
            self.assertEqual(study.fit_boundaries(p[complete]),c['boundaries'][k])
        self.assertTrue(all(u['asof']==u['label_exit'] for u in c['updates']))
        self.assertTrue(all(b['origin_index']-a['origin_index']>=6 for a,b in zip(c['updates'],c['updates'][1:])))
        with self.assertRaises(ValueError):study.calibrate(['2026-01-01'],np.ones(1),np.ones((1,28)),np.ones((1,92)),[],[])

    def test_future_2025_observations_cannot_revise_earlier_calibration_forecasts(self):
        dates,level,xb,_,base,year_end,_=study.load_data()
        features=temporal_features(level,base);xt=matrix(features)
        end=max(i for i,d in enumerate(dates) if d<='2025-09-30')
        prefix=study.calibrate(dates[:end+1],level[:end+1],xb[:end+1],xt[:end+1],base,features)
        n=len(prefix['dates'])
        for k in study.KEYS:
            np.testing.assert_allclose(prefix['predictions'][k],self.c['predictions'][k][:n],rtol=0,atol=1e-12)
        start=dates.index(self.c['dates'][0])
        for k,x in [('ridge_original',xb),('ridge_temporal',xt)]:
            p=linear_prediction(self.c['models'][k],x[start:year_end+1])
            np.testing.assert_allclose(p,self.c['predictions'][k],rtol=0,atol=1e-12)

    def test_original_forecasts_and_all_group_counts_and_errors_reconcile(self):
        source=study.read('forecast_model_comparison.json')
        self.assertEqual(set(self.r['models']),set(study.KEYS))
        self.assertEqual(self.r['meta']['test_complete_n'],183)
        for k,m in self.r['models'].items():
            current=m['test2026']
            np.testing.assert_array_equal(current['prediction'],source['models'][k]['prediction'])
            self.assertEqual(current['target'],source['target'])
            self.assertEqual(current['quality']['pending_n'],6)
            self.assertEqual(current['quality']['unsupported_n'],0)
            for period,expected_n in [('test2026',183),('calibration2025',120)]:
                r=m[period]
                self.assertEqual(sum(x['n'] for x in r['rows']),expected_n)
                self.assertEqual(sum(x['forecast_n'] for x in r['rows']),r['quality']['forecast_n'])
                p,y=np.array(r['prediction']),np.array(r['target'],float)
                complete=np.isfinite(y)
                weighted_mae=sum(x['n']*x['mae'] for x in r['rows'] if x['n'])/expected_n
                weighted_mse=sum(x['n']*x['rmse']**2 for x in r['rows'] if x['n'])/expected_n
                self.assertAlmostEqual(weighted_mae,np.mean(np.abs(p[complete]-y[complete])))
                self.assertAlmostEqual(weighted_mse,np.mean((p[complete]-y[complete])**2))
                for row in r['rows']:
                    if row['direction_n']:
                        self.assertEqual(row['direction_accuracy'],row['correct']/row['direction_n'])
                    if row['n']:
                        self.assertAlmostEqual(row['signed_mean'],row['direction']*row['actual_mean'])
        # Sparse 100% must not become an unsupported confidence claim.
        sparse=self.r['models']['ridge_original']['test2026']['rows'][-1]
        self.assertEqual(sparse['n'],4)
        self.assertTrue(sparse['small_sample'])
        self.assertIsNone(sparse['interval95']['direction_accuracy']['low'])

    def test_idempotent_without_refit_or_new_boundaries_and_all_old_artifacts_protected(self):
        files=study.PROTECTED+(study.CALIBRATION,study.OUTPUT)
        before={n:study.sha(n) for n in files}
        with patch.object(study,'calibrate',side_effect=AssertionError('do not refit')):
            with patch.object(study,'fit_boundaries',side_effect=AssertionError('do not refit cuts')):
                with contextlib.redirect_stdout(io.StringIO()):study.build()
        self.assertEqual(before,{n:study.sha(n) for n in files})
        self.assertEqual(self.r['meta']['protected_hashes'],{n:study.sha(n) for n in study.PROTECTED})
        self.assertEqual(self.r['meta']['calibration_sha256'],study.sha(study.CALIBRATION))
        json.dumps(self.r,allow_nan=False)
        json.dumps(self.c,allow_nan=False)
        def check(v):
            if isinstance(v,dict):
                self.assertFalse({'sim','pnl','gate','position','win_rate'}&set(v))
                for x in v.values():check(x)
            elif isinstance(v,list):
                for x in v:check(x)
        check(self.r)


if __name__=='__main__':unittest.main()
