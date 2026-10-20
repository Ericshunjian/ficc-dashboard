import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import spearmanr

import prepare_forecast_comparison as study


class PredictionMetricTests(unittest.TestCase):
    def test_imbalance_does_not_masquerade_as_balanced_skill(self):
        actual = np.r_[np.ones(90), -np.ones(10)]
        m = study.metrics(np.ones(100), actual)
        self.assertEqual(m['direction_accuracy'], .9)
        self.assertEqual(m['balanced_accuracy'], .5)
        self.assertEqual(m['up_recall'], 1)
        self.assertEqual(m['down_recall'], 0)
        self.assertIsNone(m['down_precision'])
        self.assertIsNone(m['rank_ic'])

    def test_zero_predictions_are_not_deleted_and_true_zeros_are_explicit(self):
        p, y = np.array([1, -1, 0, 0, np.nan]), np.array([1, -1, 1, 0, -1])
        m = study.metrics(p, y)
        self.assertEqual(m['n'], 4)
        self.assertEqual(m['direction_n'], 3)
        self.assertEqual(m['direction_correct'], 2)
        self.assertEqual(m['actual_zero_n'], 1)
        self.assertEqual(m['predicted_zero_n'], 2)
        self.assertEqual(m['direction_accuracy'], 2/3)
        self.assertAlmostEqual(m['mae'], .25)
        self.assertAlmostEqual(m['rmse'], .5)
        self.assertAlmostEqual(m['rank_ic'], spearmanr(p[:4], y[:4]).statistic)

    def test_rolling_scores_appear_only_when_all_63_six_day_labels_mature(self):
        p = np.sin(np.arange(100))
        y = np.cos(np.arange(100))
        y[-6:] = np.nan
        r = study.rolling_metrics(p, y)
        self.assertTrue(all(v is None for v in r['mae'][:68]))
        self.assertEqual(r['mae'][68], study.metrics(p[:63], y[:63])['mae'])
        changed = y.copy()
        changed[70:] += 100
        alt = study.rolling_metrics(p, changed)
        np.testing.assert_array_equal(r['mae'][:76], alt['mae'][:76])

    def test_candidate_selection_does_not_use_pnl_and_allows_no_winner(self):
        rows = {k: {'balanced_accuracy': .5, 'direction_accuracy': .5, 'rank_ic': 0,
                     'mae': 1., 'rmse': 1.2, 'pnl': 100000} for k in study.KEYS}
        rows['ridge_temporal'].update(balanced_accuracy=.7, direction_accuracy=.7, rank_ic=.3, pnl=-100)
        rows['ridge_original'].update(balanced_accuracy=.65, direction_accuracy=.65, rank_ic=.4, mae=.5, rmse=.6)
        benchmarks = {'zero': {'rmse': .8}, 'train_mean': {'rmse': .9}}
        s = study.select_candidates(rows, benchmarks)
        self.assertEqual(s['selected'], ['ridge_temporal', 'ridge_original'])
        self.assertEqual(s['amplitude_candidate'], 'ridge_original')
        for r in rows.values():
            r['pnl'] *= -100000
        self.assertEqual(study.select_candidates(rows, benchmarks), s)
        rows = {k: dict(r, balanced_accuracy=.5) for k, r in rows.items()}
        self.assertEqual(study.select_candidates(rows, benchmarks)['selected'], [])

    def test_bootstrap_keeps_models_paired_and_equal_forecasts_have_zero_difference(self):
        y = np.sin(np.arange(80))
        p = y + .1
        config = {'block': 12, 'draws': 50, 'seed': 42}
        r = study.paired_intervals({'ridge_original': p, 'same': p.copy()}, y, config)
        for key in ('ridge_original', 'same'):
            for metric in ('mae', 'rmse', 'rank_ic', 'direction_accuracy', 'balanced_accuracy'):
                self.assertEqual(r[key][metric]['low'], 0)
                self.assertEqual(r[key][metric]['high'], 0)
        ix = study.bootstrap_indices(80, **config)
        self.assertEqual(ix.shape, (50, 80))
        self.assertTrue(np.all(np.diff(ix[:, :12], axis=1) == 1))


class PredictionArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = study.read(study.OUTPUT)

    def test_models_share_labels_and_existing_five_forecasts_are_unchanged(self):
        r = self.r
        self.assertEqual(set(r['models']), set(study.KEYS))
        self.assertEqual(r['meta']['train_n'], 649)
        self.assertEqual(r['meta']['completed_n'], 183)
        self.assertEqual(r['meta']['forecast_n'], 189)
        self.assertEqual(r['target'][-6:], [None]*6)
        old, temporal = study.read('curve_timing_bayes.json'), study.read('curve_timing_temporal.json')
        np.testing.assert_array_equal(r['models']['ridge_original']['prediction'], old['variants']['ridge20']['prediction'])
        np.testing.assert_array_equal(r['models']['dlm_original']['prediction'], old['variants']['dlm20']['prediction'])
        for k in study.KEYS[3:]:
            np.testing.assert_array_equal(r['models'][k]['prediction'], temporal['variants'][k]['prediction'])
        for k, v in r['models'].items():
            self.assertEqual(v['metrics']['n'], 183)
            self.assertEqual(sum(q['n'] for q in v['quarters'].values()), 183)
            self.assertEqual(sum(p['n'] for p in v['phases']), 183)
            self.assertEqual(v['quarters']['2026-Q4']['n'], 2)
            self.assertFalse(v['quarters']['2026-Q4']['sufficient'])
            self.assertEqual(v['metrics']['up_n']+v['metrics']['down_n']+v['metrics']['actual_zero_n'], 183)
        self.assertEqual(r['meta']['last_evaluation_signal'], '2026-10-09')
        self.assertEqual(r['meta']['last_evaluation_label_exit'], '2026-10-19')

    def test_new_xgb_training_is_2025_only_and_frozen_model_is_reproducible(self):
        frozen = study.read(study.MODELS)
        m, v = frozen['meta'], frozen['validation']
        self.assertEqual(m['last_train_label_exit'], '2025-12-31')
        self.assertEqual(m['last_train_signal'], '2025-12-23')
        self.assertEqual(v['last_inner_exit'], '2025-06-30')
        self.assertEqual(v['n'], 120)
        self.assertEqual(m['rounds'], int(np.argmin(v['rmse_path']))+1)
        self.assertFalse(any(f['clip_enabled'] for f in frozen['model']['preprocessing']['features'][:8]))
        with self.assertRaises(ValueError):
            study.fit_original_xgb(['2026-01-01'], np.ones(1), np.ones((1,28)), [])
        dates, level, xb, y, _, cut, _ = study.load_data()
        self.assertEqual(m['input_sha256'], study.xgb_fingerprint(xb[:cut+1],
                         study.make_offset_labels(level[:cut+1],6), dates[:cut+1]))
        booster = study.restore_xgb(frozen['model'])
        z = study.transform_record(frozen['model']['preprocessing'], xb[cut+1:])
        np.testing.assert_array_equal(booster.predict(study.xgboost.DMatrix(z)),
                                       self.r['models']['xgb_original']['prediction'])
        for updates in self.r['update_audit'].values():
            self.assertTrue(all(u['asof']==u['label_exit'] for u in updates))
            self.assertTrue(all(b['origin_index']-a['origin_index']>=6 for a,b in zip(updates,updates[1:])))

    def test_selection_and_intervals_are_consistent_and_no_trade_fields_exist(self):
        r=self.r
        rows={k:v['metrics'] for k,v in r['models'].items()}
        self.assertEqual(study.select_candidates(rows,r['benchmarks']),r['selection'])
        self.assertLessEqual(len(r['selection']['selected']),2)
        self.assertEqual(r['selection']['selected'],['ridge_temporal','ridge_original'])
        self.assertEqual(r['retained'],study.read(study.SHORTLIST))
        self.assertEqual(r['retained']['selected'],['ridge_temporal','ridge_original','dlm_temporal'])
        self.assertEqual(r['retained']['user_added'],['dlm_temporal'])
        self.assertNotIn('dlm_temporal',r['selection']['selected'])
        self.assertIsNone(r['selection']['amplitude_candidate'])
        ci=r['paired_intervals']['ridge_temporal']['direction_accuracy']
        self.assertLess(ci['low'],0)
        self.assertGreater(ci['high'],0)
        pair=r['selected_pairs']['ridge_temporal_vs_ridge_original']
        self.assertEqual(sum(pair.values()),183)
        self.assertEqual(pair['a_only_correct']-pair['b_only_correct'],1)
        def check(obj):
            if isinstance(obj,dict):
                self.assertFalse({'pnl','sim','gate','win_rate','position'} & set(obj))
                for value in obj.values():check(value)
            elif isinstance(obj,list):
                for value in obj:check(value)
        check(r)
        json.dumps(r,allow_nan=False)

    def test_build_is_idempotent_without_refit_and_protected_files_stay_unchanged(self):
        files=study.PROTECTED+(study.MODELS,study.OUTPUT)
        before={n:study.sha(n) for n in files}
        with patch.object(study,'fit_original_xgb',side_effect=AssertionError('must restore, not refit')):
            with contextlib.redirect_stdout(io.StringIO()):study.build()
        self.assertEqual(before,{n:study.sha(n) for n in files})
        self.assertEqual(self.r['meta']['protected_hashes'],{n:study.sha(n) for n in study.PROTECTED})


if __name__=='__main__':
    unittest.main()
