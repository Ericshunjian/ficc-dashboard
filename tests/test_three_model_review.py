import copy
import unittest

import numpy as np

import prepare_three_model_review as study


class ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=study.read(study.OUTPUT)
        cls.manifest=study.read(study.MANIFEST)

    def test_sources_are_frozen_and_predictors_identical(self):
        for name,digest in self.r['meta']['protected_hashes'].items():
            self.assertEqual(study.sha(name),digest,name)
        previous=study.read('forecast_model_comparison.json')
        params=study.read('bayes_parameter_study.json')
        self.assertEqual(self.r['periods']['full']['dates'],previous['dates'])
        for key in study.KEYS[:2]:
            self.assertEqual(self.r['predictions'][key],previous['models'][key]['prediction'])
        self.assertEqual(self.r['predictions']['bayes_temporal_new'],params['models']['selected']['prediction'])

    def test_all_27_paths_reconcile_every_ledger_and_partition(self):
        for period,p in self.r['periods'].items():
            count=0
            for k,m in p['models'].items():
                for entry,exits in m.items():
                    for policy,row in exits.items():
                        count+=1;s=row['sim'];d=row['diagnostics']
                        self.assertEqual(len(s['daily_pnl']),len(p['dates']))
                        self.assertEqual(s['position'][0],0)
                        self.assertAlmostEqual(sum(t['pnl'] for t in s['trades']),sum(s['daily_pnl']),places=9)
                        self.assertAlmostEqual(sum(s['daily_pnl']),s['metrics']['total'],places=5)
                        self.assertLessEqual(s['metrics']['max_holding'],6)
                        for group in ['months','quarters']:
                            self.assertAlmostEqual(sum(v['total'] for v in d[group].values()),sum(s['daily_pnl']),places=9)
                        self.assertAlmostEqual(sum(d['side_pnl'].values()),sum(s['daily_pnl']),places=9)
                        self.assertAlmostEqual(d['concentration']['total_after_top3']+d['concentration']['top3_pnl'],sum(s['daily_pnl']),places=9)
                        self.assertEqual(len(d['starts']),6)
                        self.assertEqual(d['starts'][0]['metrics'],s['metrics'])
                        self.assertEqual(d['path_sha256'],study.fingerprint(s['position']))
                        if entry!='rank':self.assertTrue(all(v is None for v in row['percentile']))
            self.assertEqual(count,27)

    def test_prior_pure_and_rank_accounts_are_preserved(self):
        sources={p:study.read(f) for p,f in [('fixed','entry_rules_study.json'),('reverse','exit_rules_study.json'),('neutral','neutral_exit_study.json')]}
        for period,p in self.r['periods'].items():
            for key in study.KEYS[:2]:
                for entry in ['sign','rank']:
                    for policy,source in sources.items():
                        self.assertEqual(p['models'][key][entry][policy]['sim'],source['periods'][period]['models'][key][entry]['sim'])

    def test_new_bayes_existing_two_mappings_reproduced(self):
        old=study.read('bayes_parameter_study.json')
        for key,source in [('ridge_temporal','ridge'),('bayes_temporal_new','selected')]:
            for entry,policy in [('sign','reverse'),('asymmetric','neutral')]:
                self.assertEqual(self.r['periods']['full']['models'][key][entry][policy]['sim'],old['models'][source]['trading'][entry]['sim'])

    def test_rank_warmup_and_common_restart(self):
        start=self.r['periods']['common']['start_index']
        self.assertEqual(start,60)
        full=self.r['periods']['full'];common=self.r['periods']['common']
        for key in study.KEYS:
            for entry in study.RULES:
                for policy in study.EXITS:
                    a=full['models'][key][entry][policy];b=common['models'][key][entry][policy]
                    self.assertEqual(a['signal'][start:],b['signal'])
                    if entry=='rank':
                        self.assertEqual(a['sim']['position'][:60],[0]*60)
                        self.assertEqual(a['percentile'][start:],b['percentile'])

    def test_selection_is_reproducible_and_not_all_eligible_paths_selected(self):
        self.assertEqual(study.shortlist(self.r['periods']),self.r['selection'])
        s=self.r['selection'];selected=[s['rows'][k] for k in s['selected']]
        self.assertLessEqual(len(selected),2)
        self.assertEqual(len({v['model'] for v in selected}),len(selected))
        self.assertEqual(len({(v['paths']['full'],v['paths']['common']) for v in selected}),len(selected))
        self.assertTrue(all(v['eligible'] for v in selected))
        for p in self.r['periods'].values():
            for v in selected:
                r=p['models'][v['model']][v['entry']][v['exit']]
                self.assertGreater(r['diagnostics']['start_min'],0)
                self.assertGreater(r['diagnostics']['concentration']['total_after_top3'],0)

    def test_failure_cannot_be_overridden_by_high_profit(self):
        changed=copy.deepcopy(self.r['periods'])
        for period in changed.values():
            for entry in period['models']['bayes_temporal_new'].values():
                for row in entry.values():
                    row['diagnostics']['concentration']['total_after_top3']=-1
                    row['sim']['metrics']['total']=999
        result=study.shortlist(changed)
        self.assertTrue(all(result['rows'][k]['model']!='bayes_temporal_new' for k in result['selected']))

    def test_manifest_does_not_backdate_selection_or_claim_deployment(self):
        m=self.manifest
        self.assertGreater(m['selected_on'],m['research_asof'])
        self.assertEqual(m['selected_on'],self.r['meta']['selected_on'])
        self.assertEqual([v['model'] for v in m['candidates']],[self.r['selection']['rows'][k]['model'] for k in self.r['selection']['selected']])
        self.assertEqual(m['candidates'][0]['role'],'主交易参考')
        self.assertIn('未切换',m['status'])
        for name,digest in m['source_hashes'].items():self.assertEqual(study.sha(name),digest)

    def test_duplicate_rules_and_relationship_are_disclosed(self):
        for period,p in self.r['periods'].items():
            paths={}
            for k,m in p['models'].items():
                a=m['sign']['reverse']['sim'];b=m['sign']['neutral']['sim']
                self.assertEqual(a['position'],b['position'])
                for q in m.values():
                    for row in q.values():paths[row['diagnostics']['path_sha256']]=1
            self.assertEqual(len(paths),self.r['meta']['unique_paths'][period])
        rel=self.r['relationships'][0]
        a,b=(self.r['selection']['rows'][rel[k]] for k in ['a','b'])
        for period,p in self.r['periods'].items():
            get=lambda v:p['models'][v['model']][v['entry']][v['exit']]['sim']
            first,second=get(a),get(b)
            delta=np.array(first['daily_pnl'])-np.array(second['daily_pnl'])
            self.assertAlmostEqual(rel['periods'][period]['pnl_difference'],delta.sum(),places=9)
            self.assertGreaterEqual(rel['periods'][period]['same_position'],0)
            self.assertLessEqual(rel['periods'][period]['same_position'],1)


if __name__=='__main__':unittest.main()
