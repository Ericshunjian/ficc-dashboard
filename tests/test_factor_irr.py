import unittest
import numpy as np
from factor_irr import build,DIFF,MA5,CHANGE5,T_NAME,TL_NAME
from prepare_irr_factor_checkup import rank_hac,fdr_adjust


class IrrTests(unittest.TestCase):
    def fixture(self):
        d=['2026-01-05','2026-01-06','2026-01-07','2026-01-08','2026-01-09','2026-01-12','2026-01-13']
        values={T_NAME:[1.,2.,3.,4.,5.,6.,7.],TL_NAME:[1.]*7,'T主力':[100.]*7,'TL主力':[110.]*7}
        return d,{'series':{k:{'dates':d,'values':v} for k,v in values.items()}}

    def unpack(self,rows,name,n):
        s=next(s for s in rows if s['name']==name);v=np.full(n,np.nan)
        v[s['i0']:s['i0']+len(s['v'])]=[np.nan if a is None else a for a in s['v']]
        return v

    def test_difference_has_no_quantity_multiplier_and_uses_sessions(self):
        d,c=self.fixture();rows=build(c,d)
        np.testing.assert_allclose(self.unpack(rows,DIFF,7),np.arange(7))
        np.testing.assert_allclose(self.unpack(rows,MA5,7)[4:],[2.,3.,4.])
        np.testing.assert_allclose(self.unpack(rows,CHANGE5,7)[5:],[5.,5.])

    def test_missing_not_zero_and_strict_ma(self):
        d,c=self.fixture();c['series'][T_NAME]['values'][2]=None
        rows=build(c,d);v=self.unpack(rows,DIFF,7)
        self.assertTrue(np.isnan(v[2]));self.assertEqual(v[0],0.)
        self.assertFalse(any(s['name']==MA5 for s in rows))

    def test_prelisting_zero_not_valid_tl(self):
        d,c=self.fixture();c['series']['TL主力']['values'][0]=0
        v=self.unpack(build(c,d),DIFF,7)
        self.assertTrue(np.isnan(v[0]))

    def test_future_change_does_not_rewrite_history(self):
        d,c=self.fixture();a=build(c,d)
        c['series'][T_NAME]['values'][-1]=500
        b=build(c,d)
        for name in (DIFF,MA5,CHANGE5):
            np.testing.assert_allclose(self.unpack(a,name,7)[:-1],self.unpack(b,name,7)[:-1],equal_nan=True)

    def test_hac_matches_independent_statsmodels_rank_regression(self):
        from statsmodels.regression.linear_model import OLS
        from scipy.stats import rankdata
        rng=np.random.default_rng(12);x=rng.normal(size=400);y=np.cumsum(rng.normal(size=400))+.5*x
        a=rank_hac(x,y)
        m=OLS(rankdata(y),np.column_stack([np.ones(len(x)),rankdata(x)])).fit(cov_type='HAC',cov_kwds={'maxlags':a['lag'],'use_correction':True})
        self.assertAlmostEqual(a['t'],m.tvalues[1],places=10)

    def test_fdr_is_adjusted_probability_not_boolean(self):
        q=fdr_adjust([.01,.04,.03,.2,np.nan])
        np.testing.assert_allclose(q[:4],[.04,.0533333333333,.0533333333333,.2])
        self.assertTrue(np.isnan(q[-1]))


if __name__=='__main__':unittest.main()
