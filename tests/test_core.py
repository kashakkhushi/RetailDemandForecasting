import numpy as np
import pandas as pd
import pytest
from rdf.data.schema import fill_calendar,validate_panel,SchemaError
from rdf.features.lags import build_lag_features
from rdf.evaluation.metrics import mae,rmse,bias,wape,smape,mase,rmsse,pinball_loss,coverage
from rdf.evaluation.backtest import BacktestConfig,make_cutoffs,rolling_origin
from rdf.models.baselines import Naive,SeasonalNaive,MovingAverage,SeasonalMovingAverage,CrostonSBA
from rdf.decision.newsvendor import NewsvendorCosts,policy_cost,recommended_quantile

def panel(n=40): return pd.DataFrame({'unique_id':['a']*n,'ds':pd.date_range('2020-01-01',periods=n),'y':np.arange(n)%5})
@pytest.mark.parametrize('fn,expect',[(mae,1.),(rmse,1.),(bias,1.)])
def test_metrics_hand(fn,expect): assert fn([1,2],[2,3])==expect
@pytest.mark.parametrize('y,h,expect',[([0,0],[0,0],np.nan),([1,1],[0,0],1.),([2,2],[3,3],.5)])
def test_wape(y,h,expect):
 r=wape(y,h); assert np.isnan(r) if np.isnan(expect) else r==expect
@pytest.mark.parametrize('m', [Naive,SeasonalNaive,MovingAverage,SeasonalMovingAverage,CrostonSBA])
def test_models_shape_nonnegative(m):
 x=panel(); p=m().fit(x).predict(3,pd.DataFrame({'unique_id':['a']*3,'ds':pd.date_range('2020-02-10',periods=3)})); assert len(p)==3 and np.isfinite(p.yhat).all() and (p.yhat>=0).all()
@pytest.mark.parametrize('fill', [0.,2.])
def test_fill_calendar(fill):
 x=fill_calendar(pd.DataFrame({'unique_id':['a','a'],'ds':['2020-01-01','2020-01-03'],'y':[1,3]}),fill_value=fill); assert len(x)==3 and x.y.sum()==4+fill and x.y.iloc[1]==fill
def test_duplicate_rejected():
 with pytest.raises(SchemaError): validate_panel(pd.DataFrame({'unique_id':['a','a'],'ds':['2020-01-01']*2,'y':[1,2]}))
@pytest.mark.parametrize('h',[1,7,28])
def test_lags_no_shorter_than_h(h):
 x=build_lag_features(panel(),h); assert min(int(c.split('_')[1]) for c in x if c.startswith('lag_'))>=h
def test_lag_future_perturbation_no_effect():
 x=panel(50); a=build_lag_features(x,7); x.loc[x.ds>'2020-01-30','y']=999; b=build_lag_features(x,7); pd.testing.assert_frame_equal(a[a.ds<='2020-01-30'],b[b.ds<='2020-01-30'])
def test_mase_constant_nan(): assert np.isnan(mase([1,1],[1,1],[1]*20))
def test_rmsse_constant_nan(): assert np.isnan(rmsse([1,1],[1,1],[1]*20))
def test_pinball_and_coverage(): assert pinball_loss([1],[2],.75)==.25 and coverage([2],[1],[3])==1
def test_cutoffs_disjoint():
 x=panel(100); c=make_cutoffs(x,BacktestConfig(horizon=7,n_folds=3,initial_train=20)); assert all((b-a).days==7 for a,b in zip(c,c[1:]))
def test_factory_per_fold():
 seen=[]
 def f(): seen.append(1); return Naive()
 rolling_origin(panel(60),f,BacktestConfig(horizon=7,n_folds=3,initial_train=20)); assert len(seen)==3
@pytest.mark.parametrize('ratio,expect',[(1,.5),(3,.75),(9,.9)])
def test_critical_ratio(ratio,expect): assert NewsvendorCosts(ratio,1).critical_ratio==expect
def test_nothing_cost(): assert policy_cost([2,3],[0,0],NewsvendorCosts())['total_cost']==15
def test_perfect_free(): assert policy_cost([2,3],[2,3],NewsvendorCosts(0,0))['total_cost']==0
def test_recommended_quantile(): assert recommended_quantile(NewsvendorCosts(3,1),[.5,.9,.75])==.75
@pytest.mark.parametrize('actual,order,short,over',[(0,0,0,0),(1,0,1,0),(0,1,0,1),(5,3,2,0),(3,5,0,2),(7,7,0,0),(10,8,2,0),(8,10,0,2)])
def test_policy_unit_accounting(actual,order,short,over):
 r=policy_cost([actual],[order],NewsvendorCosts()); assert r['units_short']==short and r['units_over']==over
@pytest.mark.parametrize('actual,q1,q2',[(10,5,9),(3,0,3),(7,1,8)])
def test_fill_monotone(actual,q1,q2): assert policy_cost([actual],[q2],NewsvendorCosts())['fill_rate']>=policy_cost([actual],[q1],NewsvendorCosts())['fill_rate']
