from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from rdf.evaluation.backtest import BacktestConfig,rolling_origin,score
from rdf.models.baselines import Naive,SeasonalNaive,MovingAverage,SeasonalMovingAverage,CrostonSBA
from rdf.models.ets import ETS
from rdf.models.gbm import DirectGBM
p=pd.read_parquet('data/panel.parquet'); cfg=BacktestConfig(); models={'Naive':Naive,'SeasonalNaive':SeasonalNaive,'MovingAverage':MovingAverage,'SeasonalMA':SeasonalMovingAverage,'Croston-SBA':CrostonSBA,'ETS':ETS,'DirectGBM':lambda:DirectGBM(quantiles=(.5,.75,.9,.95))}; rows=[]
for name,f in models.items():
 r=rolling_origin(p,f,cfg); r['model']=name; rows.append(r)
allr=pd.concat(rows,ignore_index=True); metrics=pd.concat([score(x) .assign(model=n) for n,x in allr.groupby('model')],ignore_index=True); base=float(metrics.loc[metrics.model=='SeasonalNaive','wape'].iloc[0]); metrics['improvement_vs_seasonal_naive']=1-metrics.wape/base; Path('reports').mkdir(exist_ok=True); metrics.to_csv('reports/metrics_models.csv',index=False); allr[allr.model=='DirectGBM'].to_csv('reports/backtest_gbm.csv',index=False); print(metrics.to_string(index=False))
