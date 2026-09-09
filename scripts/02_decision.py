from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from rdf.decision.newsvendor import NewsvendorCosts,compare_policies,sensitivity
r=pd.read_csv('reports/backtest_gbm.csv'); c=NewsvendorCosts(); a=compare_policies(r,c); s=sensitivity(r); a.to_csv('reports/newsvendor_policies.csv',index=False); s.to_csv('reports/newsvendor_sensitivity.csv',index=False); print(a.to_string(index=False)); print(s.to_string(index=False))
