from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rdf.data.olist import load_demand_panel
from rdf.data.schema import panel_summary
p=load_demand_panel(); print(panel_summary(p).to_string(index=False)); print(f"Dropped categories: {p.attrs.get('dropped_categories',0)}"); Path('data').mkdir(exist_ok=True); p.to_parquet('data/panel.parquet',index=False)
