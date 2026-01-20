from horizoncapital.config import ProjectConfig
from horizoncapital.io.fred_client import FredClient
from horizoncapital.features.monthly_align import to_monthly
from horizoncapital.features.feature_engineering import add_lags, add_rolling
from horizoncapital.data.build_supervised import build_supervised
from horizoncapital.modeling.train_xgb import train

cfg=ProjectConfig()
fred=FredClient(cfg.fred_api_key)

print("Pulling target...")
tgt_raw=fred.fetch(cfg.target_series_id,cfg.start_date,cfg.end_date)
tgt=to_monthly(tgt_raw,cfg.target_series_id,how="last")

df=tgt.copy()

print("Pulling features...")
for s in cfg.feature_series_ids:
    r=fred.fetch(s,cfg.start_date,cfg.end_date)
    m=to_monthly(r,s,how="last")
    df=df.merge(m,on="ds",how="left")

df=df.sort_values("ds").ffill()

cols=[c for c in df.columns if c!="ds"]
df=add_lags(df,cols)
df=add_rolling(df,cols)

sup=build_supervised(df,cfg.target_series_id,cfg.horizons)

print("Training XGB baseline...")
model=train(sup)

print("SUCCESS ✅")
print("Supervised rows:", len(sup))
print("Columns:", len(sup.columns))
