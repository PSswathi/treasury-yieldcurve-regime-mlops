import os
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).parent

FILES = {
"requirements.txt": """
pandas
numpy
requests
scikit-learn
xgboost
boto3
sagemaker
pytest
ruff
joblib
""",

"pyproject.toml": """
[project]
name = "horizoncapital-mlops"
version = "0.1.0"
requires-python = ">=3.10"

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
""",

"src/horizoncapital/__init__.py": "__all__ = []\n",
"src/horizoncapital/io/__init__.py": "__all__ = []\n",
"src/horizoncapital/features/__init__.py": "__all__ = []\n",
"src/horizoncapital/data/__init__.py": "__all__ = []\n",
"src/horizoncapital/modeling/__init__.py": "__all__ = []\n",

"src/horizoncapital/config.py": """
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class ProjectConfig:
    fred_api_key: str = os.getenv("FRED_API_KEY", "")
    start_date: str = "2005-01-01"
    end_date: str = "2024-12-31"

    target_series_id: str = "STLFSI4"

    feature_series_ids = [
        "UNRATE","CPIAUCSL","FEDFUNDS","GDPC1",
        "T10Y2Y","T10Y3M","DRCCLACBS","DRSFRMACBS"
    ]

    horizons = [6,12,18,24]
""",

"src/horizoncapital/io/fred_client.py": """
import requests
import pandas as pd

class FredClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"

    def fetch(self, series_id, start, end):
        if not self.api_key:
            raise ValueError("FRED_API_KEY is missing. In PowerShell: $env:FRED_API_KEY='YOUR_KEY'")
        params = {
            "series_id": series_id,
            "api_key": self.api_key,
            "file_type":"json",
            "observation_start":start,
            "observation_end":end
        }
        r = requests.get(self.base_url, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json()["observations"]

        df = pd.DataFrame(obs)[["date","value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"].replace(".", pd.NA), errors="coerce")
        df.rename(columns={"value":series_id}, inplace=True)
        return df
""",

"src/horizoncapital/features/monthly_align.py": """
import pandas as pd

def to_monthly(df, col, how="last"):
    df = df.dropna().copy()
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    g = df.groupby("month")[col]
    if how == "last":
        out = g.last()
    elif how == "mean":
        out = g.mean()
    elif how == "sum":
        out = g.sum()
    else:
        raise ValueError("how must be last|mean|sum")
    return out.reset_index().rename(columns={"month":"ds"})
""",

"src/horizoncapital/features/feature_engineering.py": """
def add_lags(df, cols, lags=(1,3,6,12)):
    df = df.copy()
    for c in cols:
        for l in lags:
            df[f"{c}_lag{l}"] = df[c].shift(l)
    return df

def add_rolling(df, cols, windows=(3,6,12)):
    df = df.copy()
    for c in cols:
        for w in windows:
            df[f"{c}_rollmean{w}"] = df[c].rolling(w).mean()
            df[f"{c}_rollstd{w}"] = df[c].rolling(w).std()
    return df
""",

"src/horizoncapital/data/build_supervised.py": """
import pandas as pd

def build_supervised(df, target, horizons):
    out=[]
    for h in horizons:
        tmp=df.copy()
        tmp["horizon"]=h
        tmp["y"]=tmp[target].shift(-h)
        out.append(tmp)
    return pd.concat(out, ignore_index=True).dropna()
""",

"src/horizoncapital/modeling/train_xgb.py": """
from xgboost import XGBRegressor

def train(df):
    X=df.drop(columns=["ds","y"])
    y=df["y"]

    model=XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X,y)
    return model
""",

"pipelines/run_pipeline.py": """
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
""",

".github/workflows/ci.yml": """
name: ci

on:
  push:
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest -q
""",

"tests/test_basic.py": """
def test_math():
    assert 2 + 2 == 4
"""
}

def main():
    for path, content in FILES.items():
        full = ROOT / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(dedent(content).lstrip(), encoding="utf-8")
        print("CREATED:", path)

if __name__ == "__main__":
    main()
