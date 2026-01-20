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
