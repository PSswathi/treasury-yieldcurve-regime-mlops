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
