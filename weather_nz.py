#!/usr/bin/env python3

# testing comment
"""
Fetches daily weather data for major New Zealand cities from the Open-Meteo API
and saves a cleaned dataset to CSV.

Usage examples:
  - Default (today's date):
      python weather_nz.py
  - Specific date:
      python weather_nz.py --date 2025-01-31
  - Date range:
      python weather_nz.py --start 2025-01-01 --end 2025-01-31
  - Custom output path:
      python weather_nz.py --out data/daily_weather_nz.csv
  - Filter cities:
      python weather_nz.py --cities Auckland Wellington

Requires: requests, pandas
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd
import requests


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Pacific/Auckland"


@dataclass(frozen=True)
class City:
    name: str
    region: str
    latitude: float
    longitude: float


def nz_cities() -> List[City]:
    # Major NZ cities and large centres (approximate CBD coordinates)
    # Source: public coordinates, rounded for practicality
    return [
        City("Auckland", "Auckland", -36.8485, 174.7633),
        City("Wellington", "Wellington", -41.2866, 174.7756),
        City("Christchurch", "Canterbury", -43.5321, 172.6362),
        City("Hamilton", "Waikato", -37.7870, 175.2793),
        City("Tauranga", "Bay of Plenty", -37.6878, 176.1651),
        City("Dunedin", "Otago", -45.8788, 170.5028),
        City("Palmerston North", "Manawatū-Whanganui", -40.3523, 175.6082),
        City("Napier", "Hawke's Bay", -39.4928, 176.9120),
        City("Nelson", "Nelson", -41.2706, 173.2840),
        City("Rotorua", "Bay of Plenty", -38.1370, 176.2510),
        City("New Plymouth", "Taranaki", -39.0556, 174.0752),
        City("Invercargill", "Southland", -46.4132, 168.3538),
        City("Whangarei", "Northland", -35.7251, 174.3237),
    ]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    today = dt.date.today()
    p = argparse.ArgumentParser(description="Fetch daily NZ city weather and save cleaned dataset.")
    date_group = p.add_mutually_exclusive_group()
    date_group.add_argument("--date", type=str, help="Single date YYYY-MM-DD (default: today)")
    date_group.add_argument("--start", type=str, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", type=str, help="End date YYYY-MM-DD (inclusive, required if --start)")
    p.add_argument("--out", type=str, default="data/daily_weather_nz.csv", help="Output CSV path")
    p.add_argument(
        "--cities",
        nargs="*",
        default=None,
        help="Optional subset of city names to fetch (e.g., Auckland Wellington)",
    )
    p.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout seconds (default 20)")
    p.add_argument("--retries", type=int, default=3, help="HTTP retries on failure (default 3)")
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep seconds between requests (default 0.2)")

    args = p.parse_args(list(argv) if argv is not None else None)

    # Normalize and validate dates
    if args.date:
        start = end = _parse_date(args.date)
    elif args.start:
        if not args.end:
            p.error("--end is required when using --start")
        start = _parse_date(args.start)
        end = _parse_date(args.end)
        if end < start:
            p.error("--end must be on/after --start")
    else:
        start = end = today

    args.start_date = start
    args.end_date = end

    # Normalize city filter to set of lowercase names for matching
    if args.cities:
        args.cities = {c.strip().lower() for c in args.cities}
    return args


def _parse_date(s: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid date '{s}', expected YYYY-MM-DD") from e


def build_params(start: dt.date, end: dt.date) -> Dict[str, str]:
    # Daily variables chosen for breadth and general utility
    daily_vars = [
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "precipitation_sum",
        "rain_sum",
        "showers_sum",
        "snowfall_sum",
        "precipitation_hours",
        "windspeed_10m_max",
        "windgusts_10m_max",
        "winddirection_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        # "sunshine_duration",  # Not available in all models; omit to reduce missingness
    ]

    return {
        "timezone": TIMEZONE,
        "daily": ",".join(daily_vars),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }


def http_get_json(url: str, params: Dict[str, str], timeout: float, retries: int) -> Dict:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < retries:
                # Exponential backoff with jitter
                time.sleep(min(2 ** attempt, 8) + 0.1 * attempt)
            else:
                break
    assert last_err is not None
    raise last_err


def fetch_city_daily(city: City, start: dt.date, end: dt.date, timeout: float, retries: int) -> pd.DataFrame:
    params = build_params(start, end)
    params.update({"latitude": city.latitude, "longitude": city.longitude})
    doc = http_get_json(OPEN_METEO_URL, params=params, timeout=timeout, retries=retries)

    # Basic validation
    daily = doc.get("daily")
    if not daily or "time" not in daily:
        raise ValueError(f"Unexpected API shape for {city.name}: {json.dumps(doc)[:300]}...")

    dates = pd.to_datetime(pd.Series(daily["time"]) , utc=False).dt.date
    # Build dataframe with all daily fields present
    df = pd.DataFrame({k: v for k, v in daily.items() if k != "time"})
    df.insert(0, "date", dates)

    # Attach city metadata
    df.insert(0, "city", city.name)
    df.insert(1, "region", city.region)
    df["latitude"] = float(doc.get("latitude", city.latitude))
    df["longitude"] = float(doc.get("longitude", city.longitude))
    df["elevation_m"] = doc.get("elevation")
    df["generationtime_ms"] = doc.get("generationtime_ms")
    df["utc_offset_seconds"] = doc.get("utc_offset_seconds")
    return df


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Standardize column order and types; ensure numeric columns are numeric
    preferred_order = [
        "city",
        "region",
        "date",
        "latitude",
        "longitude",
        "elevation_m",
        "temperature_2m_max",
        "temperature_2m_min",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "precipitation_sum",
        "rain_sum",
        "showers_sum",
        "snowfall_sum",
        "precipitation_hours",
        "windspeed_10m_max",
        "windgusts_10m_max",
        "winddirection_10m_dominant",
        "shortwave_radiation_sum",
        "et0_fao_evapotranspiration",
        "generationtime_ms",
        "utc_offset_seconds",
    ]

    for col in df.columns:
        if col not in {"city", "region", "date"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Reorder where possible; keep any extras at the end
    cols = [c for c in preferred_order if c in df.columns] + [c for c in df.columns if c not in preferred_order]
    df = df[cols]
    # Sort for deterministic output
    df = df.sort_values(["date", "city"]).reset_index(drop=True)
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    cities = nz_cities()
    if args.cities:
        allowed = args.cities
        cities = [c for c in cities if c.name.lower() in allowed]
        if not cities:
            print("No matching cities after filter.", file=sys.stderr)
            return 2

    frames: List[pd.DataFrame] = []
    for i, city in enumerate(cities, start=1):
        try:
            df_city = fetch_city_daily(city, args.start_date, args.end_date, timeout=args.timeout, retries=args.retries)
            frames.append(df_city)
        except Exception as e:
            print(f"Error fetching {city.name}: {e}", file=sys.stderr)
        if i < len(cities):
            time.sleep(args.sleep)

    if not frames:
        print("No data fetched. Exiting.", file=sys.stderr)
        return 1

    df = pd.concat(frames, ignore_index=True)
    df = clean_columns(df)
    out_path = Path(args.out)
    save_csv(df, out_path)
    print(f"Saved {len(df)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

