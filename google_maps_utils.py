#!/usr/bin/env python3
import os
import json
import time
import argparse
import unicodedata
import re
from glob import glob
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd
import jsonlines
from tqdm import tqdm
import json_repair
import random
from Levenshtein import distance,ratio
from scipy import stats
from openai import OpenAI

# -------------- NEW: external calls --------------
try:
    import requests
except ImportError:
    raise SystemExit("This script now uses 'requests'. Please install it: pip install requests")

# ---------------------------
# NEW: Google Maps helper
# ---------------------------
class GoogleMapsHelper:
    GEO_URL = "https://maps.googleapis.com/maps/api/geocode/json"
    DM_URL  = "https://maps.googleapis.com/maps/api/distancematrix/json"

    def __init__(self,
                 api_key: Optional[str],
                 cache_path: str,
                 language: str = "en",
                 region: str = "",
                 country: str = "",
                 sleep_sec: float = 0.2,
                 modes: List[str] = None):
        self.api_key = api_key or os.getenv("GOOGLE_MAPS_API_KEY", "")
        self.cache_path = cache_path
        self.language = language
        self.region = region
        self.country = country.lower().strip()  # e.g., "jp"
        self.sleep_sec = max(0.0, sleep_sec)
        self.modes = modes or ["walking", "transit", "driving", "bicycling"]
        self.session = requests.Session()

        self.cache = {"geocode": {}, "matrix": {}}
        if os.path.isfile(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache.update(json.load(f))
            except Exception:
                pass

        if not self.api_key:
            print("[WARN] GOOGLE_MAPS_API_KEY not set; will fall back to heuristics (no API calls).")

    def save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    # -------- geocoding --------
    def geocode(self, place: str) -> Optional[Dict[str, Any]]:
        """Return {'lat':..., 'lng':..., 'place_id':..., 'formatted':...} or None."""
        key = f"{place}|{self.language}|{self.region}|{self.country}"
        if key in self.cache["geocode"]:
            return self.cache["geocode"][key]

        if not self.api_key:
            # No API key → don’t attempt, cache None
            self.cache["geocode"][key] = None
            return None

        params = {"address": place, "key": self.api_key, "language": self.language}
        if self.region:
            params["region"] = self.region
        if self.country:
            params["components"] = f"country:{self.country}"

        resp = self.session.get(self.GEO_URL, params=params, timeout=20)
        time.sleep(self.sleep_sec)
        if resp.status_code != 200:
            self.cache["geocode"][key] = None
            return None
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            self.cache["geocode"][key] = None
            return None

        r0 = data["results"][0]
        loc = r0["geometry"]["location"]
        out = {
            "lat": loc["lat"],
            "lng": loc["lng"],
            "place_id": r0.get("place_id"),
            "formatted": r0.get("formatted_address", place)
        }
        self.cache["geocode"][key] = out
        return out

    # -------- distance matrix for one mode --------
    def _matrix(self, orig: Tuple[float,float], dest: Tuple[float,float], mode: str) -> Optional[Dict[str, Any]]:
        key = f"{orig[0]:.6f},{orig[1]:.6f}|{dest[0]:.6f},{dest[1]:.6f}|{mode}"
        if key in self.cache["matrix"]:
            return self.cache["matrix"][key]

        if not self.api_key:
            self.cache["matrix"][key] = None
            return None

        params = {
            "origins": f"{orig[0]},{orig[1]}",
            "destinations": f"{dest[0]},{dest[1]}",
            "mode": mode,
            "key": self.api_key,
            "language": self.language,
        }
        # departure_time impacts transit and driving (traffic)
        params["departure_time"] = "now"

        resp = self.session.get(self.DM_URL, params=params, timeout=20)
        time.sleep(self.sleep_sec)
        if resp.status_code != 200:
            self.cache["matrix"][key] = None
            return None
        data = resp.json()
        try:
            row = data["rows"][0]["elements"][0]
        except Exception:
            self.cache["matrix"][key] = None
            return None

        if row.get("status") != "OK":
            self.cache["matrix"][key] = None
            return None

        # Prefer duration_in_traffic for driving if available
        dur = row.get("duration_in_traffic", row.get("duration"))
        val = {
            "distance_meters": row["distance"]["value"],
            "duration_sec": dur["value"] if dur else row["duration"]["value"],
            "mode": mode
        }
        self.cache["matrix"][key] = val
        return val

    def best_mode(self, orig: Tuple[float,float], dest: Tuple[float,float]) -> Dict[str, Any]:
        """Return {'best_mode': str, 'best_duration_sec': int, 'distance_meters': int, 'all_modes': {...}}; heuristic fallback if API missing."""
        results = {}
        for m in self.modes:
            r = self._matrix(orig, dest, m)
            if r:
                results[m] = r

        if results:
            best = min(results.values(), key=lambda x: x["duration_sec"])
            return {
                "best_mode": best["mode"],
                "best_duration_sec": best["duration_sec"],
                "distance_meters": best["distance_meters"],
                "all_modes": {k: {"duration_sec": v["duration_sec"], "distance_meters": v["distance_meters"]} for k, v in results.items()}
            }

        # -------- Heuristic fallback (no key / failed API): decide by straight-line distance --------
        # Haversine-lite is overkill; use rough thresholds on Euclidean (lat/lng) since we don't have distances.
        # We'll default: <1.5 km → walking; <8 km → bicycling/transit; else driving.
        # Distance proxy (very rough): 1 deg lat ≈ 111 km; 1 deg lon ≈ 111 km * cos(lat)
        import math
        lat_km = (dest[0] - orig[0]) * 111.0
        lon_km = (dest[1] - orig[1]) * 111.0 * math.cos(math.radians((orig[0] + dest[0]) / 2.0))
        d_km = math.sqrt(lat_km * lat_km + lon_km * lon_km)
        if d_km <= 1.5:
            bm = "walking"
        elif d_km <= 8:
            bm = "bicycling"
        else:
            bm = "driving"
        return {
            "best_mode": bm,
            "best_duration_sec": None,
            "distance_meters": int(d_km * 1000),
            "all_modes": {}
        }
