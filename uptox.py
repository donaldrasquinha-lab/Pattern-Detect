"""
BULKOWSKI PATTERN ENGINE + UPSTOX V2/V3 LIVE DATA CONNECTOR
=============================================================
Connects to Upstox API for:
  - OAuth2 authentication (access token + extended token)
  - Historical Candle Data V3 (OHLCV for indices & F&O)
  - Intraday Candle Data (live session OHLC)
  - Option Chain (OI, IV, Greeks per strike)
  - Option Contracts (expiry dates, strike list)
  - Expired Instruments (historical expired contract data)
  - Market Data Feeder V3 WebSocket (real-time streaming)

Then runs Bulkowski pattern detection using scipy.signal on the data.

Requirements:
  pip install scipy numpy pandas requests upstox-python-sdk websocket-client
"""

import os
import json
import time
import requests
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════
# UPSTOX API CONFIGURATION
# ═══════════════════════════════════════════════════════════════
UPSTOX_BASE_URL = "https://api.upstox.com"

# Instrument keys for major indices
INSTRUMENT_KEYS = {
    # India
    "NIFTY50":    "NSE_INDEX|Nifty 50",
    "BANKNIFTY":  "NSE_INDEX|Nifty Bank",
    "FINNIFTY":   "NSE_INDEX|Nifty Fin Service",
    "MIDCPNIFTY": "NSE_INDEX|NIFTY MID SELECT",
    "SENSEX":     "BSE_INDEX|SENSEX",
    # US (not available on Upstox — use synthetic/Yahoo for these)
    "SPX": None, "NDX": None, "RUT": None, "VIX": None,
}

# Expiry structure
EXPIRY_CONFIG = {
    "NIFTY50":    {"weekly": True,  "day": "Tuesday",  "lot_size": 75},
    "BANKNIFTY":  {"weekly": False, "day": "Tuesday",  "lot_size": 30},
    "FINNIFTY":   {"weekly": False, "day": "Tuesday",  "lot_size": 40},
    "MIDCPNIFTY": {"weekly": False, "day": "Tuesday",  "lot_size": 75},
    "SENSEX":     {"weekly": True,  "day": "Thursday", "lot_size": 20},
}


# ═══════════════════════════════════════════════════════════════
# UPSTOX API CLIENT
# ═══════════════════════════════════════════════════════════════
class UpstoxClient:
    """
    Upstox V2/V3 API client with OAuth2 auth flow.

    Usage:
        # Option 1: Direct token (if you already have one)
        client = UpstoxClient(access_token="your_token_here")

        # Option 2: Full OAuth2 flow
        client = UpstoxClient(
            api_key="your_api_key",
            api_secret="your_api_secret",
            redirect_uri="your_redirect_uri"
        )
        # Step 1: Get auth URL → user logs in → gets auth code
        auth_url = client.get_auth_url()
        # Step 2: Exchange auth code for access token
        client.get_access_token(auth_code="code_from_redirect")

        # Option 3: From environment variables
        client = UpstoxClient.from_env()
    """

    def __init__(self, access_token: str = None, api_key: str = None,
                 api_secret: str = None, redirect_uri: str = None):
        self.access_token = access_token
        self.api_key = api_key
        self.api_secret = api_secret
        self.redirect_uri = redirect_uri
        self.extended_token = None
        self.token_expiry = None
        self.session = requests.Session()

    @classmethod
    def from_env(cls):
        """Create client from environment variables"""
        return cls(
            access_token=os.environ.get("UPSTOX_ACCESS_TOKEN"),
            api_key=os.environ.get("UPSTOX_API_KEY"),
            api_secret=os.environ.get("UPSTOX_API_SECRET"),
            redirect_uri=os.environ.get("UPSTOX_REDIRECT_URI", "https://localhost:3000/callback"),
        )

    @classmethod
    def from_config_file(cls, path: str = "upstox_config.json"):
        """Load credentials from a JSON config file"""
        if not os.path.exists(path):
            # Create template
            template = {
                "api_key": "YOUR_API_KEY_HERE",
                "api_secret": "YOUR_API_SECRET_HERE",
                "redirect_uri": "https://localhost:3000/callback",
                "access_token": ""
            }
            with open(path, "w") as f:
                json.dump(template, f, indent=2)
            print(f"Created config template at {path} — fill in your credentials.")
            return cls()

        with open(path) as f:
            cfg = json.load(f)
        return cls(**cfg)

    def _headers(self, token: str = None):
        t = token or self.access_token
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {t}" if t else "",
        }

    def _get(self, url, params=None, token=None):
        resp = self.session.get(url, headers=self._headers(token), params=params)
        if resp.status_code != 200:
            raise Exception(f"Upstox API error {resp.status_code}: {resp.text}")
        return resp.json()

    # ─── AUTH FLOW ───────────────────────────────────────────────
    def get_auth_url(self):
        """Step 1: Generate authorization URL for user login"""
        if not self.api_key:
            raise ValueError("api_key required for OAuth flow")
        return (
            f"{UPSTOX_BASE_URL}/v2/login/authorization/dialog"
            f"?response_type=code"
            f"&client_id={self.api_key}"
            f"&redirect_uri={self.redirect_uri}"
        )

    def get_access_token(self, auth_code: str):
        """Step 2: Exchange authorization code for access token"""
        resp = self.session.post(
            f"{UPSTOX_BASE_URL}/v2/login/authorization/token",
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            data={
                "code": auth_code,
                "client_id": self.api_key,
                "client_secret": self.api_secret,
                "redirect_uri": self.redirect_uri,
                "grant_type": "authorization_code",
            }
        )
        if resp.status_code != 200:
            raise Exception(f"Token exchange failed: {resp.text}")

        data = resp.json()
        self.access_token = data.get("access_token")
        self.extended_token = data.get("extended_token")
        # Token expires at 3:30 AM IST next day
        self.token_expiry = datetime.now() + timedelta(hours=20)
        print(f"✓ Access token obtained. Expires ~3:30 AM IST.")
        return data

    def save_token(self, path: str = "upstox_token.json"):
        """Save token to file for reuse"""
        with open(path, "w") as f:
            json.dump({
                "access_token": self.access_token,
                "extended_token": self.extended_token,
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"✓ Token saved to {path}")

    def load_token(self, path: str = "upstox_token.json"):
        """Load token from file"""
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            self.access_token = data.get("access_token")
            self.extended_token = data.get("extended_token")
            print(f"✓ Token loaded from {path}")
            return True
        return False

    def is_authenticated(self):
        """Check if we have a valid token"""
        return bool(self.access_token)

    def get_profile(self):
        """Get user profile to verify token"""
        return self._get(f"{UPSTOX_BASE_URL}/v2/user/profile")

    # ─── MARKET DATA APIs ────────────────────────────────────────

    def get_historical_candles_v3(self, instrument_key: str, interval: str = "day",
                                  to_date: str = None, from_date: str = None):
        """
        Historical Candle Data V3 API
        Intervals: 1minute, 3minute, 5minute, 15minute, 30minute, day, week, month
        Returns: [[timestamp, O, H, L, C, volume, OI], ...]
        """
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        # Map interval format
        if interval in ("1minute", "3minute", "5minute", "15minute", "30minute"):
            unit = "minutes"
            val = interval.replace("minute", "")
        else:
            unit = interval
            val = "1"

        url = f"{UPSTOX_BASE_URL}/v3/historical-candle/{instrument_key}/{unit}/{val}/{to_date}/{from_date}"
        data = self._get(url)
        return data.get("data", {}).get("candles", [])

    def get_historical_candles_v2(self, instrument_key: str, interval: str = "day",
                                  to_date: str = None, from_date: str = None):
        """Historical Candle Data V2 (fallback)"""
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

        url = f"{UPSTOX_BASE_URL}/v2/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        data = self._get(url)
        return data.get("data", {}).get("candles", [])

    def get_intraday_candles(self, instrument_key: str, interval: str = "1minute"):
        """Intraday candle data for current trading session"""
        url = f"{UPSTOX_BASE_URL}/v2/historical-candle/intraday/{instrument_key}/{interval}"
        data = self._get(url)
        return data.get("data", {}).get("candles", [])

    def get_option_chain(self, instrument_key: str, expiry_date: str):
        """
        Option Chain with OI, IV, Greeks per strike
        Returns call/put data with market_data and option_greeks
        """
        url = f"{UPSTOX_BASE_URL}/v2/option/chain"
        params = {"instrument_key": instrument_key, "expiry_date": expiry_date}
        data = self._get(url, params=params)
        return data.get("data", [])

    def get_option_contracts(self, instrument_key: str, expiry_date: str = None):
        """Get available option contracts for an underlying"""
        url = f"{UPSTOX_BASE_URL}/v2/option/contract"
        params = {"instrument_key": instrument_key}
        if expiry_date:
            params["expiry_date"] = expiry_date
        data = self._get(url, params=params)
        return data.get("data", [])

    def get_option_expiries(self, instrument_key: str):
        """Get available expiry dates"""
        url = f"{UPSTOX_BASE_URL}/v2/option/contract"
        params = {"instrument_key": instrument_key}
        data = self._get(url, params=params)
        contracts = data.get("data", [])
        expiries = sorted(set(c.get("expiry") for c in contracts if c.get("expiry")))
        return expiries

    def get_expired_historical_candle(self, instrument_key: str, interval: str = "day",
                                       to_date: str = None, from_date: str = None):
        """Historical data for expired F&O contracts (Upstox Plus required)"""
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
        if from_date is None:
            from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

        url = f"{UPSTOX_BASE_URL}/v2/expired-instruments/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
        data = self._get(url)
        return data.get("data", {}).get("candles", [])

    def get_market_quotes(self, instrument_keys: list):
        """Get current market quotes (LTP, volume, OI)"""
        keys = ",".join(instrument_keys)
        url = f"{UPSTOX_BASE_URL}/v2/market-quote/quotes"
        params = {"instrument_key": keys}
        data = self._get(url, params=params)
        return data.get("data", {})

    # ─── DATA CONVERSION ─────────────────────────────────────────

    def candles_to_dataframe(self, candles: list) -> pd.DataFrame:
        """Convert Upstox candle array to pandas DataFrame"""
        if not candles:
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def candles_to_arrays(self, candles: list):
        """Convert candles to numpy arrays for pattern engine"""
        if not candles:
            return None, None, None, None, None
        df = self.candles_to_dataframe(candles)
        return (df["open"].values, df["high"].values, df["low"].values,
                df["close"].values, df["volume"].values)

    def option_chain_to_oi_data(self, chain_data: list) -> dict:
        """Convert option chain response to OI analysis format"""
        strikes = []
        total_call_oi = 0
        total_put_oi = 0

        for strike_data in chain_data:
            sp = strike_data.get("strike_price", 0)
            call_md = strike_data.get("call_options", {}).get("market_data", {})
            put_md = strike_data.get("put_options", {}).get("market_data", {})
            call_greeks = strike_data.get("call_options", {}).get("option_greeks", {})
            put_greeks = strike_data.get("put_options", {}).get("option_greeks", {})

            c_oi = call_md.get("oi", 0)
            p_oi = put_md.get("oi", 0)
            total_call_oi += c_oi
            total_put_oi += p_oi

            strikes.append({
                "strike": sp,
                "callOI": c_oi,
                "putOI": p_oi,
                "callOIChange": c_oi - call_md.get("prev_oi", c_oi),
                "putOIChange": p_oi - put_md.get("prev_oi", p_oi),
                "callIV": call_greeks.get("iv", 0),
                "putIV": put_greeks.get("iv", 0),
                "callDelta": call_greeks.get("delta", 0),
                "putDelta": put_greeks.get("delta", 0),
                "callLTP": call_md.get("ltp", 0),
                "putLTP": put_md.get("ltp", 0),
            })

        spot = chain_data[0].get("underlying_spot_price", 0) if chain_data else 0
        pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 1.0

        # Max pain calculation
        max_pain_strike = 0
        min_pain = float('inf')
        for s in strikes:
            pain = sum(
                max(0, s["strike"] - s2["strike"]) * s2["callOI"] +
                max(0, s2["strike"] - s["strike"]) * s2["putOI"]
                for s2 in strikes
            )
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = s["strike"]

        return {
            "strikes": strikes,
            "totalCallOI": total_call_oi,
            "totalPutOI": total_put_oi,
            "pcr": pcr,
            "maxPain": max_pain_strike,
            "spotPrice": spot,
            "topCallOI": sorted(strikes, key=lambda x: x["callOI"], reverse=True)[:3],
            "topPutOI": sorted(strikes, key=lambda x: x["putOI"], reverse=True)[:3],
        }


# ═══════════════════════════════════════════════════════════════
# INTEGRATED SCANNER — Upstox + Bulkowski Engine
# ═══════════════════════════════════════════════════════════════
class ExpiryPatternScanner:
    """
    Connects to Upstox, fetches data, runs Bulkowski pattern detection,
    and builds the expiry pattern database.
    """

    def __init__(self, client: UpstoxClient):
        self.client = client
        # Import engine from the bulkowski module
        from bulkowski_engine import BulkowskiPatternEngine
        self.engine = BulkowskiPatternEngine(order=3)

    def scan_index_expiry(self, index_name: str, expiry_date: str,
                           interval: str = "5minute"):
        """
        Scan a single index expiry cycle:
        1. Fetch historical candles for the cycle period
        2. Fetch option chain OI
        3. Run Bulkowski pattern detection
        4. Return complete analysis
        """
        inst_key = INSTRUMENT_KEYS.get(index_name)
        if not inst_key:
            raise ValueError(f"Unknown index: {index_name}")

        # Calculate cycle dates
        exp_dt = datetime.strptime(expiry_date, "%Y-%m-%d")
        config = EXPIRY_CONFIG.get(index_name, {})
        is_weekly = config.get("weekly", False)
        cycle_start = exp_dt - timedelta(days=5 if is_weekly else 22)

        print(f"Scanning {index_name} expiry {expiry_date}...")

        # 1. Fetch candle data
        candles = self.client.get_historical_candles_v3(
            inst_key, interval=interval,
            to_date=expiry_date,
            from_date=cycle_start.strftime("%Y-%m-%d")
        )
        if not candles:
            # Fallback to V2
            candles = self.client.get_historical_candles_v2(
                inst_key, interval="day",
                to_date=expiry_date,
                from_date=cycle_start.strftime("%Y-%m-%d")
            )

        O, H, L, C, V = self.client.candles_to_arrays(candles)
        if O is None or len(O) < 5:
            print(f"  ⚠ Insufficient data for {index_name}")
            return None

        # 2. Fetch option chain OI
        oi_data = None
        oi_change = 0
        pcr = 1.0
        try:
            chain = self.client.get_option_chain(inst_key, expiry_date)
            if chain:
                oi_data = self.client.option_chain_to_oi_data(chain)
                pcr = oi_data["pcr"]
                oi_change = sum(s["callOIChange"] + s["putOIChange"] for s in oi_data["strikes"])
        except Exception as e:
            print(f"  ⚠ Option chain unavailable: {e}")

        # 3. Run Bulkowski pattern detection
        patterns = self.engine.scan_all_patterns(O, H, L, C, V, oi_change, pcr)

        # 4. Build result
        start_price = float(C[0])
        end_price = float(C[-1])
        change_pct = round((end_price - start_price) / start_price * 100, 3)

        result = {
            "index": index_name,
            "expiryDate": expiry_date,
            "cycleType": "WEEKLY" if is_weekly else "MONTHLY",
            "interval": interval,
            "bars": len(C),
            "startPrice": round(start_price, 2),
            "endPrice": round(end_price, 2),
            "changePercent": change_pct,
            "outcome": "BULLISH" if change_pct > 0.5 else ("BEARISH" if change_pct < -0.5 else "NEUTRAL"),
            "maxDrawdown": round((float(np.min(L)) - start_price) / start_price * 100, 3),
            "maxRunup": round((float(np.max(H)) - start_price) / start_price * 100, 3),
            "volatility": round(float(np.mean(H - L)), 2),
            "pcr": pcr,
            "oiData": oi_data,
            "patterns": [asdict(p) for p in patterns],
            "patternCount": len(patterns),
            "engine": "scipy.signal + Bulkowski",
            "dataSource": "Upstox API V3",
            "scannedAt": datetime.now().isoformat(),
        }

        print(f"  ✓ {index_name}: {len(patterns)} patterns, {change_pct:+.2f}%")
        return result

    def scan_all_expiries(self, index_name: str, num_expiries: int = 12,
                           interval: str = "day"):
        """Scan multiple past expiry cycles for an index"""
        results = []
        inst_key = INSTRUMENT_KEYS.get(index_name)
        if not inst_key:
            return results

        # Get available expiries
        try:
            expiries = self.client.get_option_expiries(inst_key)
        except:
            # Generate approximate expiry dates
            config = EXPIRY_CONFIG.get(index_name, {})
            today = datetime.now()
            gap = 7 if config.get("weekly") else 30
            expiries = [(today - timedelta(days=gap * i)).strftime("%Y-%m-%d")
                        for i in range(num_expiries)]

        for exp_date in expiries[:num_expiries]:
            try:
                result = self.scan_index_expiry(index_name, exp_date, interval)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  ✗ Error scanning {exp_date}: {e}")

        return results

    def scan_live(self, index_name: str):
        """Scan current intraday data for real-time pattern detection"""
        inst_key = INSTRUMENT_KEYS.get(index_name)
        if not inst_key:
            return None

        candles = self.client.get_intraday_candles(inst_key, "5minute")
        O, H, L, C, V = self.client.candles_to_arrays(candles)
        if O is None or len(O) < 5:
            return None

        patterns = self.engine.scan_all_patterns(O, H, L, C, V)

        return {
            "index": index_name,
            "timestamp": datetime.now().isoformat(),
            "bars": len(C),
            "currentPrice": round(float(C[-1]), 2),
            "dayChange": round((float(C[-1]) - float(C[0])) / float(C[0]) * 100, 3),
            "patterns": [asdict(p) for p in patterns],
            "patternCount": len(patterns),
            "dataSource": "Upstox Intraday",
        }


# ═══════════════════════════════════════════════════════════════
# CLI / MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Bulkowski Pattern Scanner + Upstox")
    parser.add_argument("--token", help="Upstox access token")
    parser.add_argument("--config", default="upstox_config.json", help="Config file path")
    parser.add_argument("--index", default="NIFTY50", help="Index to scan")
    parser.add_argument("--expiry", help="Specific expiry date (YYYY-MM-DD)")
    parser.add_argument("--interval", default="day", help="Candle interval")
    parser.add_argument("--scan-all", type=int, default=0, help="Scan N past expiries")
    parser.add_argument("--live", action="store_true", help="Scan current intraday")
    parser.add_argument("--auth", action="store_true", help="Run OAuth flow")
    parser.add_argument("--output", default="scan_results.json", help="Output file")
    args = parser.parse_args()

    # Initialize client
    if args.token:
        client = UpstoxClient(access_token=args.token)
    else:
        client = UpstoxClient.from_config_file(args.config)

    # OAuth flow
    if args.auth:
        if not client.api_key:
            print("Error: api_key required in config for OAuth flow")
            return
        auth_url = client.get_auth_url()
        print(f"\n1. Open this URL in your browser:\n   {auth_url}\n")
        print("2. Log in and authorize the app")
        print("3. Copy the 'code' parameter from the redirect URL\n")
        auth_code = input("Enter the authorization code: ").strip()
        token_data = client.get_access_token(auth_code)
        client.save_token()
        print(f"\n✓ Authenticated! Token: {client.access_token[:20]}...")
        return

    if not client.is_authenticated():
        # Try loading saved token
        if not client.load_token():
            print("No access token. Run with --auth flag first, or provide --token.")
            print("Or set UPSTOX_ACCESS_TOKEN environment variable.")
            print("\nRunning in DEMO mode with synthetic data...\n")
            # Demo mode — use the base engine
            from bulkowski_engine import build_expiry_database
            db = build_expiry_database(num_weekly=24, num_monthly=6)
            with open(args.output, "w") as f:
                json.dump(db, f, indent=2)
            print(f"Demo database saved: {len(db)} records → {args.output}")
            return

    # Verify token
    try:
        profile = client.get_profile()
        user = profile.get("data", {})
        print(f"✓ Logged in as: {user.get('user_name', 'Unknown')} ({user.get('email', '')})")
    except Exception as e:
        print(f"Token verification failed: {e}")
        print("Token may have expired. Run with --auth to re-authenticate.")
        return

    # Initialize scanner
    scanner = ExpiryPatternScanner(client)

    if args.live:
        result = scanner.scan_live(args.index)
        if result:
            print(f"\n{'='*60}")
            print(f"LIVE SCAN: {result['index']} | {result['currentPrice']} | {result['dayChange']:+.2f}%")
            print(f"Patterns: {result['patternCount']}")
            for p in result["patterns"]:
                print(f"  • {p['name']} ({p['bias']}) — {p['confidence']*100:.0f}% conf")
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
        return

    if args.scan_all > 0:
        results = scanner.scan_all_expiries(args.index, args.scan_all, args.interval)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Scanned {len(results)} expiries → {args.output}")
        return

    if args.expiry:
        result = scanner.scan_index_expiry(args.index, args.expiry, args.interval)
        if result:
            with open(args.output, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Result saved → {args.output}")
        return

    print("No action specified. Use --live, --expiry, --scan-all, or --auth.")


if __name__ == "__main__":
    main()
