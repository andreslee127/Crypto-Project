"""Download public Binance Vision monthly files for USDT-M perpetuals, with caching,
retry with backoff and SHA-256 checksum verification. No API keys, no exchange connection.

    python -m s2p.download

Stages: (1) list all USDT-M symbols, including delisted ones; (2) daily klines for every
symbol (used only to rank the point-in-time universe); (3) hourly klines and funding rates
for each symbol in the months it is in the universe, plus one warm-up month before.
"""
import concurrent.futures as cf
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

from s2p.config import load_config

UM = "data/futures/um/monthly"


def http_get(url, retries=6):
    """GET with exponential backoff. Returns bytes, or None if the file does not exist (404)."""
    url = urllib.parse.quote(url, safe=":/?&=")      # some newer symbols have non-ASCII names
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = e
        time.sleep(min(2 ** k, 30))
    raise RuntimeError(f"giving up on {url}: {err}")


def list_prefix(cfg, prefix):
    """All keys and sub-folders under an S3 prefix (follows pagination)."""
    keys, folders, marker = [], [], ""
    while True:
        url = f"{cfg['data']['listing_url']}?delimiter=/&prefix={prefix}&marker={marker}"
        xml = http_get(url).decode()
        k = re.findall(r"<Key>([^<]+)</Key>", xml)
        p = re.findall(r"<Prefix>([^<]+)</Prefix>", xml)[1:]      # first <Prefix> is the query
        keys += k
        folders += p
        if "<IsTruncated>true</IsTruncated>" not in xml:
            return keys, folders
        nxt = re.findall(r"<NextMarker>([^<]+)</NextMarker>", xml)
        marker = nxt[0] if nxt else (k + p)[-1]


def fetch(cfg, key):
    """Download one zip (and its .CHECKSUM) into the raw cache; verify SHA-256.

    Returns 'cached', 'downloaded' or 'missing'. A checksum mismatch raises.
    """
    raw = cfg["paths"]["raw"]
    path, csum_path = raw / key, raw / (key + ".CHECKSUM")
    if path.exists() and csum_path.exists() and _verify(path, csum_path):
        return "cached"
    csum = http_get(f"{cfg['data']['base_url']}/{key}.CHECKSUM")
    if csum is None:
        return "missing"
    for _ in range(3):
        body = http_get(f"{cfg['data']['base_url']}/{key}")
        if body is None:
            return "missing"
        if hashlib.sha256(body).hexdigest() == csum.decode().split()[0]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
            csum_path.write_bytes(csum)
            return "downloaded"
    raise RuntimeError(f"checksum mismatch three times for {key}")


def _verify(path, csum_path):
    return hashlib.sha256(path.read_bytes()).hexdigest() == csum_path.read_bytes().split()[0].decode()


def months_between(first, last):
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def kline_key(sym, interval, month):
    return f"{UM}/klines/{sym}/{interval}/{sym}-{interval}-{month}.zip"


def daily_kline_key(sym, interval, day):
    return f"data/futures/um/daily/klines/{sym}/{interval}/{sym}-{interval}-{day}.zip"


def incomplete_days(cfg, sym, month):
    """Days of a monthly hourly file with fewer than 24 bars (some monthly files are truncated;
    Binance Vision also publishes daily files, which are used to repair them)."""
    from s2p import data
    p = cfg["paths"]["raw"] / kline_key(sym, cfg["data"]["interval"], month)
    if not p.exists():
        return []
    k = data.read_klines(p)
    counts = pd.to_datetime(k["open_time"], unit="ms").dt.strftime("%Y-%m-%d").value_counts()
    days = pd.date_range(pd.Period(month).start_time, pd.Period(month).end_time, freq="D").strftime("%Y-%m-%d")
    return [d for d in days if counts.get(d, 0) < 24]


def funding_key(sym, month):
    return f"{UM}/fundingRate/{sym}/{sym}-fundingRate-{month}.zip"


def fetch_all(cfg, keys, label, workers=16):
    """Fetch many files. Files verified in an earlier run (manifest) are not re-hashed."""
    manifest_path = cfg["paths"]["raw"] / "verified.json"
    verified = set(json.loads(manifest_path.read_text())) if manifest_path.exists() else set()
    status = {k: "cached" for k in keys if k in verified and (cfg["paths"]["raw"] / k).exists()}
    todo = [k for k in keys if k not in status]
    with cf.ThreadPoolExecutor(workers) as ex:
        for key, st in zip(todo, ex.map(lambda k: fetch(cfg, k), todo)):
            status[key] = st
    verified |= {k for k, s in status.items() if s in ("cached", "downloaded")}
    manifest_path.write_text(json.dumps(sorted(verified)))
    counts = pd.Series(status).value_counts().to_dict()
    print(f"{label}: {len(keys)} files {counts}")
    return [k for k, s in status.items() if s == "missing"]


def list_symbols(cfg):
    """USDT-margined perpetual symbols ever listed (delisted ones stay in Binance Vision)."""
    cache = cfg["paths"]["raw"] / "symbols.json"
    if cache.exists():
        return json.loads(cache.read_text())
    _, folders = list_prefix(cfg, f"{UM}/klines/")
    syms = sorted(f.split("/")[-2] for f in folders)
    # perpetuals only: delivery contracts contain '_', settled/renamed contracts end in SETTLED
    syms = [s for s in syms if s.endswith("USDT") and "_" not in s]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(syms))
    return syms


def available_months(cfg, syms, interval):
    """Months with a file for each symbol, from the bucket listing (cached)."""
    cache = cfg["paths"]["raw"] / f"months_{interval}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    def one(s):
        keys, _ = list_prefix(cfg, f"{UM}/klines/{s}/{interval}/")
        return sorted(re.findall(r"-(\d{4}-\d{2})\.zip$", k)[0] for k in keys if k.endswith(".zip"))
    with cf.ThreadPoolExecutor(16) as ex:
        out = dict(zip(syms, ex.map(one, syms)))
    cache.write_text(json.dumps(out))
    return out


def main():
    from s2p import data
    cfg = load_config()
    syms = list_symbols(cfg)
    print(f"{len(syms)} USDT-M perpetual symbols in the listing (including delisted)")
    d = cfg["data"]
    rank_months = months_between(d["first_month"], pd.Period(d["last_month"]) - 1)
    avail = available_months(cfg, syms, "1d")
    keys = [kline_key(s, "1d", m) for s in syms for m in avail[s] if m in rank_months]
    missing = fetch_all(cfg, keys, "daily klines (universe ranking)")

    universe = data.build_universe(cfg)            # point-in-time: month m ranked on month m-1
    need = set()
    for month, members in universe.items():
        prev = str(pd.Period(month) - 1)
        for s in members:
            need |= {(s, month), (s, prev)}
    need = sorted(need)
    missing += fetch_all(cfg, [kline_key(s, d["interval"], m) for s, m in need], "hourly klines")
    missing += fetch_all(cfg, [funding_key(s, m) for s, m in need], "funding rates")
    (cfg["paths"]["raw"] / "missing.json").write_text(json.dumps(sorted(missing), indent=0))
    print(f"{len(missing)} files listed as missing -> {cfg['paths']['raw'] / 'missing.json'}")

    # repair truncated monthly files with daily files (404 = before listing / after delisting)
    days = [(s, day) for s, m in need for day in incomplete_days(cfg, s, m)]
    no_daily = fetch_all(cfg, [daily_kline_key(s, d["interval"], day) for s, day in days],
                         "daily hourly-kline files for incomplete days")
    print(f"{len(days) - len(no_daily)} incomplete days repaired from daily files; "
          f"{len(no_daily)} have no daily file (symbol not trading)")


if __name__ == "__main__":
    main()
