#!/usr/bin/env python3
"""
Fetch PE/PB fundamental data for CSI500 stocks using DAILY frequency
(baostock only provides peTTM/pbMRQ on daily k-lines, not weekly).
Resamples to weekly (Friday) values. Saves to cache.
"""
import baostock as bs
import pandas as pd
import numpy as np
import json
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'baostock_cache')

def main():
    const_path = os.path.join(CACHE_DIR, 'csi500_constituents_history.json')
    with open(const_path) as f:
        data = json.load(f)
    all_stocks = data['all_unique_stocks']
    print(f"Total stocks: {len(all_stocks)}")

    cache_path = os.path.join(CACHE_DIR, 'csi500_weekly_fundamentals.pkl')

    # Check partial cache
    existing = {}
    if os.path.exists(cache_path):
        old = pd.read_pickle(cache_path)
        if isinstance(old, dict) and len(old.get('pe', {})) > 10:
            existing = old
            print(f"Loaded partial cache: PE={len(existing.get('pe',{}))}, PB={len(existing.get('pb',{}))}")

    to_fetch = [s for s in all_stocks if s not in existing.get('pe', {})]
    print(f"Need to fetch: {len(to_fetch)} stocks")

    if not to_fetch:
        print("All data cached!")
        return

    lg = bs.login()
    pe_data = dict(existing.get('pe', {}))
    pb_data = dict(existing.get('pb', {}))
    failed = []
    t0 = time.time()

    for i, stock in enumerate(to_fetch):
        if i % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Fetching: {i}/{len(to_fetch)} ({elapsed:.0f}s)...")

        # Use DAILY frequency for PE/PB
        rs = bs.query_history_k_data_plus(
            stock,
            'date,peTTM,pbMRQ',
            start_date='2020-09-01',
            end_date='2026-03-31',
            frequency='d',
            adjustflag='1'
        )

        dates_list, pe_list, pb_list = [], [], []
        while rs.next():
            row = rs.get_row_data()
            try:
                d = row[0]
                pe = float(row[1]) if row[1] and row[1] != '' else np.nan
                pb = float(row[2]) if row[2] and row[2] != '' else np.nan
                dates_list.append(d)
                pe_list.append(pe)
                pb_list.append(pb)
            except (ValueError, IndexError):
                continue

        if len(dates_list) > 20:
            idx = pd.to_datetime(dates_list)
            daily_pe = pd.Series(pe_list, index=idx)
            daily_pb = pd.Series(pb_list, index=idx)
            # Resample to weekly (Friday = last trading day of week)
            weekly_pe = daily_pe.resample('W-FRI').last()
            weekly_pb = daily_pb.resample('W-FRI').last()
            pe_data[stock] = weekly_pe.dropna()
            pb_data[stock] = weekly_pb.dropna()
        else:
            failed.append(stock)

        # Checkpoint every 200
        if (i + 1) % 200 == 0:
            pd.to_pickle({'pe': pe_data, 'pb': pb_data}, cache_path)
            print(f"  Checkpoint saved: PE={len(pe_data)}, PB={len(pb_data)}")

        if i % 100 == 99:
            time.sleep(1)

    bs.logout()

    pd.to_pickle({'pe': pe_data, 'pb': pb_data}, cache_path)
    print(f"\nDone! PE: {len(pe_data)}, PB: {len(pb_data)}, Failed: {len(failed)}")
    if failed[:10]:
        print(f"  Sample failures: {failed[:10]}")
    print(f"Elapsed: {time.time()-t0:.1f}s")

if __name__ == '__main__':
    main()
