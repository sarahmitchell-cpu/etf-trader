#!/usr/bin/env python3
"""Download weekly prices and daily valuations for historical index stocks missing from DB."""

import baostock as bs
import sqlite3
import json
import os
import time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(DATA_DIR, 'stock_data.db')

def get_missing_codes():
    """Find codes in historical constituents but not in DB."""
    all_codes = set()
    for fn in ['csi300_constituents_history.json', 'csi500_constituents_history.json']:
        path = os.path.join(DATA_DIR, 'baostock_cache', fn)
        with open(path) as f:
            d = json.load(f)
        for dt, stocks in d['constituents'].items():
            for s in stocks:
                all_codes.add(s.split('.')[1])

    conn = sqlite3.connect(DB_PATH)
    db_codes = set(r[0] for r in conn.execute('SELECT DISTINCT code FROM stock_weekly').fetchall())
    conn.close()

    missing = sorted(all_codes - db_codes)
    return missing

def code_to_bs(code):
    """Convert 6-digit code to baostock format."""
    if code.startswith('6'):
        return f'sh.{code}'
    else:
        return f'sz.{code}'

def main():
    missing = get_missing_codes()
    print(f"Missing stocks: {len(missing)}")
    if not missing:
        print("Nothing to download!")
        return

    lg = bs.login()
    if lg.error_code != '0':
        print(f"Login failed: {lg.error_msg}")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")

    weekly_added = 0
    val_added = 0
    failed = []

    for i, code in enumerate(missing):
        bs_code = code_to_bs(code)
        if i % 50 == 0:
            print(f"Progress: {i}/{len(missing)} (weekly={weekly_added}, val={val_added}, failed={len(failed)})")

        # 1. Weekly prices
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, 'date,close,volume',
                start_date='2020-01-01', end_date='2026-03-31',
                frequency='w', adjustflag='1'
            )
            rows = []
            while rs.next():
                row = rs.get_row_data()
                try:
                    close = float(row[1])
                    vol = float(row[2]) if row[2] else 0
                    if close > 0:
                        rows.append((code, row[0], close, vol))
                except (ValueError, IndexError):
                    pass

            if rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO stock_weekly (code, date, close, volume) VALUES (?, ?, ?, ?)",
                    rows
                )
                weekly_added += len(rows)
        except Exception as e:
            failed.append((code, 'weekly', str(e)))

        # 2. Daily valuations
        try:
            rs = bs.query_history_k_data_plus(
                bs_code, 'date,close,peTTM,pbMRQ,psTTM',
                start_date='2020-01-01', end_date='2026-03-31',
                frequency='d', adjustflag='1'
            )
            rows = []
            while rs.next():
                row = rs.get_row_data()
                try:
                    close = float(row[1])
                    pe = float(row[2]) if row[2] else 0
                    pb = float(row[3]) if row[3] else 0
                    ps = float(row[4]) if row[4] else 0
                    if close > 0:
                        rows.append((code, row[0], close, pe, pb, ps))
                except (ValueError, IndexError):
                    pass

            if rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO stock_daily_valuation (code, date, close, pe_ttm, pb_mrq, ps_ttm) VALUES (?, ?, ?, ?, ?, ?)",
                    rows
                )
                val_added += len(rows)
        except Exception as e:
            failed.append((code, 'valuation', str(e)))

        # Commit every 50 stocks
        if (i + 1) % 50 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    bs.logout()

    print(f"\nDone! Weekly rows added: {weekly_added}, Valuation rows added: {val_added}")
    print(f"Failed: {len(failed)}")
    if failed:
        for code, dtype, err in failed[:10]:
            print(f"  {code} ({dtype}): {err}")

if __name__ == '__main__':
    main()
