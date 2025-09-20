#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process vignette-form Google Form export:
- First 2 columns = metadata (kept but not used for block selection).
- Last 7 columns = respondent info (dropped entirely).
- Between them: repeated groups per vignette:
    [ 'Select any option to your liking',  7×A(.0), 7×B(.1), 7×C(.2) ]
- For each row × vignette, exactly one block (A/B/C) is filled (others NaN).
  Keep that block and map A→'comply', B→'ref_emp', C→'ref_con'.
- Output: long dataframe with one row per (row,vignette) that had a filled block.

Usage:
    python process_vignettes.py --csv input.csv --out long.csv
"""
import argparse
import sys
import pandas as pd
import numpy as np

BASE_SELECT = "Select any option to your liking"
LIKERT_7 = [
    "Appropriateness",
    "Perceived safety",
    "Trust in the robot",
    "Perceived empathy",
    "Who is to blame if something goes wrong?",
    "I perceived the task as risky",
    "I understood the scenario",
]
RESP_MAP = {0: "comply", 1: "ref_emp", 2: "ref_con"}  # A,B,C

def strip_suffix(colname: str) -> str:
    """Return base name w/o trailing '.k'."""
    parts = str(colname).rsplit('.', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return colname

def find_select_positions(cols: pd.Index) -> list:
    """Indices of every 'Select any option to your liking' column."""
    return [i for i, c in enumerate(cols) if strip_suffix(c) == BASE_SELECT]

def process(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Keep metadata (first 2 cols) to carry through; drop last 7 from the frame.
    meta_cols = list(df.columns[:2])
    if df.shape[1] >= 9:
        df_mid = df.iloc[:, 2:-7]
        tail_dropped = 7
    else:
        # Fallback if not enough columns: assume only first 2 meta to drop from processing
        df_mid = df.iloc[:, 2:]
        tail_dropped = 0

    # Sanity: discover vignette anchors
    select_pos = find_select_positions(df_mid.columns)
    if not select_pos:
        raise RuntimeError(f"Could not find any '{BASE_SELECT}' columns.")
    # We expect for each select: next 21 columns = 3× blocks of 7.
    # We'll be defensive if fewer remain at the end.

    long_rows = []
    kept, dropped = 0, 0

    for ridx, row in df_mid.iterrows():
        # Carry metadata (optional)
        meta_vals = {c: df.loc[ridx, c] for c in meta_cols} if meta_cols else {}

        for v, s_idx in enumerate(select_pos):
            # Identify 21 columns after this select (if present)
            start = s_idx + 1
            end = min(start + 21, df_mid.shape[1])
            block_cols = list(df_mid.columns[start:end])

            # If fewer than 7 cols, nothing to do for this vignette
            if len(block_cols) < 7:
                continue

            # Slice into A/B/C (each 7 wide) where possible
            A = block_cols[0:7]
            B = block_cols[7:14] if len(block_cols) >= 14 else []
            C = block_cols[14:21] if len(block_cols) >= 21 else []

            blocks = [A, B, C]
            nonnan_counts = []
            for blk in blocks:
                if len(blk) != 7:
                    nonnan_counts.append(0)
                else:
                    nonnan_counts.append(int(pd.notna(row[blk]).sum()))

            # Pick the block with the most filled answers; ties → first
            best = int(np.argmax(nonnan_counts))
            filled = nonnan_counts[best] if blocks[best] else 0

            if filled == 0:
                dropped += 1
                continue

            # Extract values, rename to base names (no .k suffix), and record resp_type
            vals = row[blocks[best]].to_dict()
            clean = {strip_suffix(k): v for k, v in vals.items()}

            # Ensure only the 7 expected fields are kept (order-insensitive)
            for name in LIKERT_7:
                clean.setdefault(name, np.nan)
            clean = {k: clean[k] for k in LIKERT_7}

            # Also keep the raw selected option text
            sel_col = df_mid.columns[s_idx]
            response_raw = row[sel_col]

            out_rec = {
                "pid": int(ridx + 1),
                "vignette_idx": int(v + 1),
                "resp_type": RESP_MAP[best],
                "response_raw": response_raw,
            }
            out_rec.update(meta_vals)
            out_rec.update(clean)

            long_rows.append(out_rec)
            kept += 1

    long_df = pd.DataFrame(long_rows)

    # Optional: report keep/drop ratio
    total_slots = len(select_pos) * len(df_mid)  # one potential keep per vignette per row
    # (We expect ~33% kept, ~66% dropped)
    kept_ratio = kept / max(total_slots, 1)
    dropped_ratio = dropped / max(total_slots, 1)
    print(f"[info] kept {kept} ({kept_ratio:.2%}), dropped {dropped} ({dropped_ratio:.2%}) "
          f"out of {total_slots} vignette slots.")
    print(f"[info] meta kept: {meta_cols}, tail dropped: {tail_dropped} columns.")

    return long_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to the raw CSV export.")
    ap.add_argument("--out", required=True, help="Path to write the long CSV.")
    args = ap.parse_args()

    out_df = process(args.csv)
    out_df.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out} with {len(out_df)} rows.")

if __name__ == "__main__":
    sys.exit(main())
