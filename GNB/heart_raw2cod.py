#!/usr/bin/env python3
"""
raw2cod.py — Python port of the Proben1 heart/raw2cod Perl script.

Usage (mirrors original):
  python raw2cod.py [-classifier] INPUT_FILE > OUTPUT_FILE

Behavior:
  - Reads heart.raw-like CSV rows (13 attributes + 1 label at the end).
  - Encodes inputs to 35 features exactly like the original Perl script:
      * scaling and smoothing constants
      * one-hot encodings
      * explicit missing-value indicators
  - Outputs either:
      * 2-column one-hot label ("-classifier" mode), or
      * single analog label (distorted scale) if not classifier mode

Reference: proben1/heart/raw2cod (Perl) and README encoding section.
"""
from __future__ import annotations

import argparse
import io
import sys
from typing import List


def _fmt(v: float) -> str:
    """Format like Perl's %g (compact, no trailing zeros)."""
    return f"{v:g}"


def encode_row(fields: List[str], classifier: bool) -> str:
    """Encode a single CSV row (list of string tokens) to the encoded line.

    fields: [F0..F13], where F13 is the class label (0..4)
    Returns a space-separated string of 35 inputs + outputs per mode.
    """
    if len(fields) < 14:
        # Allow lines that might have whitespace; skip silently
        raise ValueError(f"Expected 14 comma-separated values, got {len(fields)}: {fields}")

    F = [f.strip() for f in fields]

    def _num(s: str) -> float | None:
        if s == "?" or s == "":
            return None
        try:
            return float(s)
        except Exception:
            return None

    out: List[str] = []

    # F[0] age: (age - 28) / (77-27)
    nv0 = _num(F[0])
    v_age = nv0 if nv0 is not None else 28.0
    out.append(_fmt((v_age - 28.0) / (77.0 - 27.0)))  # denom 50 per original

    # F[1] sex: 1 if 1 else 0
    nv1 = _num(F[1])
    out.append("1" if (nv1 == 1) else "0")

    # F[2] cp (1..4, '?' or 0 -> missing slot)
    v = F[2]; nv = _num(v)
    if nv == 1:
        out += ["1", "0", "0", "0", "0"]
    elif nv == 2:
        out += ["0", "1", "0", "0", "0"]
    elif nv == 3:
        out += ["0", "0", "1", "0", "0"]
    elif nv == 4:
        out += ["0", "0", "0", "1", "0"]
    else:  # missing or 0
        out += ["0", "0", "0", "0", "1"]

    # F[3] trestbps: present -> scaled,0 ; missing ('?' or 0) -> 0,1
    v = F[3]; nv = _num(v)
    if v == "?" or nv == 0:
        out += ["0", "1"]
    else:
        fv = ((nv if nv is not None else 80.0) - 80.0) / (200.0 - 80.0)
        out += [_fmt(fv), "0"]

    # F[4] chol: present -> scaled,0 ; missing ('?' or 0) -> 0,1
    v = F[4]; nv = _num(v)
    if v == "?" or nv == 0:
        out += ["0", "1"]
    else:
        fv = ((nv if nv is not None else 85.0) - 85.0) / (605.0 - 85.0)  # 605 smoothing
        out += [_fmt(fv), "0"]

    # F[5] fbs: '?' -> 0 0 1 ; 1 -> 0 1 0 ; else -> 0 0 0
    v = F[5]; nv = _num(v)
    if v == "?":
        out += ["0", "0", "1"]
    elif nv == 1:
        out += ["0", "1", "0"]
    else:
        out += ["0", "0", "0"]

    # F[6] restecg: 0 -> 1 0 0 0 ; 1 -> 0 1 0 0 ; 2 -> 0 0 1 0 ; '?' -> 0 0 0 1
    v = F[6]; nv = _num(v)
    if v != "?" and nv == 0:
        out += ["1", "0", "0", "0"]
    elif nv == 1:
        out += ["0", "1", "0", "0"]
    elif nv == 2:
        out += ["0", "0", "1", "0"]
    else:
        out += ["0", "0", "0", "1"]

    # F[7] thalach: present -> scaled,0 ; missing ('?' or 0) -> 0,1
    v = F[7]; nv = _num(v)
    if v == "?" or nv == 0:
        out += ["0", "1"]
    else:
        fv = ((nv if nv is not None else 60.0) - 60.0) / (210.0 - 60.0)  # 210 smoothing
        out += [_fmt(fv), "0"]

    # F[8] exang: '?' -> 0 0 1 ; 1 -> 0 1 0 ; else -> 0 0 0
    v = F[8]; nv = _num(v)
    if v == "?":
        out += ["0", "0", "1"]
    elif nv == 1:
        out += ["0", "1", "0"]
    else:
        out += ["0", "0", "0"]

    # F[9] oldpeak: '?' -> 0,1 ; else scaled,0  (note: 0 is valid, not missing)
    v = F[9]; nv = _num(v)
    if v == "?":
        out += ["0", "1"]
    else:
        fv = ((nv if nv is not None else -2.6) - (-2.6)) / (6.2 - (-2.6))
        out += [_fmt(fv), "0"]

    # F[10] slope: 1 -> 1 0 0 0 ; 2 -> 0 1 0 0 ; 3 -> 0 0 1 0 ; missing/0 -> 0 0 0 1
    v = F[10]; nv = _num(v)
    if nv == 1:
        out += ["1", "0", "0", "0"]
    elif nv == 2:
        out += ["0", "1", "0", "0"]
    elif nv == 3:
        out += ["0", "0", "1", "0"]
    else:
        out += ["0", "0", "0", "1"]

    # F[11] ca: '?' -> 0,1 ; else scaled,0
    v = F[11]; nv = _num(v)
    if v == "?":
        out += ["0", "1"]
    else:
        fv = ((nv if nv is not None else 0.0) - 0.0) / (3.0 - 0.0)
        out += [_fmt(fv), "0"]

    # F[12] thal: 3 -> 1 0 0 0 ; 6 -> 0 1 0 0 ; 7 -> 0 0 1 0 ; '?' -> 0 0 0 1
    v = F[12]; nv = _num(v)
    if nv == 3:
        out += ["1", "0", "0", "0"]
    elif nv == 6:
        out += ["0", "1", "0", "0"]
    elif nv == 7:
        out += ["0", "0", "1", "0"]
    else:
        out += ["0", "0", "0", "1"]

    # Label at F[13]
    lab = F[13]
    try:
        lab_i = int(float(lab))
    except Exception:
        lab_i = 0

    if classifier:
        # 0 -> 1 0 ; else -> 0 1
        out += ["1", "0"] if lab_i == 0 else ["0", "1"]
    else:
        # Analog: v = lab/10 + 0.1 ; then add 0.4 if v > 0.15
        v = lab_i / 10.0 + 0.1
        if v > 0.15:
            v += 0.4
        out.append(_fmt(v))

    return " ".join(out)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Encode heart.raw to .cod (Python port of raw2cod)")
    # Support both --classifier and -classifier like the Perl usage says
    parser.add_argument("-classifier", "--classifier", action="store_true", help="Use 1-of-2 classifier output instead of analog")
    parser.add_argument("input", help="Path to heart*.raw (use - for stdin)")
    args = parser.parse_args(argv)

    # Input stream
    if args.input == "-":
        fin = sys.stdin
        close_fin = False
    else:
        fin = open(args.input, "r", encoding="utf-8")
        close_fin = True

    try:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            # Split by comma exactly like Perl split(',')
            fields = line.split(',')
            try:
                encoded = encode_row(fields, args.classifier)
            except Exception as e:
                # Keep behavior permissive: skip clearly malformed lines but warn on stderr
                print(f"[WARN] Skipping malformed line: {line[:120]}... ({e})", file=sys.stderr)
                continue
            print(encoded)
    finally:
        if close_fin:
            fin.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
