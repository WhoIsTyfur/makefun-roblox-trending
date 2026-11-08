"""Backtest main.py's model against the archive (and any converted live reads) with the exact scorer.

For every read with at least MIN_HISTORY reads before it: history = the reads up to and including it, the window
starts at the next whole hour after it and lasts 24 h, the answer is every read that starts in the window. Each
window's score is the mean read score; the report gives the mean over windows, the mean per horizon bucket, the
mean per place, and the practice window apart. The house baseline is scored on the same windows.

    python tools/backtest.py [--reads data/archive/reads.ndjson ...] [--min-answers 1] [--list]
"""

import argparse
import json
import math
import os
import sys
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import main as model  # noqa: E402
from tools import score  # noqa: E402

MIN_HISTORY = 7
PRACTICE_END = "2026-11-05T21:59:22Z"
PLACES = 10


def baseline_doc(reads, places=PLACES):
    chart = reads[-1].chart
    rows = []
    for k in range(1, min(places, len(chart)) + 1):
        rows.append({"universe_id": chart[k - 1], "place": k, "p": 0.5})
        if k == 1:
            if len(chart) > 1:
                rows.append({"universe_id": chart[1], "place": 1, "p": 0.3})
            continue
        rows.append({"universe_id": chart[k - 2], "place": k, "p": 0.15})
        if len(chart) > k:
            rows.append({"universe_id": chart[k], "place": k, "p": 0.15})
    return {"probabilities": rows, "top10": chart[:places]}


def parse_doc(doc):
    return score.parse_forecast(json.dumps(doc).encode("utf-8"))


def per_place_scores(f, chart):
    out = []
    for k in range(1, min(f.places, len(chart)) + 1):
        out.append(math.log(max(score.place_probability(f, k, chart[k - 1], chart), score.P_FLOOR)))
    return out


def next_whole_hour(t):
    base = t.replace(minute=0, second=0, microsecond=0)
    return base + timedelta(hours=1)


def load_all(paths):
    reads = []
    for p in paths:
        reads.extend(model.load_history(p))
    reads.sort(key=lambda r: r.at)
    kept = []
    for r in reads:
        if kept and (r.at - kept[-1].at).total_seconds() < model.DEDUP_SECONDS:
            continue
        kept.append(r)
    return kept


def run(reads, forecaster, min_answers=1, verbose=False, only_practice=False):
    windows = []
    for idx in range(MIN_HISTORY, len(reads)):
        last = reads[idx]
        history = reads[: idx + 1]
        start = next_whole_hour(last.at)
        until = start + timedelta(hours=24)
        answers = [r for r in reads[idx + 1:] if start <= r.at < until]
        if len(answers) < min_answers:
            continue
        is_practice = last.at.strftime("%Y-%m-%dT%H:%M:%SZ") == PRACTICE_END
        if only_practice and not is_practice:
            continue
        doc = forecaster(history, start, 60, 24, PLACES)
        f = parse_doc(doc)
        read_scores = []
        place_rows = []
        prob_rows = []
        horizons = []
        for a in answers:
            pp = per_place_scores(f, a.chart)
            read_scores.append(sum(pp))
            place_rows.append(pp)
            prob_rows.append([(score.place_probability(f, k, a.chart[k - 1], a.chart), a.chart[k - 1] in f.named[k - 1])
                              for k in range(1, min(PLACES, len(a.chart)) + 1)])
            horizons.append((a.at - last.at).total_seconds() / 3600.0)
        windows.append({
            "last": last.at,
            "practice": is_practice,
            "score": sum(read_scores) / len(read_scores),
            "reads": read_scores,
            "places": place_rows,
            "probs": prob_rows,
            "named_total": list(f.sums),
            "horizons": horizons,
            "hits": sum(score.hits_of(f.top, a.chart) for a in answers) / len(answers),
        })
        if verbose:
            print(f"  {last.at:%m-%d %H:%M} n={len(answers):2d} score={windows[-1]['score']:8.3f}")
    return windows


def summarise(name, windows):
    if not windows:
        print(f"{name}: no windows")
        return
    mean = sum(w["score"] for w in windows) / len(windows)
    print(f"{name}: {len(windows)} windows, mean {mean:.3f}, hits {sum(w['hits'] for w in windows) / len(windows):.2f}")
    unnamed = [0] * PLACES
    named_total = [0.0] * PLACES
    n_rows = 0
    for w in windows:
        for row in w["probs"]:
            n_rows += 1
            for k, (p, named) in enumerate(row):
                if not named:
                    unnamed[k] += 1
        for k, tot in enumerate(w["named_total"]):
            named_total[k] += tot
    print("  actual unnamed (per read, by place): " + " ".join(f"{u / n_rows:6.2f}" for u in unnamed))
    print("  named total by place:                " + " ".join(f"{t / len(windows):6.2f}" for t in named_total))
    buckets = [(0, 6), (6, 12), (12, 18), (18, 24.01)]
    parts = []
    for lo, hi in buckets:
        vals = [s for w in windows for s, h in zip(w["reads"], w["horizons"]) if lo <= h < hi]
        parts.append(f"{lo:.0f}-{hi:.0f}h {sum(vals) / len(vals):7.3f} (n={len(vals)})" if vals else f"{lo:.0f}-{hi:.0f}h   n/a")
    print("  by horizon: " + "; ".join(parts))
    per_place = [0.0] * PLACES
    n = 0
    for w in windows:
        for row in w["places"]:
            n += 1
            for k, v in enumerate(row):
                per_place[k] += v
    print("  by place:   " + " ".join(f"{v / n:6.2f}" for v in per_place))
    pr = [w for w in windows if w["practice"]]
    if pr:
        print(f"  practice window: {pr[0]['score']:.6f} over {len(pr[0]['reads'])} reads")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", nargs="+", default=["data/archive/reads.ndjson"])
    ap.add_argument("--min-answers", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override main.py constants, NAME=VALUE")
    args = ap.parse_args()
    for item in args.set:
        name, value = item.split("=", 1)
        setattr(model, name, type(getattr(model, name))(value) if not isinstance(getattr(model, name), (list, tuple))
                else json.loads(value))
        print(f"set {name} = {getattr(model, name)}")
    reads = load_all(args.reads)
    print(f"{len(reads)} reads from {reads[0].at:%Y-%m-%d %H:%M} to {reads[-1].at:%Y-%m-%d %H:%M}")
    model_windows = run(reads, model.forecast, args.min_answers, args.list)
    summarise("model", model_windows)
    if not args.no_baseline:
        base_windows = run(reads, lambda h, *_: baseline_doc(h), args.min_answers)
        summarise("baseline", base_windows)
        wins = sum(1 for m, b in zip(model_windows, base_windows) if m["score"] > b["score"])
        print(f"model beats baseline in {wins} of {len(model_windows)} windows")


if __name__ == "__main__":
    main()
