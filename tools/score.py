"""Reimplementation of make.fun's ranked forecast scorer (grader/forecast/ranked.ts).

Pure functions over a parsed forecast and a list of charts (each a list of universe ids, place 1 first).
Used by the backtest as the judge; `python tools/score.py <forecast.json> <reads.ndjson> <from> <until>`
scores one forecast file against the reads of a window.
"""

import json
import math
import sys
from datetime import datetime, timezone

P_FLOOR = 1e-4
SUM_TOLERANCE = 1e-9
MAX_ROWS_PER_PLACE = 100
MAX_BYTES = 1_048_576


class ForecastError(ValueError):
    pass


class Forecast:
    __slots__ = ("places", "rows", "top", "named", "sums")

    def __init__(self, places, rows, top, named, sums):
        self.places = places
        self.rows = rows
        self.top = top
        self.named = named  # list of dict id -> p, index k-1
        self.sums = sums


def parse_forecast(data: bytes, places: int = 10) -> Forecast:
    """Parse forecast bytes the way parseRanked does, raising ForecastError on anything out of shape."""
    if len(data) > MAX_BYTES:
        raise ForecastError(f"the forecast is over {MAX_BYTES} bytes")
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ForecastError(f"the forecast is not UTF-8 JSON: {err}")
    if not isinstance(doc, dict):
        raise ForecastError("the forecast is not one JSON object")
    if sorted(doc.keys()) != ["probabilities", "top10"]:
        raise ForecastError(f"the forecast's keys are {sorted(doc.keys())}")
    lst = doc["probabilities"]
    if not isinstance(lst, list):
        raise ForecastError("probabilities is not a list")
    named = [dict() for _ in range(places)]
    sums = [0.0] * places
    rows = []
    for i, row in enumerate(lst):
        if not isinstance(row, dict) or set(row.keys()) != {"universe_id", "place", "p"}:
            raise ForecastError(f"probabilities[{i}] is not exactly {{universe_id, place, p}}")
        uid = row["universe_id"]
        if not (isinstance(uid, int) and not isinstance(uid, bool) and uid >= 1):
            raise ForecastError(f"probabilities[{i}].universe_id is not a positive integer")
        place = row["place"]
        if not (isinstance(place, int) and not isinstance(place, bool) and 1 <= place <= places):
            raise ForecastError(f"probabilities[{i}].place is not an integer from 1 to {places}")
        p = row["p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p) or not (0 <= p <= 1):
            raise ForecastError(f"probabilities[{i}].p is not a number from 0 to 1")
        at = named[place - 1]
        if uid in at:
            raise ForecastError(f"universe_id {uid} is named twice at place {place}")
        if len(at) == MAX_ROWS_PER_PLACE:
            raise ForecastError(f"place {place} has more than {MAX_ROWS_PER_PLACE} rows")
        at[uid] = p
        sums[place - 1] += p
        rows.append((uid, place, p))
    for k in range(places):
        if sums[k] > 1 + SUM_TOLERANCE:
            raise ForecastError(f"place {k + 1}'s probabilities add up to {sums[k]}, more than 1")
    top = doc["top10"]
    if (not isinstance(top, list) or len(top) != places
            or not all(isinstance(v, int) and not isinstance(v, bool) and v >= 1 for v in top)
            or len(set(top)) != places):
        raise ForecastError(f"top10 is not {places} distinct positive integers")
    return Forecast(places, rows, top, named, sums)


def place_probability(f: Forecast, place: int, uid, chart) -> float:
    at = f.named[place - 1]
    own = at.get(uid)
    if own is not None:
        return own
    unnamed = sum(1 for c in set(chart) if c not in at)
    left = max(0.0, 1.0 - f.sums[place - 1])
    return 0.0 if unnamed == 0 else left / unnamed


def order_log_score(f: Forecast, chart, floor: float = P_FLOOR) -> float:
    if len(set(chart)) != len(chart) or not chart:
        raise ValueError("the chart lists no id or an id twice")
    n = min(f.places, len(chart))
    total = 0.0
    for k in range(1, n + 1):
        total += math.log(max(place_probability(f, k, chart[k - 1], chart), floor))
    return total


def mean_log_score(f: Forecast, charts, floor: float = P_FLOOR) -> float:
    if not charts:
        raise ValueError("no read to score")
    return sum(order_log_score(f, c, floor) for c in charts) / len(charts)


def hits_of(top, actual) -> int:
    head = actual[: len(top)]
    return sum(1 for uid in top if uid in head)


def agreement_of(top, actual) -> int:
    width = len(top)
    head = actual[:width]
    points = 0
    for i, uid in enumerate(top):
        if uid in head:
            points += width - abs(i - head.index(uid))
    return points


def round6(x: float) -> float:
    r = round(x * 1e6) / 1e6
    return 0.0 if r == 0 else r


# ---- reads ------------------------------------------------------------------------------------------

def parse_time(s: str) -> datetime:
    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def chart_of(read) -> list:
    """A read's chart as ids, place 1 first, sponsored rows and duplicate ids dropped."""
    rows = read.get("top_trending") or []
    rows = [r for r in rows if not r.get("is_sponsored")]
    rows = sorted(rows, key=lambda r: r["rank"])
    seen = set()
    out = []
    for r in rows:
        uid = r["universe_id"]
        if uid in seen:
            continue
        seen.add(uid)
        out.append(uid)
    return out


def load_reads(path: str) -> list:
    reads = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("ok") is False or not r.get("top_trending"):
                continue
            reads.append(r)
    return reads


def window_charts(reads, start: datetime, until: datetime) -> list:
    """The charts of the reads that start in [start, until), in time order."""
    out = []
    for r in sorted(reads, key=lambda r: parse_time(r["at"])):
        t = parse_time(r["at"])
        if start <= t < until:
            out.append(chart_of(r))
    return out


def score_file(forecast_path: str, charts) -> dict:
    with open(forecast_path, "rb") as fh:
        f = parse_forecast(fh.read())
    return {
        "log_score": round6(mean_log_score(f, charts)),
        "hits": round6(sum(hits_of(f.top, c) for c in charts) / len(charts)),
        "agreement": round6(sum(agreement_of(f.top, c) for c in charts) / len(charts)),
        "reads": len(charts),
    }


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(__doc__)
        sys.exit(2)
    reads = load_reads(sys.argv[2])
    charts = window_charts(reads, parse_time(sys.argv[3]), parse_time(sys.argv[4]))
    print(json.dumps(score_file(sys.argv[1], charts)))
