"""Forecast Roblox's Top Trending chart, place by place, for the 24 hourly slots after the history ends.

Reads $MAKEFUN_INPUT_DIR/history.ndjson and task.json (default /input), writes
$MAKEFUN_OUTPUT_DIR/forecast.json (default /output). Python 3.13, standard library only, deterministic.

The model: each scored place is its own log score, so forecast each place's marginal P(game g at place k)
and average it over the window's slots. Every pair of past reads (i before j, lag up to 30 h) says where
the game at place k of read j had been on read i: an offset, or NEW when it was not on that chart. For a
slot at horizon h, the pairs are weighted by a triangular kernel around h and the offset distributions of
the place, its band, all places pooled, and a smooth prior are mixed. Offsets are then mapped through the
last read's chart to games, tilted by each game's age and player-count trend, and averaged over slots.
"""

import json
import math
import os
from datetime import datetime, timedelta, timezone

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MAX_LAG_H = 30.0          # pairs of reads further apart than this teach nothing about a 24 h window
MIN_GAMES = 10            # a read with fewer games is not a chart
DEDUP_SECONDS = 60        # a read within this of the previous one is the same read
KERNEL_WIDTH_H = 1.5      # triangular kernel half-width on |lag - horizon|
KERNEL_MIN_MASS = 15.0    # widen the kernel until the weights reach this
KERNEL_MAX_WIDTH_H = 60.0
BANDS = [(1,), (2, 3), (4, 5, 6), (7, 8, 9, 10)]
W_BAND = 120.0
W_POOL = 3.0
W_PRIOR = 4.0
PRIOR_SCALE = 2.5
PRIOR_NEW = 0.04
OLD_DAYS = 6.0
TILT_OLD = -0.5
TILT_TREND = 0.5
TREND_SPAN = 5.0
NAME_MIN_P = 0.0025
MAX_NAMED = 0.999
RECENCY_HALFLIFE_D = 0.0  # > 0: halve a pair's weight for every this many days its later read precedes the last read
SHARPEN = 1.0             # exponent on the named probabilities of a place before the tilt (1 = none)
NEW = "new"


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


class Read:
    __slots__ = ("at", "chart", "rank", "playing")

    def __init__(self, at, chart, playing):
        self.at = at
        self.chart = chart                                  # ids, place 1 first
        self.rank = {uid: i + 1 for i, uid in enumerate(chart)}
        self.playing = playing                              # id -> player count (may be missing)


def load_history(path):
    """The usable reads of history.ndjson, oldest first."""
    reads = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("ok") is False:
                continue
            rows = raw.get("top_trending") or []
            rows = [r for r in rows if not r.get("is_sponsored")]
            rows.sort(key=lambda r: r["rank"])
            chart, playing, seen = [], {}, set()
            for r in rows:
                uid = r["universe_id"]
                if uid in seen:
                    continue
                seen.add(uid)
                chart.append(uid)
                if isinstance(r.get("playing"), (int, float)):
                    playing[uid] = r["playing"]
            if len(chart) < MIN_GAMES:
                continue
            reads.append(Read(parse_time(raw["at"]), chart, playing))
    reads.sort(key=lambda r: r.at)
    kept = []
    for r in reads:
        if kept and (r.at - kept[-1].at).total_seconds() < DEDUP_SECONDS:
            continue
        kept.append(r)
    return kept


def observations(reads, places):
    """(lag hours, offsets, recency weight) for every pair i < j with lag <= MAX_LAG_H.

    offsets[k-1] is rank_i(game at place k of read j) - k, or NEW when read i did not list it.
    """
    obs = []
    last_at = reads[-1].at
    for j in range(1, len(reads)):
        rj = reads[j]
        age_days = (last_at - rj.at).total_seconds() / 86400.0
        recency = 0.5 ** (age_days / RECENCY_HALFLIFE_D) if RECENCY_HALFLIFE_D > 0 else 1.0
        for i in range(j - 1, -1, -1):
            ri = reads[i]
            lag = (rj.at - ri.at).total_seconds() / 3600.0
            if lag > MAX_LAG_H:
                break
            offs = []
            for k in range(1, min(places, len(rj.chart)) + 1):
                uid = rj.chart[k - 1]
                r = ri.rank.get(uid)
                offs.append(NEW if r is None else r - k)
            obs.append((lag, offs, recency))
    return obs


def kernel_weights(obs, horizon):
    width = KERNEL_WIDTH_H
    while True:
        w = [rec * max(0.0, 1.0 - abs(lag - horizon) / width) for lag, _, rec in obs]
        if sum(w) >= KERNEL_MIN_MASS or width >= KERNEL_MAX_WIDTH_H:
            return w
        width *= 1.5


def rank_distribution(counts, k, length):
    """Offset counts -> normalised mass over ranks 1..length of the last chart, plus NEW."""
    dist = {}
    total = 0.0
    for off, c in counts.items():
        if c <= 0:
            continue
        key = NEW if off == NEW else k + off
        if key != NEW and not (1 <= key <= length):
            key = NEW
        dist[key] = dist.get(key, 0.0) + c
        total += c
    if total <= 0:
        return {}
    return {key: v / total for key, v in dist.items()}


def prior_distribution(k, length):
    raw = {r: math.exp(-abs(r - k) / PRIOR_SCALE) for r in range(1, length + 1)}
    s = sum(raw.values())
    dist = {r: (1.0 - PRIOR_NEW) * v / s for r, v in raw.items()}
    dist[NEW] = PRIOR_NEW
    return dist


def mix(components):
    """components: list of (weight, distribution). Returns the weighted mixture."""
    total = sum(w for w, d in components if d)
    out = {}
    for w, d in components:
        if not d or w <= 0:
            continue
        for key, v in d.items():
            out[key] = out.get(key, 0.0) + w * v / total
    return out


def place_distributions(obs, weights, places, length):
    """Per place k: the mixed distribution over ranks of the last chart (and NEW) at one horizon."""
    own = [dict() for _ in range(places)]
    for (lag, offs, _rec), w in zip(obs, weights):
        if w <= 0:
            continue
        for idx, off in enumerate(offs):
            own[idx][off] = own[idx].get(off, 0.0) + w
    pooled = {}
    for d in own:
        for off, c in d.items():
            pooled[off] = pooled.get(off, 0.0) + c
    band_counts = {}
    for band in BANDS:
        c = {}
        for k in band:
            for off, v in own[k - 1].items():
                c[off] = c.get(off, 0.0) + v
        for k in band:
            band_counts[k] = c
    result = []
    for k in range(1, places + 1):
        own_mass = sum(own[k - 1].values())
        result.append(mix([
            (own_mass, rank_distribution(own[k - 1], k, length)),
            (W_BAND, rank_distribution(band_counts[k], k, length)),
            (W_POOL, rank_distribution(pooled, k, length)),
            (W_PRIOR, prior_distribution(k, length)),
        ]))
    return result


def game_features(reads):
    """Per game of the last chart: old (first seen >= OLD_DAYS before the last read) and its play trend."""
    last = reads[-1]
    first_seen = {}
    for r in reads:
        for uid in r.chart:
            if uid not in first_seen:
                first_seen[uid] = r.at
    target = last.at - timedelta(hours=24)
    earlier = None
    for r in reads[:-1]:
        gap = abs((r.at - target).total_seconds()) / 3600.0
        if gap <= 12.0 and (earlier is None or gap < abs((earlier.at - target).total_seconds()) / 3600.0):
            earlier = r
    feats = {}
    for uid in last.chart:
        old = (last.at - first_seen[uid]).total_seconds() >= OLD_DAYS * 86400.0
        trend = 0.0
        if earlier is not None:
            now_p = last.playing.get(uid)
            then_p = earlier.playing.get(uid)
            if now_p and then_p and now_p > 0 and then_p > 0:
                trend = max(-1.0, min(1.0, math.log(now_p / then_p)))
        feats[uid] = (old, trend)
    return feats


def tilt(dist, k, last, feats, strength):
    """Reweight the named games at place k by age and trend, keeping the named total unchanged."""
    named_total = sum(v for key, v in dist.items() if key != NEW)
    if named_total <= 0:
        return dist
    out = {}
    for key, v in dist.items():
        if key == NEW:
            continue
        uid = last.chart[key - 1]
        old, trend = feats[uid]
        factor = math.exp(strength * (TILT_OLD * (1.0 if old else 0.0) + TILT_TREND * trend * (key - k) / TREND_SPAN))
        out[uid] = (v ** SHARPEN) * factor
    s = sum(out.values())
    return {uid: v * named_total / s for uid, v in out.items()}


def forecast(reads, window_from, slot_minutes, slots, places):
    """The forecast document for `reads` (oldest first) over the window."""
    last = reads[-1]
    length = len(last.chart)
    obs = observations(reads, places)
    feats = game_features(reads)
    acc = [dict() for _ in range(places)]
    for s in range(slots):
        slot_time = window_from + timedelta(minutes=slot_minutes * s)
        horizon = max(0.05, (slot_time - last.at).total_seconds() / 3600.0)
        weights = kernel_weights(obs, horizon)
        dists = place_distributions(obs, weights, places, length)
        strength = min(1.0, horizon / 24.0)
        for k in range(1, places + 1):
            tilted = tilt(dists[k - 1], k, last, feats, strength)
            for uid, v in tilted.items():
                acc[k - 1][uid] = acc[k - 1].get(uid, 0.0) + v / slots
    probabilities = []
    picked = set()
    top10 = []
    for k in range(1, places + 1):
        d = {uid: p for uid, p in acc[k - 1].items() if p >= NAME_MIN_P}
        total = sum(d.values())
        if total > MAX_NAMED:
            d = {uid: p * MAX_NAMED / total for uid, p in d.items()}
        rows = sorted(d.items(), key=lambda it: (-it[1], it[0]))
        for uid, p in rows:
            probabilities.append({"universe_id": uid, "place": k, "p": round(p, 6)})
        for uid, _ in sorted(acc[k - 1].items(), key=lambda it: (-it[1], it[0])):
            if uid not in picked:
                picked.add(uid)
                top10.append(uid)
                break
    for uid in last.chart:
        if len(top10) >= places:
            break
        if uid not in picked:
            picked.add(uid)
            top10.append(uid)
    return {"probabilities": probabilities, "top10": top10}


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), encoding="utf-8") as fh:
        task = json.load(fh)
    reads = load_history(os.path.join(INPUT_DIR, "history.ndjson"))
    window = task["window"]
    doc = forecast(reads, parse_time(window["from"]), int(window["slot_minutes"]), int(window["slots"]),
                   int(task.get("places", 10)))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, task.get("output", "forecast.json")), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(doc, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
