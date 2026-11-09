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
TILT_OLD = 0.0            # the flat old-age tilt is replaced by the age hazard below (backtest: +0.05)
TILT_TREND = 0.5
TREND_SPAN = 5.0
NAME_MIN_P = 0.0025
MAX_NAMED = 0.999
RECENCY_HALFLIFE_D = 0.0  # > 0: halve a pair's weight for every this many days its later read precedes the last read
SHARPEN = 1.0             # exponent on the named probabilities of a place before the tilt (1 = none)
# Occupancy: with hourly reads, the share of the last day each game held a place (the chart has a daily cycle, and
# one forecast is scored against every hour of a day). Blended into the transition model in proportion to how much
# of the last day the hourly reads cover, and more at longer horizons.
OCC_GAP_H = 1.5           # a read with another read within this many hours counts as hourly
OCC_W2 = 0.4              # weight of hourly reads 24 to 48 h before the last read, against 1 for the last 24 h
OCC_MAX = 0.3             # blend weight at full coverage and horizon
OCC_RAMP_H = 12.0         # the blend reaches full strength at this horizon
OCC_PLACES = 2            # blend only places 1..OCC_PLACES: lower places drift down over a day, so their past share
#                           predicts less (a proxy test on the archive's reads gained at places 1-2, lost from 3 on)
# Age hazard: a game's chance of leaving the chart within the horizon, rising with its time on the chart. Its mass at
# a place passes to the games below it there (they move up when it leaves), or to the unnamed share.
HAZ_START_D = 7.0         # no hazard before this many days on the chart (known-age games under 7 days never left)
HAZ_RATE = 0.1            # leaving rate per day, per day on the chart past HAZ_START_D (0 = off)
HAZ_MAX_RATE = 0.15       # cap on the leaving rate per day: the rate seen for games already on the chart when the
#                           archive starts (65 of 476 game-days); ages beyond the history's span are not observed
# Deep origins: a game that takes a place from DEEP_OFFSET or more places below it. The offset model puts that mass
# on whichever game sits at the exact rank a past riser came from; instead the place's deep mass is shared over every
# game that far down, nearer ranks first and rising games (in rank over the last half day, or in players over the
# last day) first.
DEEP_OFFSET = 5           # ranks r with r - k >= this are deep origins for place k
DEEP_DECAY = 10.0         # a rank's share falls by a factor e for every this many ranks further down
RISER_STRENGTH = 3.0      # a full riser score multiplies a game's share by e^3
RISER_LOOKBACK_H = 12.0   # the rank rise is measured against the read closest to this long before the last read
RISER_TOLERANCE = 0.6     # ... accepted only within this fraction of the lookback of that time (else no riser scores)
# Newcomer tail: of what a place leaves after its named games, this part stays unnamed, so the scorer shares it only
# among games new to the chart; the rest is spread equally over the last chart's other games, each named.
NEWCOMER_SHARE = 0.5
MAX_ROWS = 100            # the forecast format's cap on rows per place
MIN_TAIL_P = 0.000001     # no tail when a game's equal share would be under this
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
    # The riser score's reference read: the one closest to RISER_LOOKBACK_H before the last read, if close enough.
    ref_target = last.at - timedelta(hours=RISER_LOOKBACK_H)
    ref_tolerance = RISER_TOLERANCE * RISER_LOOKBACK_H
    ref = None
    for r in reads[:-1]:
        gap = abs((r.at - ref_target).total_seconds()) / 3600.0
        if gap <= ref_tolerance and (ref is None or gap < abs((ref.at - ref_target).total_seconds()) / 3600.0):
            ref = r
    feats = {}
    for uid in last.chart:
        age_days = (last.at - first_seen[uid]).total_seconds() / 86400.0
        old = age_days >= OLD_DAYS
        trend = 0.0
        if earlier is not None:
            now_p = last.playing.get(uid)
            then_p = earlier.playing.get(uid)
            if now_p and then_p and now_p > 0 and then_p > 0:
                trend = max(-1.0, min(1.0, math.log(now_p / then_p)))
        # Riser score in [0, 1]: the larger of the log rank rise since the reference read (a game not on that chart
        # counts as one rank below its end) and the play trend, each clamped to [0, 1]; 0 with no reference read.
        riser = 0.0
        if ref is not None:
            rose = math.log(ref.rank.get(uid, len(ref.chart) + 1) / last.rank[uid])
            riser = max(max(0.0, min(1.0, rose)), max(0.0, min(1.0, trend)))
        feats[uid] = (old, trend, age_days, riser)
    return feats


def occupancy(reads, places):
    """Per place, the share of the last two days' hourly reads each game of the last chart held it (NEW for games no
    longer on the chart), and the coverage of the last day by those reads in [0, 1]. Reads with no other read
    within OCC_GAP_H do not count, so a history of sparse reads gives coverage 0."""
    last = reads[-1]
    shares = [dict() for _ in range(places)]
    total = 0.0
    coverage = 0.0
    for idx, r in enumerate(reads):
        age_h = (last.at - r.at).total_seconds() / 3600.0
        if age_h > 48.0:
            continue
        gap_prev = (r.at - reads[idx - 1].at).total_seconds() / 3600.0 if idx > 0 else float("inf")
        gap_next = (reads[idx + 1].at - r.at).total_seconds() / 3600.0 if idx + 1 < len(reads) else float("inf")
        if min(gap_prev, gap_next) > OCC_GAP_H:
            continue
        span = min(1.0, gap_prev)
        w = span * (1.0 if age_h <= 24.0 else OCC_W2)
        if age_h <= 24.0:
            coverage += span
        total += w
        for k in range(1, min(places, len(r.chart)) + 1):
            uid = r.chart[k - 1]
            key = uid if uid in last.rank else NEW
            shares[k - 1][key] = shares[k - 1].get(key, 0.0) + w
    if total <= 0:
        return None, 0.0
    return [{key: v / total for key, v in d.items()} for d in shares], min(1.0, coverage / 24.0)


def blend_occupancy(dist, occ, last, alpha):
    """Mix a place's transition distribution (keys: ranks of the last chart and NEW) with its occupancy (keys: ids
    and NEW), both expressed over ranks, with weight alpha on the occupancy."""
    out = {key: (1.0 - alpha) * v for key, v in dist.items()}
    for key, v in occ.items():
        rk = NEW if key == NEW else last.rank[key]
        out[rk] = out.get(rk, 0.0) + alpha * v
    return out


def hazard(named, k, last, feats, horizon):
    """Move each game's leaving probability within the horizon to the games below it at the place (or unnamed)."""
    if HAZ_RATE <= 0:
        return named
    out = dict(named)
    order = sorted(out, key=lambda uid: last.rank[uid])
    for uid in order:
        age = feats[uid][2]
        if age <= HAZ_START_D or out.get(uid, 0.0) <= 0:
            continue
        rate = min(HAZ_MAX_RATE, HAZ_RATE * (age - HAZ_START_D))
        leave = 1.0 - math.exp(-rate * horizon / 24.0)
        moved = out[uid] * leave
        out[uid] -= moved
        below = {g: p for g, p in out.items() if last.rank[g] > last.rank[uid] and p > 0}
        s = sum(below.values())
        if s > 0:
            for g, p in below.items():
                out[g] += moved * p / s
    return out


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
        old, trend = feats[uid][0], feats[uid][1]
        factor = math.exp(strength * (TILT_OLD * (1.0 if old else 0.0) + TILT_TREND * trend * (key - k) / TREND_SPAN))
        out[uid] = (v ** SHARPEN) * factor
    s = sum(out.values())
    return {uid: v * named_total / s for uid, v in out.items()}


def deep_share(dist, k, last, feats, length):
    """Re-share the mass place k holds on deep origins (ranks DEEP_OFFSET or more below it) over every rank that deep,
    in proportion to exp(-(r - k - DEEP_OFFSET) / DEEP_DECAY) * exp(RISER_STRENGTH * riser score of the game at r).
    Keys are ranks of the last chart and NEW; NEW and the nearer ranks are untouched. No deep mass, no change."""
    first = k + DEEP_OFFSET
    deep = sum(v for key, v in dist.items() if key != NEW and key >= first)
    if deep <= 0 or first > length:
        return dist
    out = {key: v for key, v in dist.items() if key == NEW or key < first}
    weights = {}
    for r in range(first, length + 1):
        weights[r] = math.exp(-(r - first) / DEEP_DECAY) * math.exp(RISER_STRENGTH * feats[last.chart[r - 1]][3])
    total = sum(weights.values())
    for r, w in weights.items():
        out[r] = deep * w / total
    return out


def forecast(reads, window_from, slot_minutes, slots, places):
    """The forecast document for `reads` (oldest first) over the window."""
    last = reads[-1]
    length = len(last.chart)
    obs = observations(reads, places)
    feats = game_features(reads)
    occ, coverage = occupancy(reads, places)
    acc = [dict() for _ in range(places)]
    for s in range(slots):
        slot_time = window_from + timedelta(minutes=slot_minutes * s)
        horizon = max(0.05, (slot_time - last.at).total_seconds() / 3600.0)
        weights = kernel_weights(obs, horizon)
        dists = place_distributions(obs, weights, places, length)
        strength = min(1.0, horizon / 24.0)
        alpha = OCC_MAX * coverage * min(1.0, horizon / OCC_RAMP_H) if occ else 0.0
        for k in range(1, places + 1):
            dist = blend_occupancy(dists[k - 1], occ[k - 1], last, alpha) if alpha > 0 and k <= OCC_PLACES else dists[k - 1]
            dist = deep_share(dist, k, last, feats, length)
            tilted = hazard(tilt(dist, k, last, feats, strength), k, last, feats, horizon)
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
        rows = sorted(d.items(), key=lambda it: (-it[1], it[0]))[:MAX_ROWS]
        named_total = 0.0
        for uid, p in rows:
            p = round(p, 6)
            probabilities.append({"universe_id": uid, "place": k, "p": p})
            named_total += p
        # The newcomer tail: name every other game of the last chart too, in rank order, each with an equal share of
        # (1 - NEWCOMER_SHARE) of what the place has left. What stays unnamed is then shared, by the scorer, only
        # among the games of the read's chart that were not on the last chart: the newcomers.
        others = [uid for uid in last.chart if uid not in d][:max(0, MAX_ROWS - len(rows))]
        if others:
            share = (1.0 - NEWCOMER_SHARE) * (1.0 - named_total) / len(others)
            if share >= MIN_TAIL_P:
                tail_p = round(share, 6)
                if named_total + tail_p * len(others) > 1.0:
                    tail_p = math.floor(share * 1e6) / 1e6
                for uid in others:
                    probabilities.append({"universe_id": uid, "place": k, "p": tail_p})
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
