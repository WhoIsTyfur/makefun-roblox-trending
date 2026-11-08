"""Convert raw hourly reads of Roblox's top-trending sort (data/live/<stamp>.json, the explore-api response with
`games`: universeId, rootPlaceId, name, playerCount, isSponsored) into the history.ndjson read shape, one line per
file, oldest first, written to data/live/house.ndjson. The stamp (e.g. 2026-11-08T060000Z) is the read time."""

import glob
import json
import os
import re
import sys

STAMP = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2})(\d{2})(\d{2})Z$")


def convert(path):
    name = os.path.splitext(os.path.basename(path))[0]
    m = STAMP.match(name)
    if not m:
        return None
    y, mo, d, h, mi, s = m.groups()
    at = f"{y}-{mo}-{d}T{h}:{mi}:{s}Z"
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    games = doc.get("games") if isinstance(doc, dict) else None
    if not isinstance(games, list) or not games:
        return None
    rows = []
    rank = 0
    for g in games:
        if g.get("isSponsored"):
            continue
        rank += 1
        rows.append({
            "rank": rank,
            "universe_id": g["universeId"],
            "root_place_id": g.get("rootPlaceId"),
            "name": g.get("name"),
            "playing": g.get("playerCount"),
            "up_votes": g.get("totalUpVotes"),
            "down_votes": g.get("totalDownVotes"),
            "genre": g.get("genreL1"),
        })
    return {
        "at": at,
        "source": "local/roblox-chart-recorder",
        "ok": True,
        "top_trending": rows,
        "top_playing_now": None,
        "details": None,
        "responses": None,
    }


def main(folder="data/live"):
    out = []
    for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
        read = convert(path)
        if read is not None:
            out.append(read)
    out.sort(key=lambda r: r["at"])
    target = os.path.join(folder, "house.ndjson")
    with open(target, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(out)} reads -> {target}")


if __name__ == "__main__":
    main(*sys.argv[1:])
