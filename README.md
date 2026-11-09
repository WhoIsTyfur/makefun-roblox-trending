# Roblox Top Trending forecast

An entry for make.fun's bounty "Predict tomorrow's top 10 trending Roblox games"
(roblox-trending-forecast-01). `main.py` reads the history of Roblox's Top Trending chart and writes one
forecast: for each of the chart's top 10 places, the probability of each game holding that place over the 24
hourly slots after the history ends, plus a point guess at the top 10.

## What it does

Each place is scored on its own, so the program forecasts each place's marginal distribution and averages it
over the window's 24 slots.

1. It loads every valid read of the chart, orders each by rank, drops duplicates and near-duplicate reads.
2. Every pair of reads up to 30 hours apart is an observation: for each of the ten places on the later read,
   where that game had been on the earlier read (an offset), or that it was not on the chart at all.
3. For each slot, the horizon is the time from the last read to the slot. Observations are weighted by a
   triangular kernel on the difference between their lag and the horizon, widened until enough weight is
   present. Per place, four offset distributions are mixed with pseudo-observation weights: the place's own,
   its band of neighbouring places, all places pooled, and a smooth prior centred on the place.
4. Offsets are mapped to games through the last read's chart. When the history holds hourly reads, places 1
   and 2 also blend in an occupancy share: how much of the last day (and, at a lower weight, the day before)
   each game held the place. The chart follows a daily cycle in concurrent players, and one forecast is
   scored against every hour of a day, so a game's share of the day is close to the right probability. The
   blend grows with the hourly coverage of the last day and with the horizon; with no hourly reads the
   forecast is unchanged. Lower places drift down over a day, so their past share predicts less and they are
   left to the transition model.
5. Deep-origin share. Now and then a game takes a place from five or more places below it. The offset model
   would put that mass on whichever game happens to sit at the exact rank a past riser came from, so the
   place's deep mass is pooled and shared over every game that far down instead: nearer ranks first, and
   rising games first (in rank over the last half day, or in players over the last day).
6. Each game's probability is tilted by its player-count trend over the previous day, keeping the place's
   named total unchanged. Then an age hazard applies: the chart ranks the past week's relative growth in
   daily users, so a game's chance of leaving rises once it has been on the chart about a week. The hazard is
   mild and capped at the leaving rate the data shows, and the mass it removes from a game passes to the
   games below it at that place, which move up when it leaves.
7. Slots are averaged and the games above a small threshold are named at the place. Newcomer tail: every
   other game of the last chart is named too, each with an equal share of half of what the place has left.
   The scorer shares a place's unnamed remainder among the chart's games the forecast does not name there,
   so the half kept unnamed goes only to games new to the chart, not thinly over the whole chart.

## How it is scored

For each hourly read in the window and each place, the forecast's probability on the actual game is taken
(a named row's p, or the unnamed remainder shared equally among the chart's unnamed games), floored at
0.0001, and its natural log summed over the ten places. The log score is the mean over the window's reads;
higher is better. `tools/score.py` reimplements that rule and reproduces the house baseline's published
practice score to six decimals.

## Running it locally

```
MAKEFUN_INPUT_DIR=<folder with history.ndjson and task.json> MAKEFUN_OUTPUT_DIR=out python main.py
```

Python 3.13, standard library only, no network. The forecast is deterministic: two runs on the same input
produce the same bytes. On the bounty's practice input the run takes well under a second.

## Tools

- `tools/score.py`: the scorer, as a module and as a command that scores a forecast file against the reads
  of a window.
- `tools/backtest.py`: rolls a window over every read of the archive with enough history before it, scores
  the model and the house baseline on the reads that follow, and reports means by horizon and by place.
  `--set NAME=VALUE` overrides the model's constants for parameter sweeps; `--hourly` is the hourly test, which
  cuts the history only at hourly reads and scores only against the hourly reads that follow.
- `tools/convert_live.py`: converts raw hourly reads of the chart's API response into the read shape used by
  the history, for extending the backtest with hourly data.

## Data

`data/archive/reads.ndjson` holds reads of the chart from the public archive
[Kenny-Dhub/roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots), used under its MIT licence
(see `data/archive/NOTICE.txt`): the rank and player count of every game on the chart, about every 4 to 10
hours from 2026-10-30. The backtest runs on it; the program itself reads only the input folder it is given.
