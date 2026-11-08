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
4. Offsets are mapped to games through the last read's chart. Each game's probability is tilted by its age
   on the chart (games tend to leave after about a week) and by its player-count trend over the previous
   day, then rescaled so the place's named total is unchanged.
5. Slots are averaged. Games below a small threshold are left unnamed, so their mass goes to the scorer's
   evenly shared remainder, which also covers games that enter the chart from outside it.

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
  `--set NAME=VALUE` overrides the model's constants for parameter sweeps.
- `tools/convert_live.py`: converts raw hourly reads of the chart's API response into the read shape used by
  the history, for extending the backtest with hourly data.

## Data

`data/archive/reads.ndjson` holds reads of the chart from the public archive
[Kenny-Dhub/roblox-snapshots](https://github.com/Kenny-Dhub/roblox-snapshots), used under its MIT licence
(see `data/archive/NOTICE.txt`): the rank and player count of every game on the chart, about every 4 to 10
hours from 2026-10-30. The backtest runs on it; the program itself reads only the input folder it is given.
