# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/minhvo7332/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/minhvo7332/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Logit-space CQR quantile fan v1: realised-move-scaled trophy earn-rate forecast. For each of GTA VI's 55 PlayStation trophies it forecasts five quantiles (5%, 25%, 50%, 75%, 95%) of the Exophase earn rate at the target, from the case file alone — no network, no model calls. Method: per-trophy logit trend against log-time plus an earners/players decomposition, blended by within-case walk-forward backtest, with a split-conformal (CQR) calibration layer that widens the fan to the coverage the case's own history actually supports. Memorization-proof: the final case is never in the fit; every parameter is estimated from the case file it is run on.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=ice777&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
