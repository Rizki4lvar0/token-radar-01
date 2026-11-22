# token-radar-01

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/Rizki4lvar0/token-radar-01/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/Rizki4lvar0/token-radar-01/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

A token launch radar: it reads a sealed snapshot of 758 protocols, prices every pre-token protocol it can rank with a value model fitted from the 128 peers whose tokens already trade, and shows a reader how to farm each one. The board holds 203 ranked protocols — all 40 on the watchlist, plus 163 pre-token protocols below the floor — each with an implied value, an uncertainty interval, and a six-step farm plan with costs and times.

## How the value is priced

Implied value = TVL × the market-cap-to-TVL ratio the peers pay. The ratio is fitted at run time from the snapshot's own peers: an OLS fit of log10(market-cap / TVL) on log TVL, seven-day growth, age and chain count, averaged with a shrunk category median, clamped to the peers' 5th–95th percentile range. The fit is validated held-out (every 5th peer by slug) and walk-forward (older listings train, newer listings test), and the dashboard states both errors plainly — the peer fit is noisy, so every row carries an interval instead of a single confident number. No protocol name is special-cased.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input <kit>/input/snapshot.json
```

`main.ts` needs Node.js 22.18 or later (Node 24 works) and nothing else — standard library only, no network. It writes `out/board.json`, `out/data.js`, `out/dashboard/`, `out/screenshots/` and `out/dossier.json`. Open `out/dashboard/index.html` in a browser; the page is static and makes no network requests.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
