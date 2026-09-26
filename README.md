# NFL betting model

Decision support for bets you place yourself at DraftKings. A stake is a suggestion, not an order, and it is not a promise of profit.

The operating rule, locked 2026-09-26:

- Recommend a stake only when the statistical model and the sharp or consensus price are both +EV on the same side of the DraftKings line, after the vig is removed. If only one is +EV, it is shown as a lean and the stake is $0.
- Bankroll sleeves: **65%** pregame sides, totals, and moneylines, **20%** pregame props, **10%** live, **5%** parlays.
- Straight bets are a flat **1%** of that sleeve until the record qualifies, then quarter Kelly capped at **5%** of that sleeve. A record qualifies with at least 100 graded bets and an average closing-line value above 0. Sides (spreads and moneylines), totals, and props each qualify on their own record. Live stays flat.
- Parlays are same-game and cross-game, **2 or 3 legs only**. Stake flat **1%** of the parlay sleeve when the DraftKings parlay price is +EV against the fair joint probability and every leg that has a straight price passes the both-agree rule. Parlays do not upgrade to quarter Kelly.

Other straight-bet staking methods (`flat`, `units`, `full`, `half`, `quarter`) stay selectable with `--staking`. The edge threshold is a setting (`--min-edge`, default 2 percentage points), not a fixed cutoff.

## What you need

- Python 3.11+
- A free API key from [The Odds API](https://the-odds-api.com), which publishes DraftKings as a US book
- Your bankroll in dollars

This project does not log in to DraftKings, scrape it, or place a bet.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
export THE_ODDS_API_KEY=your_key_here
export NFL_BANKROLL=1000
```

Sign up at https://the-odds-api.com, open the dashboard, and copy the key. Do not commit it. The backtest and the qualification check run with no key. The first backtest downloads public nflverse schedules and weekly player stats and caches them in `.cache`.

## Commands

```bash
python -m nfl_model backtest
python -m nfl_model qualification
python -m nfl_model board --bankroll 1000
python -m nfl_model props --bankroll 1000
python -m nfl_model live --bankroll 1000
python -m nfl_model parlays --bankroll 1000
python -m nfl_model parlays --bankroll 1000 --slip 1,2:+265
```

`board`, `props`, `live`, and `parlays` exit with setup instructions when `THE_ODDS_API_KEY` is missing.

## How to read a straight bet

Each row shows the DraftKings American price, the model probability and edge, and the market probability and edge. The market price is Pinnacle with the vig removed when Pinnacle has the same number. Otherwise it is the average of the other books on that number. A different number is not treated as the same bet.

- **stake**: model and market are both +EV. The dollars are the suggestion.
- **lean**: only one signal is +EV, or the market price is missing. Stake $0.
- **pass**: neither signal clears `--min-edge` and `--min-ev`.

Edge is the model's win probability (ignoring pushes) minus the no-vig DraftKings probability. Expected value uses the actual DraftKings payout, including a push as a refund. Spreads and moneylines share the 65% sleeve. Totals use that same sleeve but keep a separate closing-line record.

## Parlays

`parlays` lists every priced straight leg, then fair prices for 2- and 3-leg tickets whose legs already both-agree.

- Cross-game: the joint probability is the product of the leg probabilities.
- Same-game spread, total, and moneyline: one normal margin and one normal total, with the correlation estimated from historical residuals. A moneyline and a spread on the same team are the same margin, so the joint is the stricter leg, not the product.
- Same-game props: a correlated draw. Passing yards, rushing yards, receiving yards, receptions, and anytime touchdowns use shrunk correlations with the game total and the player's team margin. Anytime touchdown is a rushing, receiving, return, or fumble score, not a passing touchdown.
- A ticket that mixes two games, with two legs from one of them, uses the same-game joint for that pair and multiplies the other game.

The Odds API does not publish DraftKings parlay or same-game parlay prices. Copy the price from the slip:

```bash
python -m nfl_model parlays --bankroll 1000 --slip 1,2:+265
python -m nfl_model parlays --bankroll 1000 --slip 4,5,9:-150
```

The numbers are the leg list printed by the command. If a selected leg did not both-agree, the ticket is a lean and the stake is $0. With a $1,000 bankroll the parlay sleeve is $50, so a qualified ticket is staked at $0.50. Live parlays are not priced. More than 3 legs are rejected.

## Closing-line value

Closing-line value is the closing number minus the bet number, in the direction of the bet. Positive means the close moved your way.

| Direction | Formula | Example |
| --- | --- | --- |
| over | close − bet | 49 − 47.5 = +1.5 |
| under | bet − close | 47.5 − 49 = −1.5 |
| spread | your point − close point | −3.5 − (−6.5) = +3 |
| decimal | your decimal odds − close decimal odds | used for moneylines |

Track graded bets in `ledger.csv` (see `ledger.example.csv`):

```csv
record,bet_number,close_number,direction,note
sides,-3.5,-6.5,spread,home spread
totals,47.5,49,over,game total
props,64.5,71.5,over,receiving yards
```

`record` is `sides`, `totals`, `props`, `live`, or `parlays`. Live and parlays are shown but never upgrade. Until a straight record has 100 graded rows and a positive average, that record stays at flat 1%.

## Environment

| Variable | Default | Role |
| --- | --- | --- |
| `THE_ODDS_API_KEY` | unset | Required for board, props, live, and parlays |
| `NFL_BANKROLL` | unset | Dollars. Required for those commands |
| `NFL_FAIR_PRICE` | `both` | `model`, `market`, or `both` |
| `NFL_STAKING` | `policy` | Locked rule, or `flat`, `units`, `full`, `half`, `quarter` |
| `NFL_FLAT_FRACTION` | `0.01` | Flat stake and the pre-qualification stake |
| `NFL_MAX_STAKE_FRACTION` | `0.05` | Cap, including quarter Kelly |
| `NFL_MIN_EDGE` | `0.02` | Edge threshold |
| `NFL_MIN_EV` | `0` | Minimum expected value |
| `NFL_BANKROLL_MODE` | `sleeves` | `sleeves` or `shared` |
| `NFL_SLEEVE_SIDES` | `0.65` | With the other sleeves, must sum to 1 |
| `NFL_SLEEVE_PROPS` | `0.20` | |
| `NFL_SLEEVE_LIVE` | `0.10` | |
| `NFL_SLEEVE_PARLAYS` | `0.05` | |
| `NFL_UNIT_SIZE` | 1% of the sleeve | Used only with `--staking units` |

## Model

Team scores use each club's last 10 games of points scored and allowed, plus home field, fit by ridge regression. A week is predicted only from games already played. Margins and totals are normal, so whole-number lines can push and half-point lines cannot. DraftKings moneylines are treated as a push on a tie.

Player props shrink a recent average toward the position and scale it by the opponent's recent yards or scores allowed. Yards and receptions are normal. Anytime touchdown is Poisson.

Live lines start from that pregame forecast, then add the current score and the remaining fraction of the game. The fraction blends minutes since kickoff (about 190 minutes of wall clock) with the share of the pregame total already scored. It is not the official clock, and it does not know possession, down, or timeouts. The scores feed has no player box score, so live props are not recommended even if DraftKings posts them.

## Limits

- Outputs are decision support. They can lose money.
- Historical closes in the backtest are nflverse consensus lines, not DraftKings. Closing-line value is not measured there.
- The both-agree rule cannot be backtested on that file, because it has one price rather than a model price and a separate DraftKings price.
- Prop backtest is prediction error only. Historical prop prices are not in the open data.
- Parlay backtest is not available. DraftKings does not publish parlay prices through The Odds API.
- Same-game prop correlations are shrunk defaults plus the estimated margin/total correlation. They are not a play-by-play same-game model.
- Several bets on the same slate share one sleeve. If the suggestions add up to more than the sleeve, they are scaled down.
- Player names are matched by a normalized string. A mismatch is skipped.
- No parlays longer than 3 legs, no live parlays, no arbitrage, and no martingale.

## Tests

```bash
pytest
```
