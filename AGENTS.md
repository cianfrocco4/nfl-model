# NFL model

Decision support for bets placed at DraftKings. A stake is a suggestion. This tool does not log in, scrape, or place a bet.

## Standing goals

Every future change should serve these goals. A dollar target does not change stake size.

1. Closing-line value above 0 on sides, on totals, and on props, each on its own record. Quarter Kelly waits for that and for 100 graded bets. Live and parlays never upgrade.
2. Stay inside the loss stop. Settled losses of 25% of the passed-in bankroll end new stakes.
3. Season result near flat. About +1% of bankroll by the end of the postseason if a real edge of about 2 points is bet at the locked flat stake. Missing it is an ordinary outcome.
4. Log every bet: stake, American price, result, and closing number. A row with no close does not qualify. A row with no stake does not move the loss stop.
5. Keep sleeves separate in staking and in profit reporting. Sides are almost all of the dollars. Props, live, and parlays are small.

20% of bankroll in 3 weeks is not a reasonable expectation at the locked stake. `--goal-return` may display a window like that and must say when the slate cannot produce it. It must not change `stake_dollars`.

## Constraints on later changes

- The default fair price stays `both`. A stake requires the model and the market on the same side. One signal is a lean with stake $0.
- Flat 1% of the sleeve until that record qualifies, then quarter Kelly capped at 5% of the sleeve.
- Backtest ROI is the model against the nflverse close. It is not the both-agree rule and it is not the product result.
- Sides CLV mixes spread points and moneyline decimals. Props CLV mixes yards, receptions, and touchdowns. Do not add more mixed units. Prefer same-unit CLV, or the share of bets with positive CLV, if qualification changes.
- The ledger has no placed-on date, so a time window counts the whole file. Add a date and count rows inside the window, or archive the file when a window starts.
- The loss stop cannot see a bet that was never written down. New stake suggestions should be loggable.
