"""Walk-forward backtest of the score model and player props.

nflverse schedules include one consensus close, not a bet-time number and not a
separate DraftKings price. Closing-line value is not measured here, so this
command does not upgrade a sleeve to quarter Kelly. Parlay tickets are not
backtested: historical DraftKings parlay prices are not in the open data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_model.data_nflverse import completed_games, load_games, load_player_weeks
from nfl_model.odds_math import american_to_decimal, expected_value, implied_probability, remove_vig
from nfl_model.parlay import margin_total_correlation
from nfl_model.probabilities import moneyline_outcomes, spread_outcomes, total_outcomes
from nfl_model.props_model import build_prop_state
from nfl_model.settings import Settings
from nfl_model.team_model import nflverse_home_spread, walk_forward


def run_backtest(settings: Settings, refresh: bool = False, eval_start: int = 2018) -> str:
    games = load_games(refresh=refresh)
    done = completed_games(games)
    predictions, _state = walk_forward(done)
    if predictions.empty:
        return "No completed games were available to score."
    window = predictions[predictions["season"] >= eval_start].copy()
    margin_error = (window["home_score"] - window["away_score"]) - window["mu_margin"]
    total_error = (window["home_score"] + window["away_score"]) - window["mu_total"]
    rho = margin_total_correlation(margin_error.to_numpy(), total_error.to_numpy())
    home_win = (window["home_score"] > window["away_score"]).astype(float)
    probs = [
        moneyline_outcomes(row.mu_margin, row.sigma_margin)[0]
        for row in window.itertuples(index=False)
    ]
    brier = float(np.mean((np.array(probs) - home_win.to_numpy()) ** 2))
    accuracy = float(np.mean((np.array(probs) >= 0.5) == (home_win.to_numpy() == 1.0)))
    lines = [
        "NFL model backtest (walk-forward, no API key, no bets placed)",
        f"Completed games used for fitting: {len(predictions)}",
        f"Scored seasons: {int(window['season'].min())}-{int(window['season'].max())} ({len(window)} games)",
        f"Margin MAE: {float(margin_error.abs().mean()):.2f} points",
        f"Total MAE: {float(total_error.abs().mean()):.2f} points",
        f"Home-win accuracy: {accuracy:.1%}",
        f"Home-win Brier score: {brier:.4f}",
        f"Residual correlation of margin and total: {rho:+.2f} (used for same-game parlays)",
        "",
        "Closing-line value was not measured.",
        "nflverse publishes one consensus close (spread_line, total_line, moneylines).",
        "It does not publish the number available when a bet would have been placed,",
        "and it is not a DraftKings history. Sleeves stay on flat staking until",
        "ledger.csv has 100 graded bets and a positive average closing-line value.",
        "",
        "The locked rule needs the model and a separate sharp-versus-DraftKings price.",
        "Those two prices are not in this file, so historical stakes for that rule are not simulated.",
        "Parlay staking is not backtested. Historical DraftKings parlay prices are not in the open data.",
        "",
        _diagnostic(window, settings),
        "",
        _props(refresh),
    ]
    return "\n".join(lines)


def _diagnostic(window: pd.DataFrame, settings: Settings) -> str:
    """Model-only results against the consensus close, at flat 1% of the sides sleeve.

    This is not the both-agree rule and it is not closing-line value. The price
    taken is the closing price itself.
    """
    bankroll = settings.bankroll if settings.bankroll is not None else 1000.0
    sleeve = bankroll * settings.sleeve_sides
    stake = sleeve * settings.flat_fraction
    profit = 0.0
    bets = 0
    wins = 0
    for game in window.itertuples(index=False):
        for piece in (_spread_profit(game, stake), _total_profit(game, stake), _moneyline_profit(game, stake)):
            profit += piece[0]
            bets += piece[1]
            wins += piece[2]
    if bets == 0:
        return "Diagnostic model-versus-close: no priced closing lines in the scored window."
    roi = profit / (bets * stake) if stake else 0.0
    return (
        "Diagnostic only, model versus the consensus close, flat "
        f"{settings.flat_fraction:.0%} of a ${sleeve:,.0f} sides sleeve (${stake:,.2f} a bet).\n"
        f"Bets {bets}, wins {wins}, profit ${profit:,.2f}, ROI {roi:.1%}.\n"
        "Do not read this as DraftKings results or as the both-agree rule."
    )


def _spread_profit(game, stake: float) -> tuple[float, int, int]:
    if pd.isna(game.spread_line) or pd.isna(game.home_spread_odds) or pd.isna(game.away_spread_odds):
        return 0.0, 0, 0
    home_point = nflverse_home_spread(float(game.spread_line))
    home_p, push, away_p = spread_outcomes(game.mu_margin, game.sigma_margin, home_point)
    novig_home, novig_away = remove_vig(
        [implied_probability(game.home_spread_odds), implied_probability(game.away_spread_odds)]
    )
    margin = game.home_score - game.away_score
    return _choose(
        stake,
        ("home", home_p, push, novig_home, game.home_spread_odds, margin + home_point),
        ("away", away_p, push, novig_away, game.away_spread_odds, -(margin + home_point)),
    )


def _total_profit(game, stake: float) -> tuple[float, int, int]:
    if pd.isna(game.total_line) or pd.isna(game.over_odds) or pd.isna(game.under_odds):
        return 0.0, 0, 0
    over_p, push, under_p = total_outcomes(game.mu_total, game.sigma_total, float(game.total_line), "over")
    novig_over, novig_under = remove_vig(
        [implied_probability(game.over_odds), implied_probability(game.under_odds)]
    )
    total = game.home_score + game.away_score
    return _choose(
        stake,
        ("over", over_p, push, novig_over, game.over_odds, total - float(game.total_line)),
        ("under", under_p, push, novig_under, game.under_odds, float(game.total_line) - total),
    )


def _moneyline_profit(game, stake: float) -> tuple[float, int, int]:
    if pd.isna(game.home_moneyline) or pd.isna(game.away_moneyline):
        return 0.0, 0, 0
    home_p, tie, away_p = moneyline_outcomes(game.mu_margin, game.sigma_margin)
    novig_home, novig_away = remove_vig(
        [implied_probability(game.home_moneyline), implied_probability(game.away_moneyline)]
    )
    margin = game.home_score - game.away_score
    return _choose(
        stake,
        ("home", home_p, tie, novig_home, game.home_moneyline, margin),
        ("away", away_p, tie, novig_away, game.away_moneyline, -margin),
    )


def _choose(stake: float, *sides) -> tuple[float, int, int]:
    """Settle the model-positive side with the better expected value."""
    best = None
    for _name, p_win, p_push, novig, american, result in sides:
        decimal_odds = american_to_decimal(american)
        ev = expected_value(p_win, p_push, decimal_odds)
        decisive = p_win + (1.0 - p_win - p_push)
        edge = (p_win / decisive if decisive else 0.0) - novig
        if ev > 0 and edge > 0 and (best is None or ev > best[0]):
            best = (ev, result, decimal_odds)
    if best is None:
        return 0.0, 0, 0
    result, decimal_odds = best[1], best[2]
    if abs(result) < 1e-9:
        return 0.0, 1, 0
    if result > 0:
        return stake * (decimal_odds - 1.0), 1, 1
    return -stake, 1, 0


def _props(refresh: bool) -> str:
    try:
        stats = load_player_weeks(refresh=refresh)
        state = build_prop_state(stats)
    except Exception as exc:  # noqa: BLE001 - surface a data failure without hiding the team backtest
        return f"Player-prop accuracy was not scored: {exc}"
    parts = []
    labels = {
        "passing_yards": "Passing yards MAE",
        "rushing_yards": "Rushing yards MAE",
        "receiving_yards": "Receiving yards MAE",
        "receptions": "Receptions MAE",
        "anytime_td": "Anytime-TD Brier",
    }
    for key, label in labels.items():
        value = state.accuracy.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            parts.append(f"{label}: n/a")
        else:
            parts.append(f"{label}: {value:.3f}" if key == "anytime_td" else f"{label}: {value:.2f}")
    return (
        "Player props, prediction error only, on weeks a player was used "
        "(QB 10+ attempts, 5+ carries, or 3+ targets). "
        "No historical prop prices, so no edge or closing-line value.\n"
        + "\n".join(parts)
    )
