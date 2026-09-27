"""Command line for straight bets and 2- or 3-leg parlays.

Nothing in this tool logs into DraftKings or places a bet.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from nfl_model.backtest import run_backtest
from nfl_model.data_nflverse import completed_games, load_games, load_player_weeks, project_root
from nfl_model.odds_api import (
    MissingApiKeyError,
    OddsApiError,
    build_side_prices,
    fetch_event_props,
    fetch_game_odds,
    fetch_scores,
    parse_outcomes,
    parse_scores,
    require_api_key,
)
from nfl_model.parlay import (
    DEFAULT_PROP_RHO,
    ParlayLeg,
    apply_slip,
    game_leg_from_moneyline,
    game_leg_from_spread,
    game_leg_from_total,
    margin_total_correlation,
    prop_leg,
    render_parlays,
    suggest_tickets,
)
from nfl_model.probabilities import total_outcomes
from nfl_model.props_model import MARKET_FOR_STAT, build_prop_state, normalize_name, project_stat
from nfl_model.qualification import (
    make_profit_goal,
    qualification_report,
    season_money,
    season_progress,
    season_stop_note,
)
from nfl_model.recommend import (
    EASTERN,
    EASTERN,
    apply_season_loss_limit,
    live_board,
    pregame_board,
    prop_board,
    qualified_map,
    render,
    signals_agree,
    slate_end,
)
from nfl_model.settings import MissingBankrollError, Settings, SettingsError, load_settings
from nfl_model.staking import fit_stakes_to_cash
from nfl_model.team_model import forecast_matchup, walk_forward
from nfl_model.teams import franchise_from_name

STAT_FOR_MARKET = {market: stat for stat, market in MARKET_FOR_STAT.items()}


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2
    try:
        return int(args.func(args))
    except (SettingsError, MissingBankrollError, OddsApiError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--bankroll", type=float, default=None)
    common.add_argument("--fair-price", choices=["model", "market", "both"], default=None)
    common.add_argument("--staking", choices=["policy", "flat", "units", "full", "half", "quarter"], default=None)
    common.add_argument("--bankroll-mode", choices=["shared", "sleeves"], default=None)
    common.add_argument("--flat-fraction", type=float, default=None)
    common.add_argument("--unit-size", type=float, default=None)
    common.add_argument("--max-stake-fraction", type=float, default=None)
    common.add_argument("--min-edge", type=float, default=None)
    common.add_argument("--min-ev", type=float, default=None)
    common.add_argument("--sleeve-sides", type=float, default=None)
    common.add_argument("--sleeve-props", type=float, default=None)
    common.add_argument("--sleeve-live", type=float, default=None)
    common.add_argument("--sleeve-parlays", type=float, default=None)
    common.add_argument(
        "--loss-limit-fraction",
        type=float,
        default=None,
        help="Stop new stakes after settled losses reach this share of the bankroll. Default is 0.25.",
    )
    common.add_argument(
        "--goal-return",
        type=float,
        default=None,
        help="Profit goal as a fraction of the bankroll. 20%% is 0.20. Does not change the stake.",
    )
    common.add_argument(
        "--goal-weeks",
        type=float,
        default=None,
        help="Weeks allowed to reach --goal-return.",
    )
    common.add_argument(
        "--goal-start",
        default=None,
        help="First day of the profit-goal window, YYYY-MM-DD.",
    )
    common.add_argument("--ledger", type=Path, default=None)
    common.add_argument("--refresh", action="store_true")
    common.add_argument("--verbose", action="store_true", help="Show leans, passes, and every priced side.")
    common.add_argument(
        "--days",
        type=float,
        default=None,
        help="How many days of kickoffs to include. Default is the current NFL week, through Tuesday noon Eastern.",
    )
    parser = argparse.ArgumentParser(
        prog="nfl_model",
        description=(
            "NFL decision support for DraftKings. A stake is a suggestion. "
            "You place the bet yourself. Parlays are 2 or 3 legs."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    backtest = sub.add_parser("backtest", parents=[common], help="Score historical games. No API key.")
    backtest.add_argument("--eval-start", type=int, default=2018)
    backtest.set_defaults(func=_backtest)

    board = sub.add_parser("board", parents=[common], help="This week's pregame spread, total, and moneyline.")
    board.set_defaults(func=_board)

    props = sub.add_parser("props", parents=[common], help="Pregame passing, rushing, receiving, receptions, anytime TD.")
    props.add_argument("--max-events", type=int, default=14)
    props.set_defaults(func=_props)

    live = sub.add_parser("live", parents=[common], help="In-play spread, total, and moneyline.")
    live.set_defaults(func=_live)

    parlays = sub.add_parser("parlays", parents=[common], help="2- and 3-leg parlays. Pass DraftKings slip prices with --slip.")
    parlays.add_argument("--max-events", type=int, default=8)
    parlays.add_argument("--slip", action="append", default=[], help="Leg numbers and DraftKings price, e.g. 1,2:+265")
    parlays.add_argument("--no-props", action="store_true", help="Use game markets only.")
    parlays.set_defaults(func=_parlays)

    qualification = sub.add_parser(
        "qualification",
        parents=[common],
        help="Show quarter-Kelly qualification, the season stop, and any profit goal.",
    )
    qualification.set_defaults(func=_qualification)
    return parser


def _settings(args, *, fallback: float | None = None) -> Settings:
    return load_settings(
        fair_price=args.fair_price,
        staking=args.staking,
        bankroll_mode=args.bankroll_mode,
        bankroll=args.bankroll,
        flat_fraction=args.flat_fraction,
        unit_size=args.unit_size,
        max_stake_fraction=args.max_stake_fraction,
        min_edge=args.min_edge,
        min_ev=args.min_ev,
        sleeve_sides=args.sleeve_sides,
        sleeve_props=args.sleeve_props,
        sleeve_live=args.sleeve_live,
        sleeve_parlays=args.sleeve_parlays,
        loss_limit_fraction=args.loss_limit_fraction,
        goal_return=args.goal_return,
        goal_weeks=args.goal_weeks,
        goal_start=args.goal_start,
        bankroll_fallback=fallback,
    )


def _ledger(args) -> Path:
    return args.ledger if args.ledger is not None else project_root() / "ledger.csv"


def _qualified(args) -> dict[str, bool]:
    return qualified_map(qualification_report(_ledger(args)))


def _money(args):
    return season_money(_ledger(args))


def _until(args) -> datetime:
    return slate_end(datetime.now(timezone.utc), args.days)


def _progress(settings: Settings, ledger: Path) -> str:
    flat_stake = None
    if settings.bankroll is not None:
        flat_stake = settings.pool("sides") * settings.flat_fraction
    return season_progress(
        season_money(ledger),
        settings.bankroll,
        settings.loss_limit_fraction,
        goal=make_profit_goal(settings.goal_return, settings.goal_weeks, settings.goal_start),
        today=datetime.now(EASTERN).date(),
        flat_stake=flat_stake,
        min_edge=settings.min_edge,
    )


def _headline(settings: Settings, until: datetime | None, verbose: bool, ledger: Path) -> None:
    if verbose:
        print(settings.describe())
    elif settings.bankroll is None:
        print("Bankroll is not set.")
    else:
        print(
            f"Bankroll ${settings.bankroll:,.2f}. "
            "A flat stake is 1% of that sleeve until the record qualifies."
        )
    print(_progress(settings, ledger))
    if until is not None:
        local = until.astimezone(EASTERN)
        print(f"Slate: kickoffs before {local:%a %b %d %I:%M %p ET}. Widen with --days.")
    print()


def _backtest(args) -> int:
    settings = _settings(args, fallback=1000.0)
    print(settings.describe())
    print()
    print(run_backtest(settings, refresh=args.refresh, eval_start=args.eval_start))
    return 0


def _board(args) -> int:
    settings = _settings(args)
    key = require_api_key()
    until = _until(args)
    _headline(settings, until, args.verbose, _ledger(args))
    state, _predictions, rho = _model(args.refresh)
    payload = fetch_game_odds(key)
    _credits(payload.remaining, payload.last_cost)
    rows = apply_season_loss_limit(
        pregame_board(
            build_side_prices(parse_outcomes(payload.body)),
            state,
            settings,
            _qualified(args),
            until=until,
        ),
        settings,
        _money(args),
    )
    print(render(rows, "Pregame board", verbose=args.verbose))
    print(f"\nSame-game margin/total correlation from history: {rho:+.2f}")
    return 0


def _props(args) -> int:
    settings = _settings(args)
    key = require_api_key()
    until = _until(args)
    _headline(settings, until, args.verbose, _ledger(args))
    props = build_prop_state(load_player_weeks(refresh=args.refresh))
    prices, notes = _prop_prices(key, args.max_events, until)
    for note in notes:
        print(note)
    rows = apply_season_loss_limit(
        prop_board(prices, props, settings, _qualified(args), until=until),
        settings,
        _money(args),
    )
    print(render(rows, "Pregame props", verbose=args.verbose))
    return 0


def _live(args) -> int:
    settings = _settings(args)
    key = require_api_key()
    _headline(settings, None, args.verbose, _ledger(args))
    state, _predictions, _rho = _model(args.refresh)
    odds = fetch_game_odds(key)
    scores = fetch_scores(key)
    _credits(odds.remaining, odds.last_cost)
    rows, notes = live_board(
        build_side_prices(parse_outcomes(odds.body)),
        parse_scores(scores.body),
        state,
        settings,
        _qualified(args),
    )
    rows = apply_season_loss_limit(rows, settings, _money(args))
    for note in notes:
        print(note)
    print(render(rows, "Live board", verbose=args.verbose))
    print(
        "\nLive prices use the current score and a blend of wall-clock time and scoring pace. "
        "They do not know possession, down, or the official clock. Live props are not staked."
    )
    return 0


def _parlays(args) -> int:
    settings = _settings(args)
    key = require_api_key()
    until = _until(args)
    _headline(settings, until, args.verbose, _ledger(args))
    state, predictions, rho = _model(args.refresh)
    odds = fetch_game_odds(key)
    _credits(odds.remaining, odds.last_cost)
    prices = build_side_prices(parse_outcomes(odds.body))
    qualified = _qualified(args)
    legs = _game_legs(prices, state, settings, qualified, until)
    if not args.no_props:
        prop_prices, notes = _prop_prices(key, args.max_events, until)
        for note in notes:
            print(note)
        props = build_prop_state(load_player_weeks(refresh=args.refresh))
        legs.extend(_prop_legs(prop_prices, props, settings, qualified, until))
    suggestions = suggest_tickets(legs, settings, rho)
    priced = []
    for slip in args.slip:
        ticket = apply_slip(legs, slip, settings, rho)
        priced.append(ticket)
    if priced:
        stakes = fit_stakes_to_cash([ticket.stake for ticket in priced], settings.pool("parlays"))
        priced = [
            replace(
                ticket,
                stake=stake,
                status="stake" if stake > 0 else ("pass" if ticket.status == "stake" else ticket.status),
                reason=ticket.reason
                + (" Stake scaled so these slips fit in the parlay sleeve." if stake < ticket.stake else ""),
            )
            for ticket, stake in zip(priced, stakes)
        ]
    priced = _stop_tickets(priced, settings, _money(args))
    print(render_parlays(legs, suggestions, priced, verbose=args.verbose))
    print(f"\nMargin/total correlation used for same-game tickets: {rho:+.2f}")
    print(f"History rows behind that correlation: {len(predictions)}")
    return 0


def _stop_tickets(tickets, settings: Settings, money):
    note = season_stop_note(money, settings.bankroll, settings.loss_limit_fraction)
    if note is None:
        return tickets
    stopped = []
    for ticket in tickets:
        if ticket.stake <= 0:
            stopped.append(ticket)
            continue
        stopped.append(replace(ticket, stake=0.0, status="pass", reason=f"{ticket.reason} {note}"))
    return stopped


def _qualification(args) -> int:
    settings = _settings(args)
    if settings.bankroll is None:
        print("Bankroll is not set. Pass --bankroll or NFL_BANKROLL to size the season stop.")
    else:
        print(settings.describe())
    print()
    ledger = _ledger(args)
    print(f"Ledger: {ledger}")
    print(_progress(settings, ledger))
    if not ledger.exists():
        print("No ledger file yet. Every record stays on its flat stake.")
        print("Copy ledger.example.csv to ledger.csv and replace the examples with bets you place.")
    for status in qualification_report(ledger):
        mean = "n/a" if status.mean_clv is None else f"{status.mean_clv:.3f}"
        print(f"  {status.record}: graded {status.graded}, average CLV {mean}, qualified {status.qualified}")
        print(f"    {status.reason}")
    return 0


def _model(refresh: bool):
    done = completed_games(load_games(refresh=refresh))
    predictions, state = walk_forward(done)
    margin_error = (predictions["home_score"] - predictions["away_score"]) - predictions["mu_margin"]
    total_error = (predictions["home_score"] + predictions["away_score"]) - predictions["mu_total"]
    rho = margin_total_correlation(margin_error.to_numpy(), total_error.to_numpy())
    return state, predictions, rho


def _credits(remaining: str | None, last_cost: str | None) -> None:
    if remaining is None and last_cost is None:
        return
    print(f"The Odds API credits: last request {last_cost or '?'}, remaining {remaining or '?'}.")


def _prop_prices(key: str, max_events: int, until: datetime):
    odds = fetch_game_odds(key)
    events = []
    now = datetime.now(timezone.utc)
    for event in odds.body:
        commence = datetime.fromisoformat(str(event["commence_time"]).replace("Z", "+00:00"))
        if now < commence <= until:
            events.append(event["id"])
    notes = []
    if len(events) > max_events:
        notes.append(
            f"Prop markets requested for {max_events} of {len(events)} games in this slate. "
            "Raise --max-events to include more. Each game spends Odds API credits."
        )
        events = events[:max_events]
    prices = []
    for event_id in events:
        try:
            payload = fetch_event_props(key, event_id)
        except OddsApiError as exc:
            notes.append(f"Skipped one game after the props feed failed: {exc}")
            continue
        prices.extend(build_side_prices(parse_outcomes(payload.body)))
        _credits(payload.remaining, payload.last_cost)
    if not prices:
        notes.append("No DraftKings prop prices came back. The straight prop board will be empty.")
    return prices, notes


def _game_legs(prices, state, settings: Settings, qualified: dict[str, bool], until: datetime) -> list[ParlayLeg]:
    now = datetime.now(timezone.utc)
    legs: list[ParlayLeg] = []
    for price in prices:
        if price.market not in {"h2h", "spreads", "totals"} or not (now < price.commence <= until):
            continue
        home = franchise_from_name(price.home)
        away = franchise_from_name(price.away)
        if home is None or away is None:
            continue
        forecast = forecast_matchup(state, home, away)
        if forecast is None:
            continue
        decided = pregame_board([price], state, settings, qualified, now=now, until=until)
        both = bool(decided) and signals_agree(decided[0])
        game = f"{price.away} at {price.home}"
        if price.market == "spreads" and price.point is not None:
            betting_home = price.side_name == price.home
            home_point = price.point if betting_home else -price.point
            label = f"{game} {price.side_name} {price.point:g}"
            legs.append(
                game_leg_from_spread(
                    event_id=price.event_id,
                    label=label,
                    both_agree=both,
                    mu_margin=forecast.mu_margin,
                    sigma_margin=forecast.sigma_margin,
                    home_point=home_point,
                    betting_home=betting_home,
                )
            )
        elif price.market == "totals" and price.point is not None and price.side_name.lower() in {"over", "under"}:
            side = price.side_name.lower()
            legs.append(
                game_leg_from_total(
                    event_id=price.event_id,
                    label=f"{game} {side} {price.point:g}",
                    both_agree=both,
                    mu_total=forecast.mu_total,
                    sigma_total=forecast.sigma_total,
                    line=price.point,
                    side=side,
                )
            )
        elif price.market == "h2h" and price.side_name in {price.home, price.away}:
            legs.append(
                game_leg_from_moneyline(
                    event_id=price.event_id,
                    label=f"{game} {price.side_name} ML",
                    both_agree=both,
                    mu_margin=forecast.mu_margin,
                    sigma_margin=forecast.sigma_margin,
                    betting_home=price.side_name == price.home,
                )
            )
    return legs


def _prop_legs(prices, props, settings: Settings, qualified: dict[str, bool], until: datetime) -> list[ParlayLeg]:
    now = datetime.now(timezone.utc)
    legs: list[ParlayLeg] = []
    for price in prices:
        stat = STAT_FOR_MARKET.get(price.market)
        if stat is None or not (now < price.commence <= until) or not price.description or price.point is None and stat != "anytime_td":
            continue
        home = franchise_from_name(price.home)
        away = franchise_from_name(price.away)
        player = props.players.get(normalize_name(price.description))
        if player is None or home is None or away is None:
            continue
        if player.team == home:
            opponent, player_is_home = away, True
        elif player.team == away:
            opponent, player_is_home = home, False
        else:
            continue
        projected = project_stat(props, normalize_name(price.description), stat, opponent)
        if projected is None:
            continue
        side = price.side_name.lower()
        mean, sigma = projected
        if stat == "anytime_td":
            p_win = mean if side in {"yes", "over"} else 1.0 - mean
            mu, sigma = 0.0, 1.0
        else:
            if side not in {"over", "under"} or price.point is None:
                continue
            p_win, _push, _loss = total_outcomes(mean, sigma, price.point, side)
            mu = mean
        rows = prop_board([price], props, settings, qualified, now=now, until=until)
        both = bool(rows) and signals_agree(rows[0])
        rho_total = DEFAULT_PROP_RHO[stat]["total"]
        rho_team = DEFAULT_PROP_RHO[stat]["team_margin"]
        point = "" if price.point is None else f" {price.point:g}"
        legs.append(
            prop_leg(
                event_id=price.event_id,
                label=f"{price.description} {price.side_name}{point}",
                both_agree=both,
                stat=stat,
                side=side,
                line=price.point,
                mu=mu,
                sigma=sigma,
                p_win=p_win,
                rho_total=rho_total,
                rho_team_margin=rho_team,
                player_is_home=player_is_home,
                team_id=player.team,
            )
        )
    return legs
