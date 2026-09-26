"""Straight-bet recommendations.

A stake is suggested only when the selected fair-price rule says so. The locked
default is both: the model and the market must each be +EV on the same side.
If only one is +EV, the row is a lean and the stake is $0.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from nfl_model.live_model import elapsed_fraction, rescale
from nfl_model.odds_api import ScoreState, SidePrice
from nfl_model.odds_math import american_to_decimal, edge, expected_value, fair_american, format_american
from nfl_model.probabilities import moneyline_outcomes, spread_outcomes, total_outcomes
from nfl_model.props_model import MARKET_FOR_STAT, PlayerProjection, PropState, normalize_name, project_stat
from nfl_model.qualification import pool_name
from nfl_model.settings import FairPriceSource, Settings
from nfl_model.staking import fit_stakes_to_cash, method_for_policy, stake_dollars
from nfl_model.team_model import ModelState, ScoreForecast, forecast_matchup
from nfl_model.teams import franchise_from_name

STAT_FOR_MARKET = {market: stat for stat, market in MARKET_FOR_STAT.items()}


@dataclass
class Decision:
    game: str
    market: str
    side: str
    record: str
    dk_american: int
    model_probability: float
    model_edge: float
    model_ev: float
    model_plus: bool
    market_probability: float | None
    market_edge: float | None
    market_ev: float | None
    market_plus: bool
    market_source: str
    status: str
    stake: float
    reason: str
    note: str
    fair_american: int | None


def _signal(p_win: float, p_push: float, decimal_odds: float, novig: float, settings: Settings) -> tuple[float, float, bool]:
    p_loss = 1.0 - p_win - p_push
    side_edge = edge(p_win, p_loss, novig)
    side_ev = expected_value(p_win, p_push, decimal_odds)
    plus = side_edge >= settings.min_edge and side_ev >= settings.min_ev
    return side_edge, side_ev, plus


def decide(
    *,
    game: str,
    market: str,
    side: str,
    record: str,
    dk_american: int,
    dk_novig: float,
    p_win: float,
    p_push: float,
    market_probability: float | None,
    market_source: str,
    note: str,
    settings: Settings,
    qualified: bool,
) -> Decision:
    decimal_odds = american_to_decimal(dk_american)
    p_loss = 1.0 - p_win - p_push
    model_edge, model_ev, model_plus = _signal(p_win, p_push, decimal_odds, dk_novig, settings)
    if market_probability is None:
        market_edge = market_ev = None
        market_plus = False
    else:
        market_edge, market_ev, market_plus = _signal(market_probability, 0.0, decimal_odds, dk_novig, settings)

    take = False
    reason = "pass: neither signal clears the edge threshold"
    stake_win, stake_loss = p_win, p_loss
    source = settings.fair_price
    if source is FairPriceSource.MODEL:
        take = model_plus
        reason = "model signal" if take else "pass: model signal is below the threshold"
    elif source is FairPriceSource.MARKET:
        take = market_plus
        if market_probability is None:
            reason = "pass: market price is missing"
        else:
            stake_win, stake_loss = market_probability, 1.0 - market_probability
            reason = "market signal" if take else "pass: market signal is below the threshold"
    elif model_plus and market_plus and market_probability is not None:
        decisive = p_win / (p_win + p_loss) if p_win + p_loss else 0.0
        used = min(decisive, market_probability)
        mass = p_win + p_loss
        stake_win, stake_loss = used * mass, (1.0 - used) * mass
        conservative_ev = expected_value(stake_win, p_push, decimal_odds)
        if settings.staking.value == "policy" and method_for_policy(record, qualified, settings.staking).value == "flat":
            take = True
            reason = "stake: model and market are both +EV"
        elif conservative_ev >= settings.min_ev:
            take = True
            reason = "stake: model and market are both +EV"
        else:
            take = False
            reason = "lean: both signals clear on their own, but the smaller probability is not +EV at this price"
    elif model_plus or market_plus:
        which = "model" if model_plus else "market"
        other = "market" if model_plus else "model"
        if market_probability is None and model_plus:
            reason = "lean: model is +EV and the market price is missing, so there is no stake"
        else:
            reason = f"lean: only the {which} is +EV; the {other} does not agree, so there is no stake"
    try:
        quoted = fair_american(p_win, p_push)
    except ValueError:
        quoted = None

    method = method_for_policy(record, qualified, settings.staking)
    dollars = stake_dollars(
        method,
        settings.pool(pool_name(record)),
        take=take,
        p_win=stake_win,
        p_loss=stake_loss,
        decimal_odds=decimal_odds,
        flat_fraction=settings.flat_fraction,
        unit_size=settings.unit_size,
        max_fraction=settings.max_stake_fraction,
    )
    if take and dollars <= 0:
        reason = "signals agree, but the staking rule sizes this bet at $0"
    status = "stake" if dollars > 0 else ("lean" if reason.startswith("lean") else "pass")
    if dollars > 0:
        status = "stake"
    return Decision(
        game=game,
        market=market,
        side=side,
        record=record,
        dk_american=dk_american,
        model_probability=p_win,
        model_edge=model_edge,
        model_ev=model_ev,
        model_plus=model_plus,
        market_probability=market_probability,
        market_edge=market_edge,
        market_ev=market_ev,
        market_plus=market_plus,
        market_source=market_source,
        status=status,
        stake=dollars,
        reason=reason,
        note=note,
        fair_american=quoted,
    )


def apply_cash_limit(rows: list[Decision], settings: Settings) -> list[Decision]:
    by_pool: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_pool.setdefault(pool_name(row.record), []).append(index)
    updated = list(rows)
    for pool, indexes in by_pool.items():
        cash = settings.pool(pool)
        sized = fit_stakes_to_cash([updated[index].stake for index in indexes], cash)
        if sized == [updated[index].stake for index in indexes]:
            continue
        for index, stake in zip(indexes, sized):
            row = updated[index]
            reason = row.reason
            if stake < row.stake:
                reason += " Stake scaled so this slate fits in the sleeve."
            updated[index] = Decision(
                game=row.game,
                market=row.market,
                side=row.side,
                record=row.record,
                dk_american=row.dk_american,
                model_probability=row.model_probability,
                model_edge=row.model_edge,
                model_ev=row.model_ev,
                model_plus=row.model_plus,
                market_probability=row.market_probability,
                market_edge=row.market_edge,
                market_ev=row.market_ev,
                market_plus=row.market_plus,
                market_source=row.market_source,
                status="stake" if stake > 0 else row.status,
                stake=stake,
                reason=reason,
                note=row.note,
                fair_american=row.fair_american,
            )
    return updated


def pregame_board(prices: list[SidePrice], state: ModelState, settings: Settings, qualified: dict[str, bool], now: datetime | None = None) -> list[Decision]:
    clock = now or datetime.now(timezone.utc)
    rows: list[Decision] = []
    for price in prices:
        if price.market not in {"h2h", "spreads", "totals"}:
            continue
        if price.commence <= clock:
            continue
        forecast = _forecast_for_price(price, state)
        if forecast is None:
            continue
        outcomes = _game_outcomes(price, forecast)
        if outcomes is None:
            continue
        p_win, p_push, record = outcomes
        rows.append(
            decide(
                game=f"{price.away} at {price.home}",
                market=price.market,
                side=_side_label(price),
                record=record,
                dk_american=price.dk_american,
                dk_novig=price.dk_novig,
                p_win=p_win,
                p_push=p_push,
                market_probability=price.market_novig,
                market_source=price.market_source,
                note=price.note,
                settings=settings,
                qualified=qualified.get(record, False),
            )
        )
    return apply_cash_limit(rows, settings)


def live_board(
    prices: list[SidePrice],
    scores: list[ScoreState],
    state: ModelState,
    settings: Settings,
    qualified: dict[str, bool],
    now: datetime | None = None,
) -> tuple[list[Decision], list[str]]:
    clock = now or datetime.now(timezone.utc)
    by_id = {score.event_id: score for score in scores}
    notes: list[str] = []
    prop_events = {price.event_id for price in prices if price.market.startswith("player_")}
    if prop_events:
        notes.append(
            "DraftKings returned live player props. They are not recommended. "
            "The scores feed has the team score only, not a player's yards or touchdowns, "
            "so a live prop cannot be rescored."
        )
    rows: list[Decision] = []
    seen_live: set[str] = set()
    for price in prices:
        if price.market not in {"h2h", "spreads", "totals"}:
            continue
        score = by_id.get(price.event_id)
        if score is None or score.completed or score.home_score is None or score.away_score is None:
            continue
        if price.commence > clock:
            continue
        seen_live.add(price.event_id)
        pregame = _forecast_for_price(price, state)
        if pregame is None:
            continue
        fraction = elapsed_fraction(price.commence, clock, score.home_score, score.away_score, pregame.mu_total)
        forecast = rescale(pregame, score.home_score, score.away_score, fraction)
        outcomes = _game_outcomes(price, forecast)
        if outcomes is None:
            continue
        p_win, p_push, _record = outcomes
        note = (
            f"Score {score.away_score}-{score.home_score} (away-home). "
            f"Estimated {fraction:.0%} elapsed from the wall clock and the scoring pace, not the game clock. "
            + price.note
        )
        rows.append(
            decide(
                game=f"{price.away} at {price.home}",
                market=price.market,
                side=_side_label(price),
                record="live",
                dk_american=price.dk_american,
                dk_novig=price.dk_novig,
                p_win=p_win,
                p_push=p_push,
                market_probability=price.market_novig,
                market_source=price.market_source,
                note=note,
                settings=settings,
                qualified=False,
            )
        )
    if not seen_live:
        notes.append("No NFL game with a DraftKings line and a score is in play right now.")
    return apply_cash_limit(rows, settings), notes


def prop_board(prices: list[SidePrice], props: PropState, settings: Settings, qualified: dict[str, bool], now: datetime | None = None) -> list[Decision]:
    clock = now or datetime.now(timezone.utc)
    rows: list[Decision] = []
    for price in prices:
        stat = STAT_FOR_MARKET.get(price.market)
        if stat is None or price.commence <= clock or not price.description:
            continue
        home = franchise_from_name(price.home)
        away = franchise_from_name(price.away)
        player = props.players.get(normalize_name(price.description))
        if player is None or home is None or away is None:
            continue
        if player.team == home:
            opponent = away
        elif player.team == away:
            opponent = home
        else:
            continue
        projected = project_stat(props, normalize_name(price.description), stat, opponent)
        if projected is None:
            continue
        outcomes = _prop_outcomes(stat, price, projected[0], projected[1])
        if outcomes is None:
            continue
        p_win, p_push = outcomes
        rows.append(
            decide(
                game=f"{price.description} ({price.away} at {price.home})",
                market=price.market,
                side=_side_label(price),
                record="props",
                dk_american=price.dk_american,
                dk_novig=price.dk_novig,
                p_win=p_win,
                p_push=p_push,
                market_probability=price.market_novig,
                market_source=price.market_source,
                note=price.note,
                settings=settings,
                qualified=qualified.get("props", False),
            )
        )
    return apply_cash_limit(rows, settings)


def render(rows: list[Decision], heading: str) -> str:
    stakes = [row for row in rows if row.stake > 0]
    leans = [row for row in rows if row.status == "lean"]
    lines = [heading, ""]
    if stakes:
        lines.append("Suggested stakes (you place these yourself):")
        for row in stakes:
            lines.append(
                f"  ${row.stake:,.2f}  {row.game}  {row.side}  DK {format_american(row.dk_american)}  {row.reason}"
            )
    else:
        lines.append("Suggested stakes: none.")
    lines.append("")
    if leans:
        lines.append("Leans with no stake:")
        for row in leans:
            lines.append(f"  {row.game}  {row.side}  DK {format_american(row.dk_american)}  {row.reason}")
        lines.append("")
    lines.append("Every signal:")
    if not rows:
        lines.append("  No markets to show.")
    for row in rows:
        market = "n/a" if row.market_probability is None else f"{row.market_probability:.1%} ({row.market_source})"
        market_edge = "n/a" if row.market_edge is None else f"{row.market_edge:+.1%}"
        lines.append(
            f"  {row.game} | {row.side} | DK {format_american(row.dk_american)} | "
            f"model {row.model_probability:.1%} edge {row.model_edge:+.1%} EV {row.model_ev:+.1%} | "
            f"market {market} edge {market_edge} | {row.status} ${row.stake:,.2f}"
        )
        if row.note:
            lines.append(f"    {row.note}")
    return "\n".join(lines)


def _forecast_for_price(price: SidePrice, state: ModelState) -> ScoreForecast | None:
    home = franchise_from_name(price.home)
    away = franchise_from_name(price.away)
    if home is None or away is None:
        return None
    return forecast_matchup(state, home, away)


def _game_outcomes(price: SidePrice, forecast: ScoreForecast) -> tuple[float, float, str] | None:
    if price.market == "spreads" and price.point is not None:
        home_point = price.point if price.side_name == price.home else -price.point
        home_win, push, home_loss = spread_outcomes(forecast.mu_margin, forecast.sigma_margin, home_point)
        if price.side_name == price.home:
            return home_win, push, "sides"
        return home_loss, push, "sides"
    if price.market == "totals" and price.point is not None:
        side = price.side_name.lower()
        if side not in {"over", "under"}:
            return None
        win, push, _loss = total_outcomes(forecast.mu_total, forecast.sigma_total, price.point, side)
        return win, push, "totals"
    if price.market == "h2h":
        home_win, tie, away_win = moneyline_outcomes(forecast.mu_margin, forecast.sigma_margin)
        if price.side_name == price.home:
            return home_win, tie, "sides"
        if price.side_name == price.away:
            return away_win, tie, "sides"
    return None


def _prop_outcomes(stat: str, price: SidePrice, projected: float, sigma: float) -> tuple[float, float] | None:
    name = price.side_name.lower()
    if stat == "anytime_td":
        probability = min(0.99, max(0.0, projected))
        if name in {"yes", "over"}:
            return probability, 0.0
        if name in {"no", "under"}:
            return 1.0 - probability, 0.0
        return None
    if price.point is None or name not in {"over", "under"}:
        return None
    win, push, _loss = total_outcomes(projected, sigma, price.point, name)
    return win, push


def _side_label(price: SidePrice) -> str:
    point = "" if price.point is None else f" {price.point:g}"
    who = f"{price.description} " if price.description else ""
    return f"{who}{price.side_name}{point}".strip()


def qualified_map(statuses: list) -> dict[str, bool]:
    return {status.record: status.qualified for status in statuses}
