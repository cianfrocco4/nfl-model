"""2-leg and 3-leg parlay prices.

Cross-game legs multiply. Same-game spread, total, and moneyline legs use the
joint normal of margin and total. Same-game props use a correlation-adjusted
draw. A ticket is staked only when every leg that has a straight price passes
the both-agree rule and the DraftKings parlay price is +EV against that joint
probability. The stake is flat 1% of the parlay sleeve. Parlays do not upgrade
to quarter Kelly.

The Odds API does not publish DraftKings parlay or same-game parlay prices.
The price has to be the number on the DraftKings slip.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from nfl_model.odds_math import american_to_decimal, expected_value, fair_american, format_american, implied_probability
from nfl_model.probabilities import moneyline_outcomes, normal_cdf, spread_outcomes, total_outcomes
from nfl_model.settings import Settings
from nfl_model.staking import method_for_policy, stake_dollars

# Shrunk toward these when a stat has little history. Team margin is from the
# player's own team, not the home team. Pair correlations below are defaults.
DEFAULT_PROP_RHO = {
    "passing_yards": {"total": 0.22, "team_margin": 0.12},
    "rushing_yards": {"total": 0.08, "team_margin": 0.18},
    "receiving_yards": {"total": 0.18, "team_margin": 0.10},
    "receptions": {"total": 0.12, "team_margin": 0.06},
    "anytime_td": {"total": 0.16, "team_margin": 0.14},
}
_PAIR_RHO = {
    ("passing_yards", "receiving_yards"): 0.35,
    ("passing_yards", "receptions"): 0.25,
    ("receiving_yards", "receptions"): 0.55,
    ("passing_yards", "rushing_yards"): -0.12,
    ("rushing_yards", "receiving_yards"): -0.08,
    ("anytime_td", "receiving_yards"): 0.25,
    ("anytime_td", "rushing_yards"): 0.22,
    ("anytime_td", "passing_yards"): 0.05,
    ("anytime_td", "receptions"): 0.20,
}
MC_DRAWS = 20000
_BOUND = 1e6


@dataclass(frozen=True)
class ParlayLeg:
    """One straight leg that can sit inside a 2- or 3-leg ticket."""

    key: str
    event_id: str
    label: str
    both_agree: bool
    has_straight_price: bool
    latent: str
    low: float
    high: float
    mu: float
    sigma: float
    p_win: float
    rho_margin: float
    rho_total: float
    prop_stat: str = ""
    team_id: str = ""


@dataclass(frozen=True)
class ParlayTicket:
    legs: tuple[ParlayLeg, ...]
    kind: str
    joint: float
    independent: float
    dk_american: int | None
    edge: float | None
    ev: float | None
    status: str
    stake: float
    reason: str

    @property
    def fair_price(self) -> str:
        if self.joint <= 0.001 or self.joint >= 0.999:
            return "n/a"
        try:
            return format_american(fair_american(self.joint))
        except ValueError:
            return "n/a"


def bivariate_cdf(h: float, k: float, rho: float, terms: int = 40) -> float:
    """Standard bivariate normal CDF via the Hermite series."""
    rho = max(-0.95, min(0.95, rho))
    hes_h = _hermite(h, terms)
    hes_k = _hermite(k, terms)
    weight = 1.0
    series = 0.0
    for n in range(1, terms + 1):
        weight *= rho / n
        series += weight * hes_h[n - 1] * hes_k[n - 1]
    density = math.exp(-0.5 * (h * h + k * k)) / (2.0 * math.pi)
    value = normal_cdf(h) * normal_cdf(k) + density * series
    return min(1.0, max(0.0, value))


def _hermite(x: float, count: int) -> list[float]:
    values = [1.0]
    if count == 1:
        return values
    values.append(x)
    for m in range(1, count - 1):
        values.append(x * values[m] - m * values[m - 1])
    return values


def _inv_norm(probability: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, probability))
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def continuity_cut(line: float, higher_wins: bool) -> float:
    """Threshold on the latent where the continuity-corrected win region starts."""
    if abs(line - round(line)) <= 1e-6:
        whole = float(round(line))
        return whole + 0.5 if higher_wins else whole - 0.5
    return line


def margin_interval(home_point: float, betting_home: bool) -> tuple[float, float]:
    """Home-margin interval that wins this spread or, with point 0, this moneyline."""
    push_at = -home_point
    if betting_home:
        return continuity_cut(push_at, higher_wins=True), _BOUND
    return -_BOUND, continuity_cut(push_at, higher_wins=False)


def total_interval(line: float, side: str) -> tuple[float, float]:
    if side == "over":
        return continuity_cut(line, higher_wins=True), _BOUND
    return -_BOUND, continuity_cut(line, higher_wins=False)


def joint_probability(legs: list[ParlayLeg], rho_margin_total: float, draws: int = MC_DRAWS, seed: int = 0) -> float:
    """Probability every leg wins. Independent across games."""
    if not legs:
        return 0.0
    grouped: dict[str, list[ParlayLeg]] = {}
    for leg in legs:
        grouped.setdefault(leg.event_id, []).append(leg)
    probability = 1.0
    for event_legs in grouped.values():
        probability *= _event_joint(event_legs, rho_margin_total, draws, seed)
    return min(1.0, max(0.0, probability))


def independent_probability(legs: list[ParlayLeg]) -> float:
    value = 1.0
    for leg in legs:
        value *= leg.p_win
    return value


def _event_joint(legs: list[ParlayLeg], rho_margin_total: float, draws: int, seed: int) -> float:
    if len(legs) == 1:
        return _region_probability(legs[0])
    if all(leg.latent in {"margin", "total"} for leg in legs):
        return _game_region(legs, rho_margin_total)
    return _monte_carlo(legs, rho_margin_total, draws, seed)


def _region_probability(leg: ParlayLeg) -> float:
    return _interval(leg.mu, leg.sigma, leg.low, leg.high)


def _interval(mu: float, sigma: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    if sigma <= 0:
        return 1.0 if low < mu < high else 0.0
    upper = 1.0 if high >= _BOUND / 2 else normal_cdf((high - mu) / sigma)
    lower = 0.0 if low <= -_BOUND / 2 else normal_cdf((low - mu) / sigma)
    return max(0.0, upper - lower)


def _game_region(legs: list[ParlayLeg], rho: float) -> float:
    margin = [leg for leg in legs if leg.latent == "margin"]
    total = [leg for leg in legs if leg.latent == "total"]
    margin_box = _intersect(margin)
    total_box = _intersect(total)
    if margin and margin_box is None:
        return 0.0
    if total and total_box is None:
        return 0.0
    if margin and not total:
        mu, sigma = margin[0].mu, margin[0].sigma
        return _interval(mu, sigma, margin_box[0], margin_box[1])
    if total and not margin:
        mu, sigma = total[0].mu, total[0].sigma
        return _interval(mu, sigma, total_box[0], total_box[1])
    return _rectangle(
        margin[0].mu,
        margin[0].sigma,
        total[0].mu,
        total[0].sigma,
        rho,
        margin_box[0],
        margin_box[1],
        total_box[0],
        total_box[1],
    )


def _intersect(legs: list[ParlayLeg]) -> tuple[float, float] | None:
    if not legs:
        return None
    low = max(leg.low for leg in legs)
    high = min(leg.high for leg in legs)
    if high <= low:
        return None
    return low, high


def _rectangle(
    mu_m: float,
    sig_m: float,
    mu_t: float,
    sig_t: float,
    rho: float,
    low_m: float,
    high_m: float,
    low_t: float,
    high_t: float,
) -> float:
    def z(point: float, mu: float, sigma: float) -> float:
        if point >= _BOUND / 2:
            return 8.0
        if point <= -_BOUND / 2:
            return -8.0
        return (point - mu) / sigma

    a, b = z(low_m, mu_m, sig_m), z(high_m, mu_m, sig_m)
    c, d = z(low_t, mu_t, sig_t), z(high_t, mu_t, sig_t)

    def f(x: float, y: float) -> float:
        return bivariate_cdf(x, y, rho)

    return max(0.0, f(b, d) - f(a, d) - f(b, c) + f(a, c))


def _monte_carlo(legs: list[ParlayLeg], rho_margin_total: float, draws: int, seed: int) -> float:
    names = ["margin", "total"]
    prop_legs = [leg for leg in legs if leg.latent == "prop"]
    for index, _leg in enumerate(prop_legs):
        names.append(f"prop{index}")
    size = len(names)
    corr = np.eye(size)
    corr[0, 1] = corr[1, 0] = rho_margin_total
    for index, leg in enumerate(prop_legs):
        corr[0, 2 + index] = corr[2 + index, 0] = leg.rho_margin
        corr[1, 2 + index] = corr[2 + index, 1] = leg.rho_total
    for left, right in itertools.combinations(range(len(prop_legs)), 2):
        rho = _prop_pair(prop_legs[left], prop_legs[right])
        corr[2 + left, 2 + right] = corr[2 + right, 2 + left] = rho
    corr = _make_correlation(corr)
    draws_z = np.random.default_rng(seed).multivariate_normal(np.zeros(size), corr, size=draws)
    margin_legs = [leg for leg in legs if leg.latent == "margin"]
    total_legs = [leg for leg in legs if leg.latent == "total"]
    wins = np.ones(draws, dtype=bool)
    if margin_legs:
        margin = margin_legs[0].mu + margin_legs[0].sigma * draws_z[:, 0]
        for leg in margin_legs:
            wins &= (margin > leg.low) & (margin < leg.high)
    if total_legs:
        total = total_legs[0].mu + total_legs[0].sigma * draws_z[:, 1]
        for leg in total_legs:
            wins &= (total > leg.low) & (total < leg.high)
    for index, leg in enumerate(prop_legs):
        value = leg.mu + leg.sigma * draws_z[:, 2 + index]
        wins &= (value > leg.low) & (value < leg.high)
    return float(wins.mean())


def _prop_pair(left: ParlayLeg, right: ParlayLeg) -> float:
    same_team = bool(left.team_id) and left.team_id == right.team_id
    if not same_team:
        return -0.05
    key = tuple(sorted((left.prop_stat, right.prop_stat)))
    return _PAIR_RHO.get(key, 0.05)


def _make_correlation(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    values = np.clip(values, 1e-6, None)
    fixed = vectors @ np.diag(values) @ vectors.T
    scale = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(scale, scale)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def price_ticket(
    legs: list[ParlayLeg],
    dk_american: int | None,
    settings: Settings,
    rho_margin_total: float,
    draws: int = MC_DRAWS,
) -> ParlayTicket:
    kind = _kind(legs)
    independent = independent_probability(legs) if legs else 0.0
    if len(legs) not in {2, 3}:
        return _ticket(legs, kind, 0.0, independent, dk_american, None, None, "pass", 0.0, "A parlay must be 2 or 3 legs.")
    joint = joint_probability(legs, rho_margin_total, draws=draws)
    missing = [leg for leg in legs if not leg.has_straight_price]
    rejected = [leg for leg in legs if leg.has_straight_price and not leg.both_agree]
    if missing or rejected:
        reason = "lean: a leg with a straight price did not pass both-agree, so there is no stake"
        if missing:
            reason = "lean: a leg has no straight price to check, so there is no stake"
        return _ticket(legs, kind, joint, independent, dk_american, None, None, "lean", 0.0, reason)
    if dk_american is None:
        return _ticket(
            legs,
            kind,
            joint,
            independent,
            None,
            None,
            None,
            "pass",
            0.0,
            "No stake yet. The Odds API does not publish the DraftKings parlay price; pass the price from the slip.",
        )
    decimal_odds = american_to_decimal(dk_american)
    implied = implied_probability(dk_american)
    ev = expected_value(joint, 0.0, decimal_odds)
    edge = joint - implied
    take = ev >= settings.min_ev and edge >= settings.min_edge
    method = method_for_policy("parlays", qualified=True, selected=settings.staking)
    stake = stake_dollars(
        method,
        settings.pool("parlays"),
        take=take,
        p_win=joint,
        p_loss=1.0 - joint,
        decimal_odds=decimal_odds,
        flat_fraction=settings.flat_fraction,
        unit_size=settings.unit_size,
        max_fraction=settings.max_stake_fraction,
    )
    if take and stake > 0:
        reason = "stake: the DraftKings parlay price is +EV versus the fair joint probability"
        status = "stake"
    elif take and stake <= 0:
        reason = "the price is +EV, but the flat stake rounds to $0"
        status = "pass"
    else:
        reason = "pass: the DraftKings parlay price is not +EV versus the fair joint probability"
        status = "pass"
    return _ticket(legs, kind, joint, independent, dk_american, edge, ev, status, stake, reason)


def suggest_tickets(
    legs: list[ParlayLeg],
    settings: Settings,
    rho_margin_total: float,
    limit: int = 24,
    max_legs: int = 12,
) -> list[ParlayTicket]:
    """Fair prices for 2- and 3-leg combinations of legs that already both-agree.

    At most `max_legs` both-agree legs are combined, highest model probability
    first, so a full Sunday slate does not build thousands of tickets.
    """
    eligible = [leg for leg in legs if leg.both_agree and leg.has_straight_price]
    eligible = sorted(eligible, key=lambda leg: leg.p_win, reverse=True)[:max_legs]
    tickets: list[ParlayTicket] = []
    for size in (2, 3):
        for combo in itertools.combinations(eligible, size):
            tickets.append(price_ticket(list(combo), None, settings, rho_margin_total, draws=2000))
    tickets.sort(key=lambda ticket: ticket.joint, reverse=True)
    return tickets[:limit]


def apply_slip(
    legs: list[ParlayLeg],
    slip: str,
    settings: Settings,
    rho_margin_total: float,
) -> ParlayTicket:
    """Parse '1,2:+265' using 1-based indexes into `legs`."""
    try:
        left, right = slip.split(":")
        indexes = [int(part) for part in left.split(",") if part.strip()]
        american = int(right)
    except ValueError as exc:
        raise ValueError(
            "A slip looks like 1,2:+265 or 1,3,4:-120. "
            "The numbers are the leg list printed above, and the price is the DraftKings parlay price."
        ) from exc
    if any(index < 1 or index > len(legs) for index in indexes):
        raise ValueError(f"Slip {slip} points at a leg number outside 1..{len(legs)}.")
    chosen = [legs[index - 1] for index in indexes]
    return price_ticket(chosen, american, settings, rho_margin_total)


def render_parlays(
    legs: list[ParlayLeg],
    suggestions: list[ParlayTicket],
    priced: list[ParlayTicket],
    *,
    verbose: bool = False,
) -> str:
    lines = [
        "Parlays are 2 or 3 legs. Same-game tickets are correlation-adjusted. Cross-game tickets multiply.",
        "Each leg with a straight price must pass both-agree. The stake is flat 1% of the parlay sleeve.",
        "The Odds API does not publish DraftKings parlay prices. Copy the price from the DraftKings slip.",
        "Example: python -m nfl_model parlays --bankroll 100 --slip 1,2:+265",
        "",
        "Legs that both-agree (numbers are the --slip indexes):",
    ]
    shown = legs if verbose else [leg for leg in legs if leg.both_agree]
    if not shown:
        lines.append("  None. No straight leg passed both-agree." if not verbose else "  None. No straight leg was priced.")
    hidden = 0
    for index, leg in enumerate(legs, start=1):
        if not verbose and not leg.both_agree:
            hidden += 1
            continue
        flag = "both-agree" if leg.both_agree else "not both-agree"
        lines.append(f"  [{index}] {leg.label}  model {leg.p_win:.1%}  {flag}")
    if hidden:
        lines.append(f"{hidden} legs did not both-agree. Pass --verbose to list them.")
    lines.append("")
    lines.append("Fair prices with no DraftKings ticket yet (stake $0):")
    if not suggestions:
        lines.append("  No 2- or 3-leg ticket has every leg both-agree.")
    visible = suggestions if verbose else suggestions[:8]
    for ticket in visible:
        labels = " + ".join(leg.label for leg in ticket.legs)
        lines.append(
            f"  {ticket.kind}  fair {ticket.fair_price}  joint {ticket.joint:.1%}  "
            f"if independent {ticket.independent:.1%}  {labels}"
        )
    if not verbose and len(suggestions) > len(visible):
        lines.append(f"  {len(suggestions) - len(visible)} more tickets. Pass --verbose to list them.")
    if priced:
        lines.append("")
        lines.append("Slips you priced from DraftKings:")
        for ticket in priced:
            price = "n/a" if ticket.dk_american is None else format_american(ticket.dk_american)
            lines.append(
                f"  ${ticket.stake:,.2f}  DK {price}  fair {ticket.fair_price}  "
                f"joint {ticket.joint:.1%}  {ticket.reason}"
            )
    return "\n".join(lines)


def margin_total_correlation(margin_error: np.ndarray, total_error: np.ndarray) -> float:
    if len(margin_error) < 50:
        return 0.0
    rho = float(np.corrcoef(margin_error, total_error)[0, 1])
    if math.isnan(rho):
        return 0.0
    return float(max(-0.8, min(0.8, rho)))


def game_leg_from_spread(
    *,
    event_id: str,
    label: str,
    both_agree: bool,
    mu_margin: float,
    sigma_margin: float,
    home_point: float,
    betting_home: bool,
) -> ParlayLeg:
    low, high = margin_interval(home_point, betting_home)
    home_win, _push, home_loss = spread_outcomes(mu_margin, sigma_margin, home_point)
    return ParlayLeg(
        key=f"{event_id}|spread|{label}",
        event_id=event_id,
        label=label,
        both_agree=both_agree,
        has_straight_price=True,
        latent="margin",
        low=low,
        high=high,
        mu=mu_margin,
        sigma=sigma_margin,
        p_win=home_win if betting_home else home_loss,
        rho_margin=1.0,
        rho_total=0.0,
    )


def game_leg_from_total(
    *,
    event_id: str,
    label: str,
    both_agree: bool,
    mu_total: float,
    sigma_total: float,
    line: float,
    side: str,
) -> ParlayLeg:
    low, high = total_interval(line, side)
    win, _push, _loss = total_outcomes(mu_total, sigma_total, line, side)
    return ParlayLeg(
        key=f"{event_id}|total|{label}",
        event_id=event_id,
        label=label,
        both_agree=both_agree,
        has_straight_price=True,
        latent="total",
        low=low,
        high=high,
        mu=mu_total,
        sigma=sigma_total,
        p_win=win,
        rho_margin=0.0,
        rho_total=1.0,
    )


def game_leg_from_moneyline(
    *,
    event_id: str,
    label: str,
    both_agree: bool,
    mu_margin: float,
    sigma_margin: float,
    betting_home: bool,
) -> ParlayLeg:
    # Moneyline win is the margin past 0, with the same continuity cut as a 0-point spread.
    low, high = margin_interval(0.0, betting_home)
    home_win, _tie, away_win = moneyline_outcomes(mu_margin, sigma_margin)
    return ParlayLeg(
        key=f"{event_id}|ml|{label}",
        event_id=event_id,
        label=label,
        both_agree=both_agree,
        has_straight_price=True,
        latent="margin",
        low=low,
        high=high,
        mu=mu_margin,
        sigma=sigma_margin,
        p_win=home_win if betting_home else away_win,
        rho_margin=1.0,
        rho_total=0.0,
    )


def prop_leg(
    *,
    event_id: str,
    label: str,
    both_agree: bool,
    stat: str,
    side: str,
    line: float | None,
    mu: float,
    sigma: float,
    p_win: float,
    rho_total: float,
    rho_team_margin: float,
    player_is_home: bool,
    team_id: str,
) -> ParlayLeg:
    rho_margin = rho_team_margin if player_is_home else -rho_team_margin
    if stat == "anytime_td":
        # Standardized latent: the side you bet wins when Z is below its win quantile.
        cut = _inv_norm(p_win)
        low, high = -_BOUND, cut
        probability = p_win
        return ParlayLeg(
            key=f"{event_id}|{stat}|{label}",
            event_id=event_id,
            label=label,
            both_agree=both_agree,
            has_straight_price=True,
            latent="prop",
            low=low,
            high=high,
            mu=0.0,
            sigma=1.0,
            p_win=probability,
            rho_margin=rho_margin,
            rho_total=rho_total,
            prop_stat=stat,
            team_id=team_id,
        )
    low, high = total_interval(line if line is not None else 0.0, side)
    return ParlayLeg(
        key=f"{event_id}|{stat}|{label}",
        event_id=event_id,
        label=label,
        both_agree=both_agree,
        has_straight_price=True,
        latent="prop",
        low=low,
        high=high,
        mu=mu,
        sigma=sigma,
        p_win=p_win,
        rho_margin=rho_margin,
        rho_total=rho_total,
        prop_stat=stat,
        team_id=team_id,
    )


def _kind(legs: list[ParlayLeg]) -> str:
    events = {leg.event_id for leg in legs}
    if len(events) <= 1:
        return "same-game"
    if len(events) == len(legs):
        return "cross-game"
    return "mixed"


def _ticket(
    legs: list[ParlayLeg],
    kind: str,
    joint: float,
    independent: float,
    dk_american: int | None,
    edge: float | None,
    ev: float | None,
    status: str,
    stake: float,
    reason: str,
) -> ParlayTicket:
    return ParlayTicket(
        legs=tuple(legs),
        kind=kind,
        joint=joint,
        independent=independent,
        dk_american=dk_american,
        edge=edge,
        ev=ev,
        status=status,
        stake=stake,
        reason=reason,
    )
