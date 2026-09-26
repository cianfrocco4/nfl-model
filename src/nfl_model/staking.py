"""Stake sizing. Every method is selectable. None of them places a bet.

Kelly uses the ticket's decimal payout and the fair win/loss probabilities.
A zero or negative Kelly fraction stakes 0. Flat and units stake their fixed
size only when the caller has already decided the bet clears the edge rule.
Every positive stake is capped at `max_fraction` of the bankroll it is sized
against (the shared bankroll, or one sleeve).
"""

from __future__ import annotations

from nfl_model.settings import StakingMethod


def kelly_fraction(p_win: float, p_loss: float, decimal_odds: float) -> float:
    """Full Kelly fraction of bankroll.

    f* = (p_win * b - p_loss) / (b * (p_win + p_loss)), where b = decimal - 1.
    Pushes do not grow or shrink the bankroll, so they drop out of the
    denominator's probability mass. Returns a negative number when the bet is
    -EV; callers stake 0 in that case.
    """
    b = decimal_odds - 1.0
    mass = p_win + p_loss
    if b <= 0 or mass <= 0:
        return 0.0
    return (p_win * b - p_loss) / (b * mass)


def kelly_multiplier(method: StakingMethod) -> float | None:
    if method is StakingMethod.FULL:
        return 1.0
    if method is StakingMethod.HALF:
        return 0.5
    if method is StakingMethod.QUARTER:
        return 0.25
    return None


def stake_dollars(
    method: StakingMethod,
    bankroll: float,
    *,
    take: bool,
    p_win: float,
    p_loss: float,
    decimal_odds: float,
    flat_fraction: float,
    unit_size: float | None,
    max_fraction: float,
) -> float:
    """Dollar stake for one bet, or 0 when the bet is declined or has no edge."""
    if method is StakingMethod.POLICY:
        raise ValueError("Call method_for_policy before stake_dollars")
    if not take or bankroll <= 0:
        return 0.0
    multiplier = kelly_multiplier(method)
    if multiplier is None and method is StakingMethod.FLAT:
        fraction = flat_fraction
    elif multiplier is None and method is StakingMethod.UNITS:
        unit = flat_fraction * bankroll if unit_size is None else unit_size
        fraction = unit / bankroll
    elif multiplier is not None:
        fraction = multiplier * max(0.0, kelly_fraction(p_win, p_loss, decimal_odds))
    else:
        raise ValueError(f"Unknown staking method {method}")
    fraction = min(max(fraction, 0.0), max_fraction)
    if fraction <= 0:
        return 0.0
    return round(fraction * bankroll, 2)


def method_for_policy(record: str, qualified: bool, selected: StakingMethod) -> StakingMethod:
    """Resolve the locked staking policy, or pass through an explicit override.

    `record` is sides, totals, props, live, or parlays. Live never upgrades.
    Parlays never upgrade, including when another staking method is selected.
    """
    if record == "parlays":
        return StakingMethod.FLAT
    if selected is not StakingMethod.POLICY:
        return selected
    if record == "live" or not qualified:
        return StakingMethod.FLAT
    return StakingMethod.QUARTER


def fit_stakes_to_cash(stakes: list[float], cash: float) -> list[float]:
    """Scale a simultaneous slate so the dollars reserved do not exceed cash.

    This is a cash constraint, not a staking method. Bets on the same slate
    are not settled yet, so they all draw from the same dollars.
    """
    if cash <= 0:
        return [0.0 for _ in stakes]
    total = sum(s for s in stakes if s > 0)
    if total <= cash or total <= 0:
        return [round(max(s, 0.0), 2) for s in stakes]
    scale = cash / total
    return [round(max(s, 0.0) * scale, 2) for s in stakes]
