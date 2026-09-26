"""Score-margin and total probabilities from a normal approximation.

NFL scores are integers. A whole-number line can push. A half-point line cannot.
"""

from __future__ import annotations

import math


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _unit(parts: tuple[float, float, float]) -> tuple[float, float, float]:
    cleaned = tuple(min(1.0, max(0.0, p)) for p in parts)
    total = sum(cleaned)
    if total <= 0:
        return (0.0, 0.0, 1.0)
    return tuple(p / total for p in cleaned)  # type: ignore[return-value]


def outcome_split(mu: float, sigma: float, line: float) -> tuple[float, float, float]:
    """Return P(X > line), P(X == line), P(X < line) for an integer-valued X.

    `line` is the push point. Half-points and any non-integer line have no push.
    """
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    integer_line = abs(line - round(line)) <= 1e-6
    if not integer_line:
        if sigma == 0:
            if mu > line:
                return (1.0, 0.0, 0.0)
            if mu < line:
                return (0.0, 0.0, 1.0)
            return (0.0, 0.0, 1.0)
        p_loss = normal_cdf((line - mu) / sigma)
        return _unit((1.0 - p_loss, 0.0, p_loss))

    k = int(round(line))
    if sigma == 0:
        if mu > k:
            return (1.0, 0.0, 0.0)
        if mu < k:
            return (0.0, 0.0, 1.0)
        return (0.0, 1.0, 0.0)
    p_lt = normal_cdf((k - 0.5 - mu) / sigma)
    p_le = normal_cdf((k + 0.5 - mu) / sigma)
    return _unit((1.0 - p_le, p_le - p_lt, p_lt))


def spread_outcomes(
    mu_margin: float,
    sigma: float,
    home_spread: float,
) -> tuple[float, float, float]:
    """Home cover probabilities (win, push, loss).

    `home_spread` is the points DraftKings attaches to the home team.
    Negative means the home team is favored. Home covers when
    (home score - away score) + home_spread > 0.
    """
    push_margin = -home_spread
    win, push, loss = outcome_split(mu_margin, sigma, push_margin)
    return win, push, loss


def total_outcomes(
    mu_total: float,
    sigma: float,
    line: float,
    side: str,
) -> tuple[float, float, float]:
    """Over or under probabilities (win, push, loss)."""
    if side not in {"over", "under"}:
        raise ValueError("side must be 'over' or 'under'")
    over, push, under = outcome_split(mu_total, sigma, line)
    if side == "over":
        return over, push, under
    return under, push, over


def moneyline_outcomes(mu_margin: float, sigma: float) -> tuple[float, float, float]:
    """Home win, tie, away win. A tie is a push on a DraftKings moneyline."""
    home, tie, away = outcome_split(mu_margin, sigma, 0.0)
    return home, tie, away
