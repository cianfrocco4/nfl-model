"""American odds, vig removal, expected value, and edge.

Edge compares a fair win probability with the no-vig price. Expected value uses
the actual decimal payout, because that is what the ticket pays.
"""

from __future__ import annotations


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds (stake included)."""
    if american == 0:
        raise ValueError("American odds of 0 are not a price")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to the nearest American odds integer."""
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1")
    if decimal_odds >= 2.0:
        return int(round((decimal_odds - 1.0) * 100.0))
    return int(round(-100.0 / (decimal_odds - 1.0)))


def implied_probability(american: float) -> float:
    """Raw implied probability, including the book's vig."""
    return 1.0 / american_to_decimal(american)


def remove_vig(implied: list[float]) -> list[float]:
    """Proportionally remove vig so the probabilities sum to 1.

    Each input is a raw implied probability from a price. This is the
    multiplicative method: divide each price by the sum of the prices.
    """
    if len(implied) < 2:
        raise ValueError("Vig removal needs at least two outcomes")
    if any(p <= 0 for p in implied):
        raise ValueError("Implied probabilities must be positive")
    total = sum(implied)
    return [p / total for p in implied]


def fair_decimal(p_win: float, p_push: float = 0.0) -> float:
    """Decimal odds that make the bet zero-EV, treating a push as a refund."""
    if p_win <= 0:
        raise ValueError("Win probability must be positive to quote a fair price")
    if p_push < 0 or p_win + p_push > 1.0 + 1e-9:
        raise ValueError("Win and push probabilities are not a valid split")
    return (1.0 - p_push) / p_win


def fair_american(p_win: float, p_push: float = 0.0) -> int:
    return decimal_to_american(fair_decimal(p_win, p_push))


def expected_value(p_win: float, p_push: float, decimal_odds: float) -> float:
    """Profit per $1 staked. A push returns the stake. A loss returns nothing.

    EV = p_win * decimal_odds + p_push * 1 - 1
    """
    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1")
    p_loss = 1.0 - p_win - p_push
    if p_win < -1e-9 or p_push < -1e-9 or p_loss < -1e-9:
        raise ValueError("Probabilities must be non-negative and sum to 1")
    return p_win * decimal_odds + p_push - 1.0


def decisive_win_probability(p_win: float, p_loss: float) -> float:
    """P(this side wins | the bet does not push)."""
    total = p_win + p_loss
    if total <= 0:
        return 0.0
    return p_win / total


def edge(p_win: float, p_loss: float, novig_implied: float) -> float:
    """Fair decisive win probability minus the no-vig implied probability.

    Both sides of this subtraction ignore pushes, so a whole-number spread is
    compared on the same basis as a two-way price.
    """
    return decisive_win_probability(p_win, p_loss) - novig_implied


def format_american(american: int | None) -> str:
    if american is None:
        return "n/a"
    if american > 0:
        return f"+{american}"
    return str(american)
