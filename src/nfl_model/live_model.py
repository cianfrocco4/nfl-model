"""In-game spread, total, and moneyline from the pregame forecast and the score.

The elapsed fraction blends wall-clock time since kickoff with the share of
the pregame total already scored. It is not the official game clock. Possession,
down, distance, and timeouts are not in the feed, so they are not used.
"""

from __future__ import annotations

from datetime import datetime

from nfl_model.team_model import ScoreForecast

# A typical NFL broadcast runs about this many minutes from kickoff to the final whistle.
GAME_WALL_MINUTES = 190.0


def elapsed_fraction(
    commence: datetime,
    now: datetime,
    home_score: int,
    away_score: int,
    pregame_total: float,
) -> float:
    elapsed_minutes = (now - commence).total_seconds() / 60.0
    wall = min(0.99, max(0.0, elapsed_minutes / GAME_WALL_MINUTES))
    if pregame_total <= 0:
        return wall
    pace = min(0.99, max(0.0, (home_score + away_score) / pregame_total))
    return min(0.99, 0.5 * wall + 0.5 * pace)


def rescale(pregame: ScoreForecast, home_score: int, away_score: int, fraction: float) -> ScoreForecast:
    """Expected final margin and total given the current score and time elapsed."""
    fraction = min(0.999, max(0.0, fraction))
    remain = 1.0 - fraction
    margin_now = home_score - away_score
    total_now = home_score + away_score
    mu_margin = margin_now + remain * pregame.mu_margin
    mu_total = total_now + remain * pregame.mu_total
    return ScoreForecast(
        mu_margin=mu_margin,
        mu_total=max(mu_total, float(total_now)),
        sigma_margin=max(pregame.sigma_margin * remain**0.5, 0.75),
        sigma_total=max(pregame.sigma_total * remain**0.5, 0.75),
        mu_home=(mu_total + mu_margin) / 2.0,
        mu_away=(mu_total - mu_margin) / 2.0,
    )
