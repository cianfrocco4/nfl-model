"""Walk-forward team scoring model.

Each team's last 10 games of points scored and points allowed, plus a home
indicator, are fit with ridge regression to the points scored in a game.
Training for a week uses only games played before that week. Margins and
totals are treated as normal when prices are quoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_model.teams import franchise

ROLL = 10
MIN_PERIODS = 4
MIN_TRAIN = 200
LEAGUE_POINTS = 21.7
PRIOR_BETA = np.array([4.3, 0.40, 0.40, 2.5])
RIDGE = 8.0


@dataclass(frozen=True)
class ScoreForecast:
    mu_margin: float
    mu_total: float
    sigma_margin: float
    sigma_total: float
    mu_home: float
    mu_away: float


@dataclass(frozen=True)
class ModelState:
    beta: np.ndarray
    sigma_margin: float
    sigma_total: float
    entering: pd.DataFrame


def _long_team_games(games: pd.DataFrame) -> pd.DataFrame:
    home = games[["game_id", "gameday", "season", "week", "home_team", "home_score", "away_score"]].copy()
    home.columns = ["game_id", "gameday", "season", "week", "team", "pf", "pa"]
    home["is_home"] = 1
    away = games[["game_id", "gameday", "season", "week", "away_team", "away_score", "home_score"]].copy()
    away.columns = ["game_id", "gameday", "season", "week", "team", "pf", "pa"]
    away["is_home"] = 0
    long = pd.concat([home, away], ignore_index=True)
    long["franchise"] = long["team"].map(lambda value: franchise(str(value)))
    return long.sort_values(["franchise", "gameday", "game_id"])


def _features(completed: pd.DataFrame) -> pd.DataFrame:
    long = _long_team_games(completed)
    grouped = long.groupby("franchise", group_keys=False)
    long["pf_enter"] = grouped["pf"].transform(lambda s: s.shift(1).rolling(ROLL, min_periods=MIN_PERIODS).mean())
    long["pa_enter"] = grouped["pa"].transform(lambda s: s.shift(1).rolling(ROLL, min_periods=MIN_PERIODS).mean())
    long["pf_exit"] = grouped["pf"].transform(lambda s: s.rolling(ROLL, min_periods=MIN_PERIODS).mean())
    long["pa_exit"] = grouped["pa"].transform(lambda s: s.rolling(ROLL, min_periods=MIN_PERIODS).mean())
    home = long[long["is_home"] == 1][
        ["game_id", "franchise", "pf_enter", "pa_enter", "pf_exit", "pa_exit"]
    ].rename(
        columns={
            "franchise": "home_franchise",
            "pf_enter": "home_pf",
            "pa_enter": "home_pa",
            "pf_exit": "home_pf_exit",
            "pa_exit": "home_pa_exit",
        }
    )
    away = long[long["is_home"] == 0][
        ["game_id", "franchise", "pf_enter", "pa_enter", "pf_exit", "pa_exit"]
    ].rename(
        columns={
            "franchise": "away_franchise",
            "pf_enter": "away_pf",
            "pa_enter": "away_pa",
            "pf_exit": "away_pf_exit",
            "pa_exit": "away_pa_exit",
        }
    )
    merged = completed.merge(home, on="game_id", how="left").merge(away, on="game_id", how="left")
    return merged


def _fit_beta(train: pd.DataFrame) -> np.ndarray:
    usable = train.dropna(subset=["home_pf", "home_pa", "away_pf", "away_pa"])
    if len(usable) < MIN_TRAIN:
        return PRIOR_BETA.copy()
    home_x = np.column_stack(
        [
            np.ones(len(usable)),
            usable["home_pf"].to_numpy(),
            usable["away_pa"].to_numpy(),
            np.ones(len(usable)),
        ]
    )
    away_x = np.column_stack(
        [
            np.ones(len(usable)),
            usable["away_pf"].to_numpy(),
            usable["home_pa"].to_numpy(),
            np.zeros(len(usable)),
        ]
    )
    x = np.vstack([home_x, away_x])
    y = np.concatenate([usable["home_score"].to_numpy(), usable["away_score"].to_numpy()])
    xtx = x.T @ x
    penalty = np.eye(x.shape[1]) * RIDGE
    penalty[0, 0] = 0.0
    xty = x.T @ y
    try:
        return np.linalg.solve(xtx + penalty, xty)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(xtx + penalty, xty, rcond=None)[0]


def _predict_points(beta: np.ndarray, pf: float, opp_pa: float, is_home: int) -> float:
    raw = float(beta[0] + beta[1] * pf + beta[2] * opp_pa + beta[3] * is_home)
    return float(min(45.0, max(3.0, raw)))


def forecast_from_features(
    beta: np.ndarray,
    sigma_margin: float,
    sigma_total: float,
    home_pf: float,
    home_pa: float,
    away_pf: float,
    away_pa: float,
) -> ScoreForecast:
    mu_home = _predict_points(beta, home_pf, away_pa, 1)
    mu_away = _predict_points(beta, away_pf, home_pa, 0)
    return ScoreForecast(
        mu_margin=mu_home - mu_away,
        mu_total=mu_home + mu_away,
        sigma_margin=sigma_margin,
        sigma_total=sigma_total,
        mu_home=mu_home,
        mu_away=mu_away,
    )


def walk_forward(completed: pd.DataFrame) -> tuple[pd.DataFrame, ModelState]:
    """Predict every completed game from ratings fit on earlier games only."""
    featured = _features(completed).sort_values(["gameday", "game_id"]).reset_index(drop=True)
    weeks = (
        featured[["season", "week", "gameday"]]
        .groupby(["season", "week"], as_index=False)["gameday"]
        .min()
        .sort_values("gameday")
    )
    rows: list[dict] = []
    margin_errors: list[float] = []
    total_errors: list[float] = []
    beta = PRIOR_BETA.copy()
    for record in weeks.itertuples(index=False):
        slate = featured[(featured["season"] == record.season) & (featured["week"] == record.week)]
        train = featured[featured["gameday"] < record.gameday]
        beta = _fit_beta(train)
        if len(margin_errors) >= 80:
            sigma_margin = float(np.std(margin_errors[-400:], ddof=1))
            sigma_total = float(np.std(total_errors[-400:], ddof=1))
        else:
            sigma_margin, sigma_total = 13.5, 12.5
        sigma_margin = max(sigma_margin, 8.0)
        sigma_total = max(sigma_total, 8.0)
        for game in slate.itertuples(index=False):
            home_pf = game.home_pf if pd.notna(game.home_pf) else LEAGUE_POINTS
            home_pa = game.home_pa if pd.notna(game.home_pa) else LEAGUE_POINTS
            away_pf = game.away_pf if pd.notna(game.away_pf) else LEAGUE_POINTS
            away_pa = game.away_pa if pd.notna(game.away_pa) else LEAGUE_POINTS
            forecast = forecast_from_features(beta, sigma_margin, sigma_total, home_pf, home_pa, away_pf, away_pa)
            actual_margin = float(game.home_score - game.away_score)
            actual_total = float(game.home_score + game.away_score)
            margin_errors.append(actual_margin - forecast.mu_margin)
            total_errors.append(actual_total - forecast.mu_total)
            rows.append(
                {
                    "game_id": game.game_id,
                    "season": int(game.season),
                    "week": int(game.week),
                    "gameday": game.gameday,
                    "home_team": game.home_team,
                    "away_team": game.away_team,
                    "home_franchise": game.home_franchise,
                    "away_franchise": game.away_franchise,
                    "home_score": float(game.home_score),
                    "away_score": float(game.away_score),
                    "mu_margin": forecast.mu_margin,
                    "mu_total": forecast.mu_total,
                    "sigma_margin": forecast.sigma_margin,
                    "sigma_total": forecast.sigma_total,
                    "spread_line": game.spread_line,
                    "total_line": game.total_line,
                    "home_spread_odds": game.home_spread_odds,
                    "away_spread_odds": game.away_spread_odds,
                    "over_odds": game.over_odds,
                    "under_odds": game.under_odds,
                    "home_moneyline": game.home_moneyline,
                    "away_moneyline": game.away_moneyline,
                }
            )
    predictions = pd.DataFrame(rows)
    entering = _latest_ratings(featured)
    beta = _fit_beta(featured)
    if len(margin_errors) >= 80:
        sigma_margin = float(max(np.std(margin_errors[-400:], ddof=1), 8.0))
        sigma_total = float(max(np.std(total_errors[-400:], ddof=1), 8.0))
    else:
        sigma_margin, sigma_total = 13.5, 12.5
    state = ModelState(beta=beta, sigma_margin=sigma_margin, sigma_total=sigma_total, entering=entering)
    return predictions, state


def _latest_ratings(featured: pd.DataFrame) -> pd.DataFrame:
    home = featured[["gameday", "home_franchise", "home_pf_exit", "home_pa_exit"]].rename(
        columns={"home_franchise": "franchise", "home_pf_exit": "pf", "home_pa_exit": "pa"}
    )
    away = featured[["gameday", "away_franchise", "away_pf_exit", "away_pa_exit"]].rename(
        columns={"away_franchise": "franchise", "away_pf_exit": "pf", "away_pa_exit": "pa"}
    )
    long = pd.concat([home, away], ignore_index=True).dropna(subset=["pf", "pa"])
    long = long.sort_values("gameday").groupby("franchise", as_index=False).tail(1)
    return long.reset_index(drop=True)


def forecast_matchup(state: ModelState, home_franchise: str, away_franchise: str) -> ScoreForecast | None:
    ratings = state.entering.set_index("franchise")
    if home_franchise not in ratings.index or away_franchise not in ratings.index:
        return None
    home = ratings.loc[home_franchise]
    away = ratings.loc[away_franchise]
    return forecast_from_features(
        state.beta,
        state.sigma_margin,
        state.sigma_total,
        float(home["pf"]),
        float(home["pa"]),
        float(away["pf"]),
        float(away["pa"]),
    )


def nflverse_home_spread(spread_line: float) -> float:
    """Convert an nflverse closing spread to a DraftKings-style home point.

    nflverse `spread_line` is positive when the home team is favored.
    DraftKings attaches a negative number to the favorite.
    """
    return -float(spread_line)
