"""Pregame player props from recent production, shrunk toward the position.

Passing yards, rushing yards, receiving yards, and receptions use a normal
distribution. Anytime touchdown is a Poisson chance of a rushing, receiving,
return, or fumble-recovery score. Passing touchdowns are not anytime scores.

Historical DraftKings prop prices are not in the open data, so the backtest
reports prediction error only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_model.teams import franchise

STAT_COLUMNS = {
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "anytime_td": "anytime_td",
}
MARKET_FOR_STAT = {
    "passing_yards": "player_pass_yds",
    "rushing_yards": "player_rush_yds",
    "receiving_yards": "player_reception_yds",
    "receptions": "player_receptions",
    "anytime_td": "player_anytime_td",
}
SIGMA_FLOOR = {
    "passing_yards": 45.0,
    "rushing_yards": 18.0,
    "receiving_yards": 20.0,
    "receptions": 1.2,
}
ROLL = 8
MIN_PLAYER_GAMES = 4
SHRINK_GAMES = 6.0


@dataclass(frozen=True)
class PlayerProjection:
    name: str
    team: str
    position: str
    means: dict[str, float]
    sigmas: dict[str, float]


@dataclass(frozen=True)
class PropState:
    players: dict[str, PlayerProjection]
    defense: dict[tuple[str, str], float]
    league: dict[str, float]
    accuracy: dict[str, float]


def normalize_name(name: str) -> str:
    text = name.lower().replace(".", " ").replace("'", "").replace("-", " ")
    text = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", " ", text)
    return " ".join(text.split())


def _safe_franchise(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return franchise(str(value))
    except KeyError:
        return None


def _with_anytime(stats: pd.DataFrame) -> pd.DataFrame:
    frame = stats.copy()
    for column in ("rushing_tds", "receiving_tds", "special_teams_tds", "fumble_recovery_tds"):
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["anytime_td"] = (
        frame["rushing_tds"] + frame["receiving_tds"] + frame["special_teams_tds"] + frame["fumble_recovery_tds"]
    )
    frame["team_franchise"] = frame["team"].map(_safe_franchise)
    frame["opp_franchise"] = frame["opponent_team"].map(_safe_franchise)
    frame = frame.dropna(subset=["team_franchise", "opp_franchise", "player_id", "position"])
    frame["player_display_name"] = frame["player_display_name"].fillna(frame.get("player_name"))
    frame["season_order"] = frame["season_type"].map({"REG": 0, "POST": 1}).fillna(9)
    return frame


def _add_time(frame: pd.DataFrame) -> pd.DataFrame:
    keys = frame[["season", "season_order", "week"]].drop_duplicates().sort_values(["season", "season_order", "week"])
    keys = keys.copy()
    keys["t"] = np.arange(len(keys))
    return frame.merge(keys, on=["season", "season_order", "week"], how="left")


def _defense_table(frame: pd.DataFrame, stat: str) -> pd.DataFrame:
    grouped = frame.groupby(["t", "team_franchise", "opp_franchise"], as_index=False)[stat].sum()
    allowed = grouped.rename(columns={"opp_franchise": "defense", "team_franchise": "offense", stat: "allowed"})
    allowed = allowed.sort_values(["defense", "t"])
    allowed["def_enter"] = allowed.groupby("defense")["allowed"].transform(
        lambda series: series.shift(1).rolling(ROLL, min_periods=MIN_PLAYER_GAMES).mean()
    )
    allowed["def_exit"] = allowed.groupby("defense")["allowed"].transform(
        lambda series: series.rolling(ROLL, min_periods=MIN_PLAYER_GAMES).mean()
    )
    return allowed


def build_prop_state(stats: pd.DataFrame) -> PropState:
    frame = _add_time(_with_anytime(stats))
    frame = frame.sort_values(["player_id", "t"])
    accuracy: dict[str, float] = {}
    sigmas: dict[str, float] = {}
    defense_exit: dict[tuple[str, str], float] = {}
    league: dict[str, float] = {}

    for stat in STAT_COLUMNS:
        frame[f"{stat}_roll"] = frame.groupby("player_id")[stat].transform(
            lambda series: series.shift(1).rolling(ROLL, min_periods=MIN_PLAYER_GAMES).mean()
        )
        frame[f"{stat}_n"] = frame.groupby("player_id")[stat].transform(
            lambda series: series.shift(1).expanding().count()
        )
        frame[f"{stat}_exit"] = frame.groupby("player_id")[stat].transform(
            lambda series: series.rolling(ROLL, min_periods=MIN_PLAYER_GAMES).mean()
        )
        weekly = frame.groupby(["t", "position"], as_index=False)[stat].mean().sort_values(["position", "t"])
        weekly["pos_mean"] = weekly.groupby("position")[stat].transform(
            lambda series: series.shift(1).expanding(min_periods=1).mean()
        )
        frame = frame.merge(weekly[["t", "position", "pos_mean"]], on=["t", "position"], how="left")
        frame = frame.rename(columns={"pos_mean": f"{stat}_pos"})
        allowed = _defense_table(frame, stat)
        enter = allowed[["t", "defense", "def_enter"]].drop_duplicates(["t", "defense"])
        frame = frame.merge(
            enter,
            left_on=["t", "opp_franchise"],
            right_on=["t", "defense"],
            how="left",
        ).drop(columns=["defense"])
        league_by_t = frame.groupby("t")[stat].mean().sort_index()
        league_prior = league_by_t.shift(1).expanding(min_periods=1).mean()
        frame[f"{stat}_league"] = frame["t"].map(league_prior)
        factor = (frame["def_enter"] / frame[f"{stat}_league"]).clip(0.75, 1.25)
        factor = factor.fillna(1.0)
        weight = frame[f"{stat}_n"] / (frame[f"{stat}_n"] + SHRINK_GAMES)
        base = weight * frame[f"{stat}_roll"] + (1.0 - weight) * frame[f"{stat}_pos"]
        frame[f"{stat}_pred"] = base * factor
        scored = frame.loc[_starter_mask(frame, stat)].copy()
        if stat == "anytime_td":
            predicted = scored[f"{stat}_pred"].clip(lower=0)
            probability = 1.0 - np.exp(-predicted)
            actual = (scored[stat] > 0).astype(float)
            paired = pd.DataFrame({"p": probability, "y": actual}).dropna()
            accuracy[stat] = float(((paired["p"] - paired["y"]) ** 2).mean()) if len(paired) else float("nan")
            sigmas[stat] = 0.0
        else:
            residual = (scored[stat] - scored[f"{stat}_pred"]).dropna()
            accuracy[stat] = float(residual.abs().mean()) if len(residual) else float("nan")
            sigmas[stat] = float(
                max(residual.std(ddof=1) if len(residual) > 2 else SIGMA_FLOOR[stat], SIGMA_FLOOR[stat])
            )
        latest_def = (
            allowed.sort_values("t").groupby("defense", as_index=False).tail(1)[["defense", "def_exit"]]
        )
        for row in latest_def.itertuples(index=False):
            if pd.notna(row.def_exit):
                defense_exit[(str(row.defense), stat)] = float(row.def_exit)
        league[stat] = float(frame[stat].mean())
        frame = frame.drop(columns=["def_enter", f"{stat}_league"])

    players: dict[str, PlayerProjection] = {}
    last = frame.sort_values("t").groupby("player_id", as_index=False).tail(1)
    position_means = {stat: frame.groupby("position")[stat].mean() for stat in STAT_COLUMNS}
    for row in last.itertuples(index=False):
        name = str(row.player_display_name or "").strip()
        if not name:
            continue
        games = int(getattr(row, "passing_yards_n") or 0) + MIN_PLAYER_GAMES
        # _n is prior games; the exit roll includes the latest game, so count is prior + 1 when prior exists.
        means: dict[str, float] = {}
        for stat in STAT_COLUMNS:
            exit_roll = getattr(row, f"{stat}_exit")
            if pd.isna(exit_roll):
                continue
            prior = getattr(row, f"{stat}_n")
            count = 0 if pd.isna(prior) else float(prior) + 1.0
            pos_lookup = position_means[stat]
            pos_mean = float(pos_lookup.get(row.position, np.nan)) if hasattr(pos_lookup, "get") else float("nan")
            if math.isnan(pos_mean):
                pos_mean = float(exit_roll)
            weight = count / (count + SHRINK_GAMES)
            means[stat] = weight * float(exit_roll) + (1.0 - weight) * pos_mean
        if not means:
            continue
        projection = PlayerProjection(
            name=name,
            team=str(row.team_franchise),
            position=str(row.position),
            means=means,
            sigmas=dict(sigmas),
        )
        key = normalize_name(name)
        current = players.get(key)
        if current is None or games >= MIN_PLAYER_GAMES:
            players[key] = projection
    return PropState(players=players, defense=defense_exit, league=league, accuracy=accuracy)


def project_stat(state: PropState, player_key: str, stat: str, opponent: str | None) -> tuple[float, float] | None:
    player = state.players.get(player_key)
    if player is None or stat not in player.means:
        return None
    if not _eligible(player, stat):
        return None
    base = max(0.0, player.means[stat])
    league = state.league.get(stat) or 0.0
    factor = 1.0
    if opponent and league > 0:
        allowed = state.defense.get((opponent, stat))
        if allowed is not None:
            factor = min(1.25, max(0.75, allowed / league))
    mean = base * factor
    if stat == "anytime_td":
        probability = 1.0 - math.exp(-mean)
        return probability, 0.0
    return mean, player.sigmas.get(stat, SIGMA_FLOOR[stat])


def _starter_mask(frame: pd.DataFrame, stat: str) -> pd.Series:
    """Rows used for error. Zeros from players who did not play that stat would hide the miss."""
    position = frame["position"].astype(str).str.upper()
    predicted = frame[f"{stat}_pred"].notna()
    if stat == "passing_yards":
        return predicted & (position == "QB") & (frame["attempts"] >= 10)
    if stat == "rushing_yards":
        return predicted & position.isin(["RB", "QB"]) & (frame["carries"] >= 5)
    if stat in {"receiving_yards", "receptions"}:
        return predicted & position.isin(["WR", "TE", "RB"]) & (frame["targets"] >= 3)
    usage = frame["attempts"].fillna(0) + frame["carries"].fillna(0) + frame["targets"].fillna(0)
    return predicted & position.isin(["QB", "RB", "WR", "TE"]) & (usage >= 5)


def _eligible(player: PlayerProjection, stat: str) -> bool:
    position = player.position.upper()
    if stat == "passing_yards":
        return position == "QB" and player.means.get(stat, 0) >= 80
    if stat == "rushing_yards":
        return position in {"QB", "RB"} and player.means.get(stat, 0) >= 8
    if stat in {"receiving_yards", "receptions"}:
        return position in {"RB", "WR", "TE"} and player.means.get(stat, 0) > 0
    if stat == "anytime_td":
        return position in {"QB", "RB", "WR", "TE"}
    return False
