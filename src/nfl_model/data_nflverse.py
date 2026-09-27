"""Download public nflverse schedules and weekly player stats.

No API key. Files are cached under .cache. The first backtest needs
a network connection; later runs reuse the cache unless --refresh is set.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv"
PLAYER_WEEK_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/"
    "stats_player_week_{season}.csv.gz"
)
PLAYED_GAME_TYPES = {"REG", "WC", "DIV", "CON", "SB"}
PLAYER_SEASONS = range(2015, 2027)

def project_root() -> Path:
    """Directory that holds this checkout.

    The current working directory wins, so a normal install still reads
    `.cache` and `ledger.csv` from the folder you launched in. The source
    tree is the fallback when the package file itself sits in that checkout.
    """

    def marked(path: Path) -> bool:
        return (path / "pyproject.toml").is_file() and (path / "ledger.example.csv").is_file()

    cwd = Path.cwd()
    for candidate in [cwd, *cwd.parents]:
        if marked(candidate):
            return candidate
    for candidate in Path(__file__).resolve().parents:
        if marked(candidate):
            return candidate
    return cwd


def cache_dir() -> Path:
    path = project_root() / ".cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(url: str, dest: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-model/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            dest.write_bytes(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Download failed ({exc.code}) for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not download nflverse data. A network connection is required "
            f"the first time.\n{url}\n{exc.reason}"
        ) from exc


def _cached(url: str, filename: str, refresh: bool) -> Path:
    dest = cache_dir() / filename
    if refresh or not dest.exists() or dest.stat().st_size == 0:
        _download(url, dest)
    return dest


def load_games(refresh: bool = False) -> pd.DataFrame:
    path = _cached(GAMES_URL, "games.csv", refresh)
    frame = pd.read_csv(path, low_memory=False)
    frame = frame[frame["game_type"].isin(PLAYED_GAME_TYPES)].copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"])
    for column in (
        "home_score",
        "away_score",
        "spread_line",
        "total_line",
        "home_moneyline",
        "away_moneyline",
        "home_spread_odds",
        "away_spread_odds",
        "over_odds",
        "under_odds",
        "result",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    return frame.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def completed_games(games: pd.DataFrame) -> pd.DataFrame:
    return games.dropna(subset=["home_score", "away_score"]).copy()


def load_player_weeks(refresh: bool = False, seasons: range = PLAYER_SEASONS) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    for season in seasons:
        filename = f"stats_player_week_{season}.csv.gz"
        url = PLAYER_WEEK_URL.format(season=season)
        dest = cache_dir() / filename
        if refresh or not dest.exists() or dest.stat().st_size == 0:
            try:
                _download(url, dest)
            except RuntimeError as exc:
                if "404" in str(exc):
                    missing.append(season)
                    if dest.exists():
                        dest.unlink()
                    continue
                raise
        frame = pd.read_csv(dest, low_memory=False)
        frames.append(frame)
    if not frames:
        raise RuntimeError(
            "No weekly player stat files were downloaded from nflverse. "
            f"Missing seasons: {missing}"
        )
    stats = pd.concat(frames, ignore_index=True)
    stats = stats[stats["season_type"].isin(["REG", "POST"])].copy()
    numeric = [
        "season",
        "week",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "rushing_tds",
        "receiving_tds",
        "passing_tds",
        "special_teams_tds",
        "fumble_recovery_tds",
        "attempts",
        "carries",
        "targets",
    ]
    for column in numeric:
        if column in stats.columns:
            stats[column] = pd.to_numeric(stats[column], errors="coerce").fillna(0)
    return stats
