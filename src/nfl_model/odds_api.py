"""The Odds API client for DraftKings and a sharp or consensus comparison.

https://the-odds-api.com publishes DraftKings as a US book. This module only
reads odds. It does not log in to DraftKings or place a bet.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime

from nfl_model.odds_math import implied_probability, remove_vig

API_ROOT = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
GAME_MARKETS = "h2h,spreads,totals"
PROP_MARKETS = "player_pass_yds,player_rush_yds,player_reception_yds,player_receptions,player_anytime_td"
BOOKS = "draftkings,pinnacle,betonlineag,lowvig,fanduel,betmgm"
SHARP_BOOK = "pinnacle"

API_KEY_HELP = """THE_ODDS_API_KEY is not set.

DraftKings lines are read from The Odds API (https://the-odds-api.com), which
lists DraftKings as a US book. This tool does not log in to DraftKings, scrape
the site, or place bets. You place every bet yourself.

  1. Create an account at https://the-odds-api.com
  2. Copy the API key from the dashboard
  3. export THE_ODDS_API_KEY=your_key_here
  4. Re-run this command with --bankroll set to your real bankroll

The backtest and the qualification check do not need a key:

  python -m nfl_model backtest
  python -m nfl_model qualification
"""


class OddsApiError(RuntimeError):
    """The odds request could not be completed."""


class MissingApiKeyError(OddsApiError):
    """No API key is configured."""


@dataclass(frozen=True)
class Outcome:
    event_id: str
    home: str
    away: str
    commence: datetime
    book: str
    market: str
    name: str
    description: str
    point: float | None
    american: int


@dataclass(frozen=True)
class SidePrice:
    event_id: str
    home: str
    away: str
    commence: datetime
    market: str
    side_name: str
    description: str
    point: float | None
    dk_american: int
    dk_novig: float
    pinnacle_novig: float | None
    consensus_novig: float | None
    market_novig: float | None
    market_source: str
    note: str


@dataclass(frozen=True)
class ScoreState:
    event_id: str
    home: str
    away: str
    commence: datetime
    completed: bool
    home_score: int | None
    away_score: int | None


@dataclass(frozen=True)
class ApiPayload:
    body: list
    remaining: str | None
    last_cost: str | None


def coerce_events(body: object, path: str) -> list:
    """Odds and scores return a list. One event's odds return a single object."""
    if isinstance(body, list):
        return body
    if isinstance(body, dict):
        if body.get("message") and "bookmakers" not in body and "id" not in body:
            raise OddsApiError(f"The Odds API said: {body['message']} ({path})")
        return [body]
    raise OddsApiError(f"Unexpected response from {path}")


def require_api_key() -> str:
    key = os.environ.get("THE_ODDS_API_KEY", "").strip()
    if not key:
        raise MissingApiKeyError(API_KEY_HELP)
    return key


def _request(path: str, params: dict[str, str], key: str) -> ApiPayload:
    query = dict(params)
    query["apiKey"] = key
    url = f"{API_ROOT}{path}?{urllib.parse.urlencode(query)}"
    request = urllib.request.Request(url, headers={"User-Agent": "nfl-model/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            body = json.loads(response.read().decode())
            headers = {name.lower(): value for name, value in response.headers.items()}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        if exc.code in {401, 403}:
            raise OddsApiError(
                "The Odds API rejected THE_ODDS_API_KEY. Check the key in the dashboard "
                f"at https://the-odds-api.com and export it again.\n{detail}"
            ) from exc
        if exc.code == 429:
            raise OddsApiError(
                "The Odds API rate limit or credit balance was hit. Wait and retry, "
                "or check the usage dashboard at https://the-odds-api.com.\n" + detail
            ) from exc
        raise OddsApiError(f"The Odds API returned HTTP {exc.code} for {path}.\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise OddsApiError(f"Could not reach The Odds API ({path}): {exc.reason}") from exc
    return ApiPayload(
        body=coerce_events(body, path),
        remaining=headers.get("x-requests-remaining"),
        last_cost=headers.get("x-requests-last"),
    )


def fetch_game_odds(key: str) -> ApiPayload:
    params = {
        "regions": "us,eu",
        "markets": GAME_MARKETS,
        "oddsFormat": "american",
        "bookmakers": BOOKS,
    }
    try:
        return _request(f"/sports/{SPORT}/odds", params, key)
    except OddsApiError as exc:
        if "HTTP 422" not in str(exc):
            raise
        params.pop("bookmakers")
        return _request(f"/sports/{SPORT}/odds", params, key)


def fetch_scores(key: str) -> ApiPayload:
    return _request(f"/sports/{SPORT}/scores", {"daysFrom": "1"}, key)


def fetch_event_props(key: str, event_id: str) -> ApiPayload:
    params = {
        "regions": "us,eu",
        "markets": PROP_MARKETS,
        "oddsFormat": "american",
        "bookmakers": BOOKS,
    }
    path = f"/sports/{SPORT}/events/{event_id}/odds"
    try:
        return _request(path, params, key)
    except OddsApiError as exc:
        if "HTTP 422" not in str(exc):
            raise
        params.pop("bookmakers")
        return _request(path, params, key)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def parse_outcomes(payload: list[dict]) -> list[Outcome]:
    parsed: list[Outcome] = []
    for event in payload:
        commence_raw = event.get("commence_time")
        if not commence_raw or not event.get("id"):
            continue
        commence = _parse_time(str(commence_raw))
        for book in event.get("bookmakers") or []:
            book_key = str(book.get("key") or "")
            for market in book.get("markets") or []:
                market_key = str(market.get("key") or "")
                for outcome in market.get("outcomes") or []:
                    price = outcome.get("price")
                    if price is None or outcome.get("name") is None:
                        continue
                    point = outcome.get("point")
                    parsed.append(
                        Outcome(
                            event_id=str(event["id"]),
                            home=str(event.get("home_team") or ""),
                            away=str(event.get("away_team") or ""),
                            commence=commence,
                            book=book_key,
                            market=market_key,
                            name=str(outcome["name"]),
                            description=str(outcome.get("description") or ""),
                            point=None if point is None else float(point),
                            american=int(price),
                        )
                    )
    return parsed


def parse_scores(payload: list[dict]) -> list[ScoreState]:
    states: list[ScoreState] = []
    for event in payload:
        if not event.get("id") or not event.get("commence_time"):
            continue
        home = str(event.get("home_team") or "")
        away = str(event.get("away_team") or "")
        by_name = {str(item.get("name")): item.get("score") for item in event.get("scores") or []}
        home_score = _score(by_name.get(home))
        away_score = _score(by_name.get(away))
        states.append(
            ScoreState(
                event_id=str(event["id"]),
                home=home,
                away=away,
                commence=_parse_time(str(event["commence_time"])),
                completed=bool(event.get("completed")),
                home_score=home_score,
                away_score=away_score,
            )
        )
    return states


def _score(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _pair_key(outcome: Outcome) -> tuple:
    if outcome.market == "spreads" and outcome.point is not None:
        point: float | None = round(abs(outcome.point), 2)
    elif outcome.point is None:
        point = None
    else:
        point = round(outcome.point, 2)
    return (outcome.event_id, outcome.book, outcome.market, outcome.description, point)


def _same_point(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(left - right) < 0.01


def build_side_prices(outcomes: list[Outcome]) -> list[SidePrice]:
    grouped: dict[tuple, list[Outcome]] = {}
    for outcome in outcomes:
        grouped.setdefault(_pair_key(outcome), []).append(outcome)
    prices: list[SidePrice] = []
    for key, booksides in grouped.items():
        if key[1] != "draftkings" or len(booksides) < 2:
            continue
        try:
            novigs = remove_vig([implied_probability(item.american) for item in booksides])
        except ValueError:
            continue
        for side, dk_novig in zip(booksides, novigs):
            pinnacle, consensus, note = _comparison(grouped, side)
            if pinnacle is not None:
                market_novig, source = pinnacle, "pinnacle"
            elif consensus is not None:
                market_novig, source = consensus, "consensus"
            else:
                market_novig, source = None, "missing"
            prices.append(
                SidePrice(
                    event_id=side.event_id,
                    home=side.home,
                    away=side.away,
                    commence=side.commence,
                    market=side.market,
                    side_name=side.name,
                    description=side.description,
                    point=side.point,
                    dk_american=side.american,
                    dk_novig=dk_novig,
                    pinnacle_novig=pinnacle,
                    consensus_novig=consensus,
                    market_novig=market_novig,
                    market_source=source,
                    note=note,
                )
            )
    return prices


def _comparison(
    grouped: dict[tuple, list[Outcome]],
    side: Outcome,
) -> tuple[float | None, float | None, str]:
    pinnacle: float | None = None
    samples: list[float] = []
    other_points: list[str] = []
    for key, booksides in grouped.items():
        event_id, book, market, description, _point = key
        if book == "draftkings" or event_id != side.event_id or market != side.market or description != side.description:
            continue
        if len(booksides) < 2:
            continue
        match = next((item for item in booksides if item.name == side.name and _same_point(item.point, side.point)), None)
        if match is None:
            offered = ", ".join(f"{item.name} {item.point}" for item in booksides[:4])
            other_points.append(f"{book} has {offered}")
            continue
        try:
            novigs = remove_vig([implied_probability(item.american) for item in booksides])
        except ValueError:
            continue
        for item, novig in zip(booksides, novigs):
            if item is match:
                samples.append(novig)
                if book == SHARP_BOOK:
                    pinnacle = novig
                break
    consensus = sum(samples) / len(samples) if samples else None
    note = ""
    if consensus is None and other_points:
        note = "No like-for-like price. " + "; ".join(other_points[:3])
    elif pinnacle is not None:
        note = "Market price is Pinnacle with the vig removed."
    elif consensus is not None:
        note = "Pinnacle was not on this number. Market price is the average of the other books."
    else:
        note = "No other book published this number."
    return pinnacle, consensus, note
