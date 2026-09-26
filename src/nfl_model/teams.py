"""Team abbreviations used by nflverse, and the names DraftKings prints."""

from __future__ import annotations

# Historical abbreviations collapse onto the current franchise so a move
# (San Diego to the Chargers, St. Louis to the Rams) does not reset the rating.
FRANCHISE = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BUF": "BUF",
    "CAR": "CAR",
    "CHI": "CHI",
    "CIN": "CIN",
    "CLE": "CLE",
    "DAL": "DAL",
    "DEN": "DEN",
    "DET": "DET",
    "GB": "GB",
    "HOU": "HOU",
    "IND": "IND",
    "JAC": "JAX",
    "JAX": "JAX",
    "KC": "KC",
    "LA": "LAR",
    "LAR": "LAR",
    "LAC": "LAC",
    "SD": "LAC",
    "LV": "LV",
    "OAK": "LV",
    "MIA": "MIA",
    "MIN": "MIN",
    "NE": "NE",
    "NO": "NO",
    "NYG": "NYG",
    "NYJ": "NYJ",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SF": "SF",
    "STL": "LAR",
    "TB": "TB",
    "TEN": "TEN",
    "WAS": "WAS",
    "WSH": "WAS",
}

FULL_NAME_TO_FRANCHISE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

FRANCHISE_TO_NAME = {value: key for key, value in FULL_NAME_TO_FRANCHISE.items()}


def franchise(abbreviation: str) -> str:
    key = abbreviation.strip().upper()
    if key not in FRANCHISE:
        raise KeyError(f"Unknown NFL team abbreviation {abbreviation!r}")
    return FRANCHISE[key]


def franchise_from_name(name: str) -> str | None:
    return FULL_NAME_TO_FRANCHISE.get(name.strip())
