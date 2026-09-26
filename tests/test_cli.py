from nfl_model.cli import main
from nfl_model.odds_api import API_KEY_HELP, build_side_prices, parse_outcomes


def test_board_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    code = main(["board", "--bankroll", "1000"])
    assert code == 2
    error = capsys.readouterr().err
    assert error == API_KEY_HELP or "THE_ODDS_API_KEY" in error
    assert "https://the-odds-api.com" in error


def test_parlays_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    code = main(["parlays", "--bankroll", "1000"])
    assert code == 2
    assert "THE_ODDS_API_KEY" in capsys.readouterr().err


def test_parser_reads_draftkings_spread():
    payload = [
        {
            "id": "evt",
            "home_team": "Buffalo Bills",
            "away_team": "Kansas City Chiefs",
            "commence_time": "2026-09-28T17:00:00Z",
            "bookmakers": [
                {
                    "key": "draftkings",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Buffalo Bills", "price": -110, "point": -2.5},
                                {"name": "Kansas City Chiefs", "price": -110, "point": 2.5},
                            ],
                        }
                    ],
                },
                {
                    "key": "pinnacle",
                    "markets": [
                        {
                            "key": "spreads",
                            "outcomes": [
                                {"name": "Buffalo Bills", "price": -105, "point": -2.5},
                                {"name": "Kansas City Chiefs", "price": -105, "point": 2.5},
                            ],
                        }
                    ],
                },
            ],
        }
    ]
    prices = build_side_prices(parse_outcomes(payload))
    assert len(prices) == 2
    assert prices[0].market_source == "pinnacle"
    assert abs(prices[0].dk_novig - 0.5) < 1e-9
