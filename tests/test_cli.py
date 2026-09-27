from pathlib import Path

from nfl_model.cli import main
from nfl_model.ledger_log import StakeOffer, save_offers
from nfl_model.odds_api import API_KEY_HELP, build_side_prices, coerce_events, parse_outcomes


def test_board_without_api_key(monkeypatch, capsys):
    monkeypatch.delenv("THE_ODDS_API_KEY", raising=False)
    code = main(["board", "--bankroll", "1000"])
    assert code == 2
    error = capsys.readouterr().err
    assert error == API_KEY_HELP or "THE_ODDS_API_KEY" in error
    assert "https://the-odds-api.com" in error


def test_qualification_reports_a_profit_goal(tmp_path: Path, capsys):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "record,bet_number,close_number,direction,note,stake,american,result\n"
    )
    code = main(
        [
            "qualification",
            "--bankroll",
            "1000",
            "--ledger",
            str(ledger),
            "--goal-return",
            "0.20",
            "--goal-weeks",
            "3",
            "--goal-start",
            "2026-09-27",
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "profit goal: +20% of the bankroll within 3 weeks from 2026-09-27" in output
    assert "Goal: +20% ($200.00) by Oct 18." in output
    assert "The stake stays on the locked rule." in output


def test_qualification_reports_the_season_stop(tmp_path: Path, capsys):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "record,bet_number,close_number,direction,note,stake,american,result\n"
        "sides,-3.5,-4,spread,lost,25,-110,loss\n"
    )
    code = main(["qualification", "--bankroll", "100", "--ledger", str(ledger)])
    assert code == 0
    output = capsys.readouterr().out
    assert "Season stop" in output
    assert "$-25.00" in output
    assert "Standing goals:" in output
    assert "about +$1.00 by the postseason" in output
    assert "sides $-25.00" in output


def test_log_writes_only_accepted_stakes(tmp_path: Path, monkeypatch, capsys):
    saved = tmp_path / "board.json"
    monkeypatch.setattr("nfl_model.cli.offers_path", lambda source, root=None: saved)
    save_offers(
        saved,
        [
            StakeOffer(1, "sides", -3.5, "spread", "A at B Home -3.5", 6.5, -110),
            StakeOffer(2, "totals", 47.5, "over", "A at B Over 47.5", 6.5, -105),
        ],
    )
    ledger = tmp_path / "ledger.csv"
    code = main(["log", "board", "--accept", "1", "--price", "1:-108", "--ledger", str(ledger)])
    assert code == 0
    output = capsys.readouterr().out
    assert "Logged 1 stake" in output
    assert "Close and result are blank." in output
    row = ledger.read_text().splitlines()[1]
    assert row == "sides,-3.5,,spread,A at B Home -3.5,6.50,-108,"

    code = main(["log", "board", "--accept", "1,2", "--ledger", str(ledger)])
    assert code == 0
    again = capsys.readouterr().out
    assert "Already open" in again
    lines = ledger.read_text().splitlines()
    assert len(lines) == 3
    assert lines[2].startswith("totals,47.5,,over,")

    code = main(["log", "board", "--accept", "9", "--ledger", str(ledger)])
    assert code == 2
    assert "9" in capsys.readouterr().err
    assert len(ledger.read_text().splitlines()) == 3


def test_log_without_a_saved_list_fails(tmp_path: Path, monkeypatch, capsys):
    missing = tmp_path / "props.json"
    monkeypatch.setattr("nfl_model.cli.offers_path", lambda source, root=None: missing)
    code = main(["log", "props", "--accept", "1", "--ledger", str(tmp_path / "ledger.csv")])
    assert code == 2
    assert "No saved props stakes" in capsys.readouterr().err
    assert not (tmp_path / "ledger.csv").exists()


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


def test_event_odds_response_is_one_object():
    payload = {
        "id": "b225458a282e140c44c45651255f2f6c",
        "home_team": "Buffalo Bills",
        "away_team": "Kansas City Chiefs",
        "commence_time": "2026-09-28T17:00:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "player_pass_yds",
                        "outcomes": [
                            {"name": "Over", "description": "Josh Allen", "price": -115, "point": 250.5},
                            {"name": "Under", "description": "Josh Allen", "price": -105, "point": 250.5},
                        ],
                    }
                ],
            }
        ],
    }
    events = coerce_events(payload, "/sports/americanfootball_nfl/events/b225458a282e140c44c45651255f2f6c/odds")
    prices = build_side_prices(parse_outcomes(events))
    assert len(events) == 1
    assert len(prices) == 2
    assert prices[0].description == "Josh Allen"
