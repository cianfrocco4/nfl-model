"""Core prices, staking, qualification, and parlay joints."""

import math
from datetime import date, datetime, timezone
from pathlib import Path

from nfl_model.odds_math import (
    american_to_decimal,
    decimal_to_american,
    edge,
    expected_value,
    implied_probability,
    remove_vig,
)
from nfl_model.ledger_log import (
    LedgerLogError,
    StakeOffer,
    append_accepted,
    offers_from_decisions,
    offers_from_tickets,
)
from nfl_model.parlay import (
    ParlayTicket,
    bivariate_cdf,
    game_leg_from_moneyline,
    game_leg_from_spread,
    game_leg_from_total,
    independent_probability,
    joint_probability,
    price_ticket,
    render_parlays,
)
from nfl_model.probabilities import moneyline_outcomes, spread_outcomes
from nfl_model.data_nflverse import project_root
from nfl_model.qualification import (
    LedgerError,
    SeasonMoney,
    bet_identity,
    bet_profit,
    closing_line_value,
    grade_record,
    qualification_report,
    ledger_gaps,
    make_profit_goal,
    season_money,
    season_progress,
)
from nfl_model.recommend import apply_season_loss_limit, decide, render, slate_end
from nfl_model.settings import Settings, SettingsError, StakingMethod
from nfl_model.staking import kelly_fraction, method_for_policy, stake_dollars


def test_american_decimal_round_trip():
    assert american_to_decimal(150) == 2.5
    assert american_to_decimal(-200) == 1.5
    assert decimal_to_american(2.5) == 150
    assert decimal_to_american(1.5) == -200
    assert implied_probability(-110) == 1 / american_to_decimal(-110)


def test_vig_removal_and_ev_and_edge():
    novig = remove_vig([implied_probability(-110), implied_probability(-110)])
    assert novig == [0.5, 0.5]
    assert expected_value(0.5, 0.0, 2.0) == 0
    assert math.isclose(expected_value(0.55, 0.0, 2.0), 0.1)
    assert math.isclose(edge(0.55, 0.45, 0.5), 0.05)


def test_kelly_zero_negative_and_cap():
    assert kelly_fraction(0.5, 0.5, 2.0) == 0
    assert kelly_fraction(0.4, 0.6, 2.0) < 0
    assert stake_dollars(
        StakingMethod.QUARTER, 1000, take=True, p_win=0.5, p_loss=0.5,
        decimal_odds=2.0, flat_fraction=0.01, unit_size=None, max_fraction=0.05,
    ) == 0
    assert stake_dollars(
        StakingMethod.FULL, 1000, take=True, p_win=0.4, p_loss=0.6,
        decimal_odds=2.0, flat_fraction=0.01, unit_size=None, max_fraction=0.05,
    ) == 0
    # Even money, 60% win: full Kelly is 20%. Quarter is 5%, which hits the 5% cap.
    capped = stake_dollars(
        StakingMethod.QUARTER, 1000, take=True, p_win=0.6, p_loss=0.4,
        decimal_odds=2.0, flat_fraction=0.01, unit_size=None, max_fraction=0.02,
    )
    assert capped == 20


def test_policy_upgrade_and_parlay_flat():
    assert method_for_policy("sides", qualified=False, selected=StakingMethod.POLICY) is StakingMethod.FLAT
    assert method_for_policy("totals", qualified=True, selected=StakingMethod.POLICY) is StakingMethod.QUARTER
    assert method_for_policy("live", qualified=True, selected=StakingMethod.POLICY) is StakingMethod.FLAT
    assert method_for_policy("parlays", qualified=True, selected=StakingMethod.QUARTER) is StakingMethod.FLAT
    settings = Settings(bankroll=1000)
    assert settings.sleeve_sides == 0.65
    assert settings.sleeve_props == 0.20
    assert settings.sleeve_live == 0.10
    assert settings.sleeve_parlays == 0.05
    assert math.isclose(settings.pool("parlays"), 50)


def test_closing_line_value_and_qualification():
    assert closing_line_value(47.5, 49, "over") == 1.5
    assert closing_line_value(47.5, 49, "under") == -1.5
    assert closing_line_value(-3.5, -6.5, "spread") == 3
    assert grade_record("sides", [0.1] * 99).qualified is False
    assert grade_record("sides", [0.0] * 100).qualified is False
    assert grade_record("totals", [0.2] * 100).qualified is True
    assert grade_record("live", [1.0] * 200).qualified is False
    assert grade_record("parlays", [1.0] * 200).qualified is False


def test_both_agree_stakes_and_a_single_signal_is_a_lean():
    settings = Settings(bankroll=1000, min_edge=0.02, min_ev=0.0)
    both = decide(
        game="A at B", market="spreads", side="Home -3.5", record="sides",
        dk_american=-110, dk_novig=0.5, p_win=0.58, p_push=0.0,
        market_probability=0.56, market_source="pinnacle", note="",
        settings=settings, qualified=False,
    )
    assert both.status == "stake"
    assert both.stake == round(1000 * 0.65 * 0.01, 2)
    lean = decide(
        game="A at B", market="spreads", side="Home -3.5", record="sides",
        dk_american=-110, dk_novig=0.5, p_win=0.58, p_push=0.0,
        market_probability=0.50, market_source="pinnacle", note="",
        settings=settings, qualified=False,
    )
    assert lean.status == "lean"
    assert lean.stake == 0


def test_bivariate_cdf_matches_known_values():
    assert abs(bivariate_cdf(0, 0, 0) - 0.25) < 1e-6
    expected = 0.25 + math.asin(0.5) / (2 * math.pi)
    assert abs(bivariate_cdf(0, 0, 0.5) - expected) < 1e-3


def test_same_game_spread_and_moneyline_use_the_stricter_margin():
    spread = game_leg_from_spread(
        event_id="g", label="home -7.5", both_agree=True,
        mu_margin=3, sigma_margin=13, home_point=-7.5, betting_home=True,
    )
    moneyline = game_leg_from_moneyline(
        event_id="g", label="home ML", both_agree=True,
        mu_margin=3, sigma_margin=13, betting_home=True,
    )
    joint = joint_probability([spread, moneyline], rho_margin_total=0.2)
    home_cover, _push, _loss = spread_outcomes(3, 13, -7.5)
    assert abs(joint - home_cover) < 0.01
    # Nested same-game legs move together, so the joint is the stricter leg
    # and is larger than the independent product.
    assert joint > independent_probability([spread, moneyline])


def test_cross_game_parlay_is_the_product_and_stakes_flat():
    first = game_leg_from_moneyline(
        event_id="a", label="a", both_agree=True, mu_margin=0, sigma_margin=13, betting_home=True,
    )
    second = game_leg_from_total(
        event_id="b", label="b", both_agree=True, mu_total=45, sigma_total=12, line=44.5, side="over",
    )
    joint = joint_probability([first, second], rho_margin_total=0.4)
    assert abs(joint - first.p_win * second.p_win) < 0.01
    settings = Settings(bankroll=1000, min_edge=0.0, min_ev=0.0, staking=StakingMethod.QUARTER)
    # A very long price is +EV. Quarter Kelly on the settings must not change the parlay stake.
    ticket = price_ticket([first, second], 500, settings, rho_margin_total=0.0)
    assert ticket.kind == "cross-game"
    assert ticket.stake == round(50 * 0.01, 2)
    short = price_ticket([first, second], -500, settings, rho_margin_total=0.0)
    assert short.stake == 0
    blocked = price_ticket(
        [first, game_leg_from_total(
            event_id="b", label="b", both_agree=False, mu_total=45, sigma_total=12, line=44.5, side="over",
        )],
        500, settings, rho_margin_total=0.0,
    )
    assert blocked.status == "lean"
    assert blocked.stake == 0


def test_opposite_totals_cannot_both_win():
    over = game_leg_from_total(
        event_id="g", label="over", both_agree=True, mu_total=45, sigma_total=10, line=45.5, side="over",
    )
    under = game_leg_from_total(
        event_id="g", label="under", both_agree=True, mu_total=45, sigma_total=10, line=45.5, side="under",
    )
    assert joint_probability([over, under], rho_margin_total=0.0) == 0


def test_saturday_slate_ends_tuesday_noon_eastern():
    saturday_evening = datetime(2026, 9, 26, 23, 0, tzinfo=timezone.utc)
    assert slate_end(saturday_evening) == datetime(2026, 9, 29, 16, 0, tzinfo=timezone.utc)


def test_blank_closing_number_is_reported_and_not_graded(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "record,bet_number,close_number,direction,note\n"
        "sides,-3.5,,spread,waiting\n"
        "sides,-3.5,-6.5,spread,graded\n"
    )
    status = next(row for row in qualification_report(ledger) if row.record == "sides")
    assert status.graded == 1
    assert status.ungraded == 1
    assert status.qualified is False
    assert "do not count" in status.reason


def test_default_render_is_the_stake_list():
    settings = Settings(bankroll=1000, min_edge=0.02, min_ev=0.0)
    stake = decide(
        game="A at B", market="spreads", side="Home -3.5", record="sides",
        dk_american=-110, dk_novig=0.5, p_win=0.58, p_push=0.0,
        market_probability=0.56, market_source="pinnacle", note="",
        settings=settings, qualified=False,
        kickoff=datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc),
    )
    passed = decide(
        game="C at D", market="totals", side="Over 44.5", record="totals",
        dk_american=-110, dk_novig=0.5, p_win=0.5, p_push=0.0,
        market_probability=0.5, market_source="pinnacle", note="",
        settings=settings, qualified=False,
    )
    text = render([stake, passed], "Pregame board")
    assert "Suggested stakes" in text
    assert "log board" not in text
    logged = render([stake, passed], "Pregame board", log_source="board")
    assert "python -m nfl_model log board --accept 1" in logged
    assert "Home -3.5" in text
    assert "ET" in text
    assert "Every signal" not in text
    assert "1 other side is a lean or a pass" in text
    assert "Every signal" in render([stake, passed], "Pregame board", verbose=True)


def test_project_root_is_the_checkout_that_holds_the_ledger_example():
    root = project_root()
    assert (root / "ledger.example.csv").is_file()
    assert (root / "pyproject.toml").is_file()


def test_settled_profit_uses_stake_and_result(tmp_path: Path):
    assert bet_profit(0.65, -110, "loss") == -0.65
    assert bet_profit(0.65, 150, "win") == 0.65 * 1.5
    assert bet_profit(0.20, None, "push") == 0
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "record,bet_number,close_number,direction,note,stake,american,result\n"
        "sides,-3.5,-4.5,spread,lost,0.65,-110,loss\n"
        "sides,-3.5,,spread,open,0.65,-110,\n"
        "totals,47.5,49,over,won,0.65,150,win\n"
        "props,64.5,60,over,push,0.20,-110,push\n"
    )
    money = season_money(ledger)
    assert money.tracks_money is True
    assert money.settled == 3
    assert money.unsettled == 1
    assert (money.wins, money.losses, money.pushes) == (1, 1, 1)
    assert math.isclose(money.profit, -0.65 + 0.65 * 1.5)
    status = next(row for row in qualification_report(ledger) if row.record == "sides")
    assert status.graded == 1
    assert status.ungraded == 1


def test_old_ledger_without_money_columns_is_not_a_profit(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "record,bet_number,close_number,direction,note\n"
        "sides,-3.5,-6.5,spread,graded\n"
    )
    money = season_money(ledger)
    assert money.tracks_money is False
    assert money.profit == 0
    assert money.settled == 0


def test_a_win_without_odds_is_rejected(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "record,bet_number,close_number,direction,note,stake,american,result\n"
        "sides,-3.5,-6.5,spread,missing price,0.65,,win\n"
    )
    try:
        season_money(ledger)
    except LedgerError as exc:
        assert "american" in str(exc)
    else:
        raise AssertionError("expected a ledger error")


def test_season_loss_limit_zeros_a_stake():
    settings = Settings(bankroll=100)
    both = decide(
        game="A at B", market="spreads", side="Home -3.5", record="sides",
        dk_american=-110, dk_novig=0.5, p_win=0.58, p_push=0.0,
        market_probability=0.56, market_source="pinnacle", note="",
        settings=settings, qualified=False,
    )
    assert both.stake == round(100 * 0.65 * 0.01, 2)
    stopped_money = SeasonMoney(1, 0, -25.0, 0, 1, 0, True)
    stopped = apply_season_loss_limit([both], settings, stopped_money)
    assert stopped[0].stake == 0
    assert "Season stop" in stopped[0].reason
    still_open = SeasonMoney(1, 0, -24.0, 0, 1, 0, True)
    kept = apply_season_loss_limit([both], settings, still_open)
    assert kept[0].stake == both.stake


def test_profit_goal_is_a_progress_check_and_leaves_the_stake():
    start = date(2026, 9, 27)
    plain = Settings(bankroll=1000)
    aimed = Settings(bankroll=1000, goal_return=0.20, goal_weeks=3, goal_start=start)
    kwargs = dict(
        game="A at B", market="spreads", side="Home -3.5", record="sides",
        dk_american=-110, dk_novig=0.5, p_win=0.58, p_push=0.0,
        market_probability=0.56, market_source="pinnacle", note="", qualified=False,
    )
    assert decide(**kwargs, settings=aimed).stake == decide(**kwargs, settings=plain).stake
    goal = make_profit_goal(aimed.goal_return, aimed.goal_weeks, aimed.goal_start)
    money = SeasonMoney(0, 0, 0.0, 0, 0, 0, True)
    text = season_progress(
        money, 1000, 0.25,
        goal=goal,
        today=start,
        flat_stake=6.50,
        min_edge=0.02,
    )
    assert "Goal: +20% ($200.00) by Oct 18." in text
    assert "at least 1539 bets" in text
    assert "at most 48 games" in text
    assert "The stake stays on the locked rule." in text
    reached = season_progress(
        SeasonMoney(4, 0, 200.0, 4, 0, 0, True), 1000, 0.25,
        goal=goal,
        today=date(2026, 10, 1),
        flat_stake=6.50,
    )
    assert "has reached $200.00" in reached


def test_ledger_gaps_and_profit_by_record(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "record,bet_number,close_number,direction,note,stake,american,result\n"
        "sides,-3.5,-6.5,spread,won,10,-110,win\n"
        "props,64.5,,over,open,2,-115,\n"
        "totals,47.5,49,over,no stake,,,\n"
    )
    money = season_money(ledger)
    assert math.isclose(dict(money.record_profit)["sides"], 10 * (american_to_decimal(-110) - 1))
    assert dict(money.record_profit)["props"] == 0
    gaps = ledger_gaps(ledger)
    assert gaps is not None
    assert gaps.rows == 3
    assert gaps.missing_close == 1
    assert gaps.missing_stake == 1
    assert gaps.open_bets == 1


def test_profit_goal_requires_return_weeks_and_start():
    try:
        Settings(bankroll=1000, goal_return=0.20)
    except SettingsError as exc:
        assert "--goal-return" in str(exc)
    else:
        raise AssertionError("expected a settings error")
    try:
        Settings(bankroll=1000, goal_return=1.5, goal_weeks=3, goal_start=date(2026, 9, 27))
    except SettingsError as exc:
        assert "0.20" in str(exc)
    else:
        raise AssertionError("expected a settings error")


def test_bet_identity_uses_the_line_or_the_decimal_price():
    assert bet_identity("spreads", "Buffalo Bills", -3.5, -110) == (-3.5, "spread")
    assert bet_identity("totals", "Over", 47.5, -110) == (47.5, "over")
    assert bet_identity("player_pass_yds", "Under", 245.5, -115) == (245.5, "under")
    number, direction = bet_identity("h2h", "Buffalo Bills", None, -110)
    assert direction == "decimal"
    assert number == round(american_to_decimal(-110), 4)
    td_number, td_direction = bet_identity("player_anytime_td", "Yes", None, 150)
    assert (td_number, td_direction) == (2.5, "decimal")
    try:
        bet_identity("spreads", "Buffalo Bills", None, -110)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a spread without a point to fail")


def test_accepted_stakes_fill_the_ledger_and_skip_open_duplicates(tmp_path: Path):
    settings = Settings(bankroll=1000, min_edge=0.02, min_ev=0.0)
    spread = decide(
        game="Chiefs at Bills", market="spreads", side="Bills -3.5", record="sides",
        dk_american=-110, dk_novig=0.5, p_win=0.58, p_push=0.0,
        market_probability=0.56, market_source="pinnacle", note="comparison",
        settings=settings, qualified=False, bet_number=-3.5, direction="spread",
    )
    total = decide(
        game="Chiefs at Bills", market="totals", side="Over 47.5", record="totals",
        dk_american=-105, dk_novig=0.5, p_win=0.58, p_push=0.0,
        market_probability=0.56, market_source="pinnacle", note="",
        settings=settings, qualified=False, bet_number=47.5, direction="over",
    )
    moneyline = decide(
        game="Chiefs at Bills", market="h2h", side="Bills", record="sides",
        dk_american=-150, dk_novig=0.6, p_win=0.7, p_push=0.0,
        market_probability=0.65, market_source="pinnacle", note="",
        settings=settings, qualified=False,
        bet_number=round(american_to_decimal(-150), 4), direction="decimal",
    )
    offers = offers_from_decisions([spread, total, moneyline])
    assert [offer.number for offer in offers] == [1, 2, 3]
    assert offers[0].note == "Chiefs at Bills Bills -3.5"
    assert "comparison" not in offers[0].note
    ledger = tmp_path / "ledger.csv"
    written = append_accepted(ledger, offers, [1, 3], {1: -108})
    text = ledger.read_text()
    assert text.splitlines()[0] == "record,bet_number,close_number,direction,note,stake,american,result"
    assert "sides,-3.5,,spread,Chiefs at Bills Bills -3.5," in text
    assert ",-108," in text
    assert "totals" not in text
    assert f"{round(american_to_decimal(-150), 4)},,decimal" in text
    assert written.written[0].american == -108
    assert written.written[0].bet_number == -3.5
    money = season_money(ledger)
    assert money.unsettled == 2
    assert money.settled == 0
    status = next(row for row in qualification_report(ledger) if row.record == "sides")
    assert status.graded == 0
    assert status.ungraded == 2

    again = append_accepted(ledger, offers, [1], {1: -120})
    assert again.written == ()
    assert len(again.skipped) == 1
    assert again.skipped[0].ledger_american == -108
    assert ledger.read_text().count("sides,-3.5,") == 1

    moved = append_accepted(ledger, offers, [3], {3: 150})
    assert moved.written[0].direction == "decimal"
    assert moved.written[0].bet_number == 2.5
    assert moved.written[0].american == 150
    assert ledger.read_text().count("decimal") == 2


def test_unknown_stake_number_writes_nothing(tmp_path: Path):
    ledger = tmp_path / "ledger.csv"
    offer = StakeOffer(1, "props", 64.5, "over", "Player Over 64.5", 2.0, -115)
    try:
        append_accepted(ledger, [offer], [4], {})
    except LedgerLogError as exc:
        assert "4" in str(exc)
    else:
        raise AssertionError("expected an unknown stake number")
    assert not ledger.exists()


def test_priced_parlay_slips_are_numbered_for_the_log():
    first = game_leg_from_moneyline(
        event_id="a", label="A ML", both_agree=True, mu_margin=1, sigma_margin=13, betting_home=True,
    )
    second = game_leg_from_moneyline(
        event_id="b", label="B ML", both_agree=True, mu_margin=-1, sigma_margin=13, betting_home=False,
    )
    ticket = ParlayTicket(
        legs=(first, second),
        kind="cross-game",
        joint=0.3,
        independent=0.3,
        dk_american=-110,
        edge=0.04,
        ev=0.02,
        status="stake",
        stake=0.5,
        reason="stake: the DraftKings parlay price is +EV",
    )
    offers = offers_from_tickets([ticket])
    assert len(offers) == 1
    assert offers[0].record == "parlays"
    assert offers[0].direction == "decimal"
    assert offers[0].bet_number == round(american_to_decimal(-110), 4)
    assert offers[0].note == "A ML + B ML"
    text = render_parlays([first, second], [], [ticket], log_source="parlays")
    assert "1  $0.50" in text
    assert "python -m nfl_model log parlays --accept 1" in text


def test_one_leg_is_not_a_parlay():
    leg = game_leg_from_moneyline(
        event_id="a", label="a", both_agree=True, mu_margin=0, sigma_margin=13, betting_home=True,
    )
    ticket = price_ticket([leg], 100, Settings(bankroll=1000), 0.0)
    assert ticket.stake == 0
    assert "2 or 3" in ticket.reason
