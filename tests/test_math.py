"""Core prices, staking, qualification, and parlay joints."""

import math

from nfl_model.odds_math import (
    american_to_decimal,
    decimal_to_american,
    edge,
    expected_value,
    implied_probability,
    remove_vig,
)
from nfl_model.parlay import (
    bivariate_cdf,
    game_leg_from_moneyline,
    game_leg_from_spread,
    game_leg_from_total,
    independent_probability,
    joint_probability,
    price_ticket,
)
from nfl_model.probabilities import moneyline_outcomes, spread_outcomes
from nfl_model.qualification import closing_line_value, grade_record
from nfl_model.recommend import decide
from nfl_model.settings import Settings, StakingMethod
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


def test_one_leg_is_not_a_parlay():
    leg = game_leg_from_moneyline(
        event_id="a", label="a", both_agree=True, mu_margin=0, sigma_margin=13, betting_home=True,
    )
    ticket = price_ticket([leg], 100, Settings(bankroll=1000), 0.0)
    assert ticket.stake == 0
    assert "2 or 3" in ticket.reason
