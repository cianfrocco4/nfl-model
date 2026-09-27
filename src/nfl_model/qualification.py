"""Closing-line value and the locked upgrade rule.

A record qualifies for quarter Kelly when it has at least 100 graded bets and
the average closing-line value is above 0. Live never qualifies. Parlays never
qualify; they stay at a flat stake.

Closing-line value is the closing number minus the bet number, oriented so a
positive value means the close moved in the direction of the bet.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from nfl_model.odds_math import american_to_decimal

QUALIFY_MIN_BETS = 100
# A full NFL week is 16 games. Bye weeks are smaller, so this is an upper bound.
GAMES_PER_FULL_WEEK = 16
RECORDS = ("sides", "totals", "props", "live", "parlays")
DIRECTIONS = ("over", "under", "spread", "decimal")

LEDGER_FIELDS = ("record", "bet_number", "close_number", "direction", "note")
MONEY_FIELDS = ("stake", "american", "result")
RESULTS = ("win", "loss", "push")


class LedgerError(ValueError):
    """A ledger row cannot be graded."""


@dataclass(frozen=True)
class RecordStatus:
    record: str
    graded: int
    mean_clv: float | None
    qualified: bool
    ungraded: int = 0

    @property
    def reason(self) -> str:
        if self.record == "live":
            text = "live stays flat; there is no closing line to grade"
        elif self.record == "parlays":
            text = "parlays stay flat; they do not upgrade to quarter Kelly"
        elif self.graded < QUALIFY_MIN_BETS:
            text = f"{self.graded} graded bets; {QUALIFY_MIN_BETS} are required"
        elif self.mean_clv is None or self.mean_clv <= 0:
            text = f"average closing-line value is {self.mean_clv or 0:.3f}; it must be above 0"
        else:
            text = f"{self.graded} graded bets, average closing-line value {self.mean_clv:.3f}"
        if self.ungraded:
            noun = "bet" if self.ungraded == 1 else "bets"
            text += f" {self.ungraded} {noun} listed without a closing number do not count."
        return text


def closing_line_value(bet_number: float, close_number: float, direction: str) -> float:
    """Positive when the close moved in the bettor's favor.

    over: close minus bet (a higher close helps the over).
    under: bet minus close (a lower close helps the under).
    spread: bet point minus close point. The point is the one attached to the
    side you bet, so -3.5 is better than a close of -6.5.
    decimal: bet decimal odds minus close decimal odds (a richer price is better).
    """
    if direction == "over":
        return close_number - bet_number
    if direction == "under":
        return bet_number - close_number
    if direction in {"spread", "decimal"}:
        return bet_number - close_number
    raise LedgerError(f"direction must be one of: {', '.join(DIRECTIONS)}")


def grade_record(record: str, clv_values: list[float], ungraded: int = 0) -> RecordStatus:
    if record not in RECORDS:
        raise LedgerError(f"record must be one of: {', '.join(RECORDS)}")
    graded = len(clv_values)
    mean = sum(clv_values) / graded if graded else None
    if record in {"live", "parlays"}:
        qualified = False
    else:
        qualified = graded >= QUALIFY_MIN_BETS and mean is not None and mean > 0
    return RecordStatus(record=record, graded=graded, mean_clv=mean, qualified=qualified, ungraded=ungraded)


def pool_name(record: str) -> str:
    """Dollar sleeve a record draws from. Totals share the pregame sleeve."""
    if record == "props":
        return "props"
    if record == "live":
        return "live"
    if record == "parlays":
        return "parlays"
    return "sides"


def load_ledger(path: Path) -> tuple[dict[str, list[float]], dict[str, int]]:
    """Read graded closing-line values and a count of rows still waiting on a close.

    A missing file means every record is empty. A blank close_number is kept
    as ungraded so the qualification report can say it does not count yet.
    """
    grouped = {record: [] for record in RECORDS}
    ungraded = {record: 0 for record in RECORDS}
    if not path.exists():
        return grouped, ungraded
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LedgerError(f"{path} is empty. Expected columns: {', '.join(LEDGER_FIELDS)}")
        missing = [name for name in LEDGER_FIELDS if name not in reader.fieldnames]
        if missing:
            raise LedgerError(f"{path} is missing columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            record = (row.get("record") or "").strip()
            if record not in RECORDS:
                raise LedgerError(f"{path}:{line_number} record must be one of: {', '.join(RECORDS)}")
            close_raw = (row.get("close_number") or "").strip()
            if close_raw == "":
                ungraded[record] += 1
                continue
            direction = (row.get("direction") or "").strip()
            try:
                bet_number = float(row["bet_number"])
                close_number = float(close_raw)
            except (TypeError, ValueError) as exc:
                raise LedgerError(f"{path}:{line_number} bet_number and close_number must be numbers") from exc
            grouped[record].append(closing_line_value(bet_number, close_number, direction))
    return grouped, ungraded


def qualification_report(path: Path) -> list[RecordStatus]:
    grouped, ungraded = load_ledger(path)
    return [grade_record(record, grouped[record], ungraded[record]) for record in RECORDS]


def is_qualified(path: Path, record: str) -> bool:
    grouped, ungraded = load_ledger(path)
    return grade_record(record, grouped[record], ungraded[record]).qualified


@dataclass(frozen=True)
class SeasonMoney:
    """Settled profit from ledger rows that record a stake and a result.

    Rows with a stake and no result are open bets. They are not in the profit.
    A file without stake, american, and result columns does not track money.
    """

    settled: int
    unsettled: int
    profit: float
    wins: int
    losses: int
    pushes: int
    tracks_money: bool


def bet_profit(stake: float, american: int | None, result: str) -> float:
    """Dollars won or lost on one settled bet. A win needs American odds."""
    if stake <= 0:
        raise LedgerError("stake must be a positive number of dollars")
    if result == "push":
        return 0.0
    if result == "loss":
        return -stake
    if result == "win":
        if american is None:
            raise LedgerError("a win needs american odds so the profit can be calculated")
        return stake * (american_to_decimal(american) - 1.0)
    raise LedgerError(f"result must be one of: {', '.join(RESULTS)}")


def season_money(path: Path) -> SeasonMoney:
    """Sum settled profit. A missing file, or a file with no money columns, is $0."""
    empty = SeasonMoney(0, 0, 0.0, 0, 0, 0, False)
    if not path.exists():
        return empty
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LedgerError(f"{path} is empty. Expected columns: {', '.join(LEDGER_FIELDS)}")
        if not any(name in reader.fieldnames for name in MONEY_FIELDS):
            return empty
        settled = unsettled = wins = losses = pushes = 0
        profit = 0.0
        for line_number, row in enumerate(reader, start=2):
            stake_raw = (row.get("stake") or "").strip()
            american_raw = (row.get("american") or "").strip()
            result = (row.get("result") or "").strip().lower()
            if stake_raw == "" and result == "":
                continue
            if result and result not in RESULTS:
                raise LedgerError(f"{path}:{line_number} result must be one of: {', '.join(RESULTS)}")
            if result and stake_raw == "":
                raise LedgerError(f"{path}:{line_number} a {result} needs a stake")
            try:
                stake = float(stake_raw) if stake_raw else 0.0
            except ValueError as exc:
                raise LedgerError(f"{path}:{line_number} stake must be a number of dollars") from exc
            american = _parse_american(path, line_number, american_raw) if american_raw else None
            if result == "":
                if stake <= 0:
                    raise LedgerError(f"{path}:{line_number} stake must be a positive number of dollars")
                unsettled += 1
                continue
            if result == "win" and american is None:
                raise LedgerError(f"{path}:{line_number} a win needs american odds")
            profit += bet_profit(stake, american, result)
            settled += 1
            if result == "win":
                wins += 1
            elif result == "loss":
                losses += 1
            else:
                pushes += 1
    return SeasonMoney(settled, unsettled, profit, wins, losses, pushes, True)


def loss_limit_reached(money: SeasonMoney, bankroll: float | None, fraction: float) -> bool:
    """True when settled losses have used up the season stop."""
    if not money.tracks_money or bankroll is None or bankroll <= 0:
        return False
    return money.profit <= -bankroll * fraction


def season_stop_note(money: SeasonMoney, bankroll: float | None, fraction: float) -> str | None:
    if not loss_limit_reached(money, bankroll, fraction):
        return None
    assert bankroll is not None
    limit = bankroll * fraction
    return (
        f"Season stop: settled profit is ${money.profit:,.2f}, "
        f"at or below -${limit:,.2f}. No new stake."
    )


@dataclass(frozen=True)
class ProfitGoal:
    """A dollar target and a deadline. It is a progress check, not a stake size."""

    return_fraction: float
    weeks: float
    start: date

    @property
    def deadline(self) -> date:
        return self.start + timedelta(days=self.weeks * 7)


def make_profit_goal(
    return_fraction: float | None,
    weeks: float | None,
    start: date | None,
) -> ProfitGoal | None:
    if return_fraction is None or weeks is None or start is None:
        return None
    return ProfitGoal(return_fraction, weeks, start)


def season_progress(
    money: SeasonMoney,
    bankroll: float | None,
    fraction: float,
    goal: ProfitGoal | None = None,
    today: date | None = None,
    flat_stake: float | None = None,
    min_edge: float = 0.02,
) -> str:
    """Season stop, plus the profit goal when one is set."""
    if bankroll is None:
        text = "Season goal: set a bankroll so the loss stop can be measured."
        if goal is not None:
            text += " The profit goal uses that same bankroll."
        return text
    limit = bankroll * fraction
    if not money.tracks_money:
        text = (
            f"Season through the postseason: keep every bet in the ledger, "
            f"and do not lose more than ${limit:,.2f}. "
            "Add stake, american, and result to ledger.csv so that stop can be checked."
        )
    else:
        note = season_stop_note(money, bankroll, fraction)
        if note is not None:
            text = note
        else:
            text = (
                f"Season through the postseason: settled profit ${money.profit:,.2f} "
                f"on {money.settled} bets ({money.unsettled} still open). "
                f"Stop at -${limit:,.2f}."
            )
    if goal is None:
        return text
    clock = today or date.today()
    return text + "\n" + _goal_lines(money, bankroll, goal, clock, flat_stake, min_edge)


def _goal_lines(
    money: SeasonMoney,
    bankroll: float,
    goal: ProfitGoal,
    today: date,
    flat_stake: float | None,
    min_edge: float,
) -> str:
    deadline = goal.deadline
    target = bankroll * goal.return_fraction
    if today < goal.start:
        status = f"The window starts {goal.start:%b %d}."
    elif today > deadline:
        if money.profit >= target:
            status = f"The window closed on {deadline:%b %d} after settled profit reached ${target:,.2f}."
        else:
            status = (
                f"The window closed on {deadline:%b %d}. "
                f"Settled profit is ${money.profit:,.2f} against ${target:,.2f}."
            )
    elif money.profit >= target:
        days_left = max((deadline - today).days, 0)
        status = (
            f"Settled profit ${money.profit:,.2f} has reached ${target:,.2f}. "
            f"{days_left} days left in the window."
        )
    else:
        share = money.profit / target if target else 0.0
        days_left = max((deadline - today).days, 0)
        status = f"Settled profit ${money.profit:,.2f} is {share:.0%} of ${target:,.2f}. {days_left} days left."
    lines = [f"Goal: +{goal.return_fraction:.0%} (${target:,.2f}) by {deadline:%b %d}. {status}"]
    if flat_stake is not None and flat_stake > 0 and min_edge > 0 and target > 0:
        expected = min_edge * flat_stake
        needed = math.ceil(target / expected)
        capacity = int(goal.weeks * GAMES_PER_FULL_WEEK)
        reach = (
            f"At a flat stake of ${flat_stake:,.2f} and an edge of {min_edge:.1%}, "
            f"one bet expects at most ${expected:,.2f}. "
            f"Reaching ${target:,.2f} takes at least {needed} bets at that edge, before vig."
        )
        if needed > capacity:
            reach += (
                f" A {goal.weeks:g}-week slate is at most {capacity} games. "
                "The stake stays on the locked rule."
            )
        else:
            reach += " The stake stays on the locked rule."
        lines.append(reach)
    return "\n".join(lines)


def _parse_american(path: Path, line_number: int, raw: str) -> int:
    try:
        value = float(raw.replace("+", ""))
    except ValueError as exc:
        raise LedgerError(f"{path}:{line_number} american odds must be a number such as -110 or +150") from exc
    if value == 0 or not value.is_integer():
        raise LedgerError(f"{path}:{line_number} american odds must be a whole number such as -110 or +150")
    return int(value)
