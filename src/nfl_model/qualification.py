"""Closing-line value and the locked upgrade rule.

A record qualifies for quarter Kelly when it has at least 100 graded bets and
the average closing-line value is above 0. Live never qualifies. Parlays never
qualify; they stay at a flat stake.

Closing-line value is the closing number minus the bet number, oriented so a
positive value means the close moved in the direction of the bet.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

QUALIFY_MIN_BETS = 100
RECORDS = ("sides", "totals", "props", "live", "parlays")
DIRECTIONS = ("over", "under", "spread", "decimal")

LEDGER_FIELDS = ("record", "bet_number", "close_number", "direction", "note")


class LedgerError(ValueError):
    """A ledger row cannot be graded."""


@dataclass(frozen=True)
class RecordStatus:
    record: str
    graded: int
    mean_clv: float | None
    qualified: bool

    @property
    def reason(self) -> str:
        if self.record == "live":
            return "live stays flat; there is no closing line to grade"
        if self.record == "parlays":
            return "parlays stay flat; they do not upgrade to quarter Kelly"
        if self.graded < QUALIFY_MIN_BETS:
            return f"{self.graded} graded bets; {QUALIFY_MIN_BETS} are required"
        if self.mean_clv is None or self.mean_clv <= 0:
            return f"average closing-line value is {self.mean_clv or 0:.3f}; it must be above 0"
        return f"{self.graded} graded bets, average closing-line value {self.mean_clv:.3f}"


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


def grade_record(record: str, clv_values: list[float]) -> RecordStatus:
    if record not in RECORDS:
        raise LedgerError(f"record must be one of: {', '.join(RECORDS)}")
    graded = len(clv_values)
    mean = sum(clv_values) / graded if graded else None
    if record in {"live", "parlays"}:
        qualified = False
    else:
        qualified = graded >= QUALIFY_MIN_BETS and mean is not None and mean > 0
    return RecordStatus(record=record, graded=graded, mean_clv=mean, qualified=qualified)


def pool_name(record: str) -> str:
    """Dollar sleeve a record draws from. Totals share the pregame sleeve."""
    if record == "props":
        return "props"
    if record == "live":
        return "live"
    if record == "parlays":
        return "parlays"
    return "sides"


def load_ledger(path: Path) -> dict[str, list[float]]:
    """Read graded closing-line values. Missing file means every record is empty."""
    grouped = {record: [] for record in RECORDS}
    if not path.exists():
        return grouped
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LedgerError(f"{path} is empty. Expected columns: {', '.join(LEDGER_FIELDS)}")
        missing = [name for name in LEDGER_FIELDS if name not in reader.fieldnames]
        if missing:
            raise LedgerError(f"{path} is missing columns: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            record = (row.get("record") or "").strip()
            close_raw = (row.get("close_number") or "").strip()
            if close_raw == "":
                continue
            if record not in RECORDS:
                raise LedgerError(f"{path}:{line_number} record must be one of: {', '.join(RECORDS)}")
            direction = (row.get("direction") or "").strip()
            try:
                bet_number = float(row["bet_number"])
                close_number = float(close_raw)
            except (TypeError, ValueError) as exc:
                raise LedgerError(f"{path}:{line_number} bet_number and close_number must be numbers") from exc
            grouped[record].append(closing_line_value(bet_number, close_number, direction))
    return grouped


def qualification_report(path: Path) -> list[RecordStatus]:
    grouped = load_ledger(path)
    return [grade_record(record, grouped[record]) for record in RECORDS]


def is_qualified(path: Path, record: str) -> bool:
    grouped = load_ledger(path)
    return grade_record(record, grouped[record]).qualified
