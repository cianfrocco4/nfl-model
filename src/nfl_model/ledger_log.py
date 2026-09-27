"""Write accepted stakes into the ledger.

`board`, `props`, `live`, and `parlays` save the numbered stakes they just
printed. `log` appends only the numbers you accept. Close and result stay
blank. A row that is already open is left as it is.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from nfl_model.data_nflverse import project_root
from nfl_model.odds_math import american_to_decimal, format_american
from nfl_model.parlay import ParlayTicket
from nfl_model.qualification import DIRECTIONS, LEDGER_FIELDS, MONEY_FIELDS, RECORDS
from nfl_model.recommend import Decision

SOURCES = ("board", "props", "live", "parlays")
LEDGER_COLUMNS = LEDGER_FIELDS + MONEY_FIELDS


class LedgerLogError(ValueError):
    """An accepted stake cannot be saved or written."""


@dataclass(frozen=True)
class StakeOffer:
    number: int
    record: str
    bet_number: float
    direction: str
    note: str
    stake: float
    american: int


@dataclass(frozen=True)
class SkippedStake:
    offer: StakeOffer
    ledger_american: int


@dataclass(frozen=True)
class LogResult:
    ledger: Path
    written: tuple[StakeOffer, ...]
    skipped: tuple[SkippedStake, ...]


def offers_path(source: str, root: Path | None = None) -> Path:
    """Where one command stores the stake list it just printed."""
    if source not in SOURCES:
        raise LedgerLogError(f"source must be one of: {', '.join(SOURCES)}")
    base = project_root() if root is None else root
    return base / ".cache" / "stake-offers" / f"{source}.json"


def offers_from_decisions(rows: list[Decision]) -> list[StakeOffer]:
    """Number the same stake rows the board print numbers, starting at 1."""
    offers: list[StakeOffer] = []
    for row in rows:
        if row.stake <= 0:
            continue
        stake = round(float(row.stake), 2)
        if (
            row.bet_number is None
            or row.direction not in DIRECTIONS
            or row.record not in RECORDS
            or stake <= 0
            or row.dk_american == 0
        ):
            raise LedgerLogError(
                f"Cannot log {row.game} {row.side}: it has a stake and no ledger number."
            )
        offers.append(
            StakeOffer(
                number=len(offers) + 1,
                record=row.record,
                bet_number=float(row.bet_number),
                direction=row.direction,
                note=f"{row.game} {row.side}".strip(),
                stake=stake,
                american=int(row.dk_american),
            )
        )
    return offers


def offers_from_tickets(tickets: list[ParlayTicket]) -> list[StakeOffer]:
    """Number priced parlay slips that still have a stake."""
    offers: list[StakeOffer] = []
    for ticket in tickets:
        if ticket.stake <= 0 or ticket.dk_american is None:
            continue
        stake = round(float(ticket.stake), 2)
        if stake <= 0 or ticket.dk_american == 0:
            raise LedgerLogError("Cannot log a parlay slip with no stake or no American price.")
        offers.append(
            StakeOffer(
                number=len(offers) + 1,
                record="parlays",
                bet_number=round(american_to_decimal(ticket.dk_american), 4),
                direction="decimal",
                note=" + ".join(leg.label for leg in ticket.legs),
                stake=stake,
                american=int(ticket.dk_american),
            )
        )
    return offers


def save_offers(path: Path, offers: list[StakeOffer]) -> None:
    """Replace the saved list. The next run of that command does this again."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(offer) for offer in offers]
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_offers(path: Path) -> list[StakeOffer]:
    source = path.stem
    if not path.is_file():
        raise LedgerLogError(
            f"No saved {source} stakes. "
            f"Run `python -m nfl_model {source}` and log the numbers you place "
            "before the next run replaces the list."
        )
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise LedgerLogError(f"Saved {source} stakes are unreadable. Run that command again.") from exc
    if not isinstance(payload, list):
        raise LedgerLogError(f"Saved {source} stakes are unreadable. Run that command again.")
    offers = [_offer_from_json(item, source) for item in payload]
    numbers = [offer.number for offer in offers]
    if len(numbers) != len(set(numbers)):
        raise LedgerLogError(f"Saved {source} stakes repeat a number. Run that command again.")
    return offers


def parse_accept(text: str) -> list[int]:
    """Stake numbers from `--accept 1,3`. Repeated numbers count once."""
    parts = [part.strip() for part in text.split(",")]
    if not parts or any(part == "" for part in parts):
        raise LedgerLogError("--accept needs stake numbers such as 1,3.")
    numbers: list[int] = []
    for part in parts:
        if not part.isdigit() or int(part) < 1:
            raise LedgerLogError(f"--accept needs stake numbers such as 1,3. Got {text!r}.")
        number = int(part)
        if number not in numbers:
            numbers.append(number)
    return numbers


def parse_price(text: str) -> tuple[int, int]:
    """One `--price 1:-108` override. The sign on a plus price is optional."""
    if text.count(":") != 1:
        raise LedgerLogError("A price looks like 1:-108.")
    left, right = text.split(":")
    left = left.strip()
    right = right.strip().replace("+", "")
    if not left.isdigit() or int(left) < 1 or right in {"", "-"}:
        raise LedgerLogError("A price looks like 1:-108.")
    try:
        american = int(right)
    except ValueError as exc:
        raise LedgerLogError("A price looks like 1:-108.") from exc
    if american == 0:
        raise LedgerLogError("American odds of 0 are not a price.")
    return int(left), american


def append_accepted(
    ledger: Path,
    offers: list[StakeOffer],
    accept: list[int],
    prices: dict[int, int],
) -> LogResult:
    """Append accepted offers. Close and result stay blank.

    A spread, total, or prop keeps its line when the American price changes.
    A decimal bet (moneyline, anytime touchdown, parlay) takes its bet number
    from the new price. An open row with the same record, direction, note, and
    bet number is not written again.
    """
    by_number = {offer.number: offer for offer in offers}
    unknown = [number for number in accept if number not in by_number]
    if unknown:
        available = ", ".join(str(offer.number) for offer in offers) or "none"
        listed = ", ".join(str(number) for number in unknown)
        raise LedgerLogError(
            f"No saved stake numbered {listed}. This list has: {available}. "
            "Run the source command again if the board changed."
        )
    extra = [number for number in prices if number not in accept]
    if extra:
        listed = ", ".join(str(number) for number in extra)
        raise LedgerLogError(
            f"--price {listed} is not in --accept. Pass that number in --accept, or drop the price."
        )
    chosen: list[StakeOffer] = []
    for number in accept:
        offer = by_number[number]
        if number in prices:
            offer = _with_price(offer, prices[number])
        chosen.append(offer)

    fieldnames, existing = _prepare_ledger(ledger)
    seen = _open_prices(existing)
    written: list[StakeOffer] = []
    skipped: list[SkippedStake] = []
    for offer in chosen:
        key = _offer_key(offer)
        if key in seen:
            skipped.append(SkippedStake(offer, seen[key]))
            continue
        seen[key] = offer.american
        written.append(offer)
    if written:
        _append_rows(ledger, fieldnames, [_ledger_row(offer) for offer in written])
    return LogResult(ledger, tuple(written), tuple(skipped))


def log_result_text(result: LogResult) -> str:
    lines: list[str] = []
    if result.written:
        noun = "stake" if len(result.written) == 1 else "stakes"
        lines.append(f"Logged {len(result.written)} {noun} in {result.ledger}. Close and result are blank.")
        lines.extend(_offer_line(offer) for offer in result.written)
    else:
        lines.append(f"No new rows in {result.ledger}.")
    if result.skipped:
        lines.append("Already open, so these were left unchanged. Edit the ledger to change a price:")
        lines.extend(_skipped_line(item) for item in result.skipped)
    lines.append("Fill close_number when DraftKings closes the number, and result when it settles.")
    return "\n".join(lines)


def format_bet_number(value: float) -> str:
    rounded = round(float(value), 4)
    text = f"{rounded:.4f}".rstrip("0").rstrip(".")
    if text in {"", "-", "-0"}:
        return "0"
    return text


def _with_price(offer: StakeOffer, american: int) -> StakeOffer:
    bet_number = offer.bet_number
    if offer.direction == "decimal":
        bet_number = round(american_to_decimal(american), 4)
    return replace(offer, american=american, bet_number=bet_number)


def _offer_from_json(item: object, source: str) -> StakeOffer:
    if not isinstance(item, dict):
        raise LedgerLogError(f"Saved {source} stakes are unreadable. Run that command again.")
    try:
        offer = StakeOffer(
            number=int(item["number"]),
            record=str(item["record"]),
            bet_number=float(item["bet_number"]),
            direction=str(item["direction"]),
            note=str(item["note"]),
            stake=float(item["stake"]),
            american=int(item["american"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise LedgerLogError(f"Saved {source} stakes are unreadable. Run that command again.") from exc
    if (
        offer.number < 1
        or offer.record not in RECORDS
        or offer.direction not in DIRECTIONS
        or offer.stake <= 0
        or offer.american == 0
    ):
        raise LedgerLogError(f"Saved {source} stakes are unreadable. Run that command again.")
    return offer


def _prepare_ledger(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as handle:
            csv.writer(handle, lineterminator="\n").writerow(LEDGER_COLUMNS)
        return list(LEDGER_COLUMNS), []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise LedgerLogError(f"{path} is empty. Expected columns: {', '.join(LEDGER_COLUMNS)}")
        fieldnames = list(reader.fieldnames)
        missing = [name for name in LEDGER_COLUMNS if name not in fieldnames]
        if missing:
            raise LedgerLogError(
                f"{path} is missing columns: {', '.join(missing)}. "
                "Add them before logging, or point --ledger at a new file."
            )
        return fieldnames, list(reader)


def _append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    needs_newline = False
    if path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) not in {b"\n", b"\r"}
    with path.open("a", newline="") as handle:
        if needs_newline:
            handle.write("\n")
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writerows(rows)


def _ledger_row(offer: StakeOffer) -> dict[str, str]:
    return {
        "record": offer.record,
        "bet_number": format_bet_number(offer.bet_number),
        "close_number": "",
        "direction": offer.direction,
        "note": offer.note,
        "stake": f"{offer.stake:.2f}",
        "american": str(offer.american),
        "result": "",
    }


def _open_prices(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, float], int]:
    prices: dict[tuple[str, str, str, float], int] = {}
    for row in rows:
        result = (row.get("result") or "").strip()
        stake = (row.get("stake") or "").strip()
        if result or stake == "":
            continue
        raw_number = (row.get("bet_number") or "").strip()
        try:
            bet_number = round(float(raw_number), 4)
        except ValueError:
            continue
        raw_american = (row.get("american") or "").strip().replace("+", "")
        try:
            american = int(raw_american)
        except ValueError:
            american = 0
        prices[
            (
                (row.get("record") or "").strip(),
                (row.get("direction") or "").strip(),
                (row.get("note") or "").strip(),
                bet_number,
            )
        ] = american
    return prices


def _offer_key(offer: StakeOffer) -> tuple[str, str, str, float]:
    return (offer.record, offer.direction, offer.note.strip(), round(float(offer.bet_number), 4))


def _offer_line(offer: StakeOffer) -> str:
    return (
        f"  {offer.number}  {offer.record}  {format_bet_number(offer.bet_number)}  "
        f"{offer.direction}  ${offer.stake:,.2f}  {format_american(offer.american)}  {offer.note}"
    )


def _skipped_line(item: SkippedStake) -> str:
    line = _offer_line(item.offer)
    if item.ledger_american != item.offer.american and item.ledger_american != 0:
        line += f"  (ledger still has {format_american(item.ledger_american)})"
    return line
